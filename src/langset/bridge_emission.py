"""Parallel-query continuous emission for multi-latent models.

`QueryBridgeEmission` uses learned query vectors to cross-attend the backbone's token-level hidden states. Each
query produces an L2-normalized latent vector and a validity logit. During training, Hungarian matching assigns
query slots to target latents, and the objective combines an InfoNCE loss over matched vectors with binary
cross-entropy over validity logits.

A typical configuration uses `QueryBridgeEmission` with `FrozenEncoderTarget`, `freeze_backbone=True`, and
`lam_multi_nce=0`. The separate multi-latent NCE term should be disabled because this objective already includes
a contrastive loss. `n_queries` controls the maximum number of vectors emitted per input.

When `hard_neg_field` is configured, the corresponding texts are encoded and added to the InfoNCE target bank as
additional negatives. At inference time, the bridge emits query slots whose validity probability exceeds 0.5,
subject to `max_steps`, with a one-slot fallback when none pass the threshold.

Freezing the backbone keeps its existing embedding parameters unchanged; this emission strategy does not freeze
the backbone automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from langset.loss import BridgeLossContext, bridge_loss
from langset.strategies import EmissionOut, _EmissionObjective, _TargetSource

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from langset.modeling import LangSetModel
    from langset.trainer import Trainer
    from langset.training_args import TrainingArguments


class FrozenEncoderTarget(_TargetSource):
    """Use the model's normalized text embeddings as target latents.

    `encode()` evaluates target texts with the same `LangSetModel` used for training and does not track gradients.
    The target source has no EMA copy: `twin` refers to the model itself and `update()` is a no-op.

    Use `freeze_backbone=True` when the target embedding space must remain fixed. This target source does not freeze
    model parameters itself.
    """

    suppresses_nce = False

    def __init__(
        self,
        model: "LangSetModel",
        args: "TrainingArguments",
        tok: "PreTrainedTokenizerBase",
        dev: torch.device,
    ) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = model  # Evaluation encodes the retrieval bank with the same model.

    def encode(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            z = self.m.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return z.to(self.dev).float()  # ty: ignore[unresolved-attribute]  # Tensor when convert_to_numpy=False

    def update(self) -> None:  # This target source has no EMA state to update.
        pass


def _heads_for(d: int) -> int:
    for h in (8, 4, 2, 1):
        if d % h == 0:
            return h
    return 1


class QueryBridge(nn.Module):
    """Decode learned queries against token-level hidden states.

    The module returns one L2-normalized vector and one validity logit for each query slot.
    """

    def __init__(self, d: int, n_queries: int, n_layers: int = 2) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, _heads_for(d), 4 * d, batch_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d, d)
        self.valid = nn.Linear(d, 1)

    def forward(
        self, substrate: torch.Tensor, smask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.queries.unsqueeze(0).expand(substrate.size(0), -1, -1)
        q = self.dec(q, substrate, memory_key_padding_mask=~smask)
        return F.normalize(self.out(q), dim=-1), self.valid(q).squeeze(-1)


class QueryBridgeEmission(_EmissionObjective):
    """Train and infer a set of continuous latent vectors in one decoder pass.

    This objective uses no discrete codebook and implements non-autoregressive inference through `emit_infer()`.
    """

    codebook = False

    def __init__(
        self,
        model: "LangSetModel",
        args: "TrainingArguments",
        dev: torch.device,
        trainer: "Trainer",
    ) -> None:
        super().__init__(model, args, dev, trainer)
        try:  # SciPy is required only for Hungarian matching in this emission strategy.
            from scipy.optimize import linear_sum_assignment
        except ModuleNotFoundError as e:  # pragma: no cover - trivial guard
            raise ModuleNotFoundError(
                "QueryBridgeEmission needs SciPy for DETR Hungarian matching. Install it with "
                "`pip install scipy` (or `pip install langset[bridge]`)."
            ) from e

        self._match = linear_sum_assignment
        d = int(model.h)
        self.n_queries = int(getattr(args, "n_queries", 16))
        self.temp = float(getattr(args, "tau", 0.05))
        self.lam_valid = float(getattr(args, "bridge_lam_valid", 1.0))
        self.pos_weight = float(getattr(args, "bridge_pos_weight", 2.0))
        bridge = QueryBridge(d, self.n_queries).to(dev)
        # Register the bridge on the model so optimization and checkpoint persistence include it.
        model.add_module("emission_bridge", bridge)
        pending = getattr(model, "_emission_bridge_state", None)
        if (
            pending is not None
        ):  # `LangSetModel.load()` stages persisted weights until this strategy is attached.
            bridge.load_state_dict(pending)
        self.bridge = bridge

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.bridge.parameters()

    def _hard_negs(self, bidx: list[int]) -> list[str]:
        """Collect nonblank hard-negative texts for the selected dataset rows.

        Returns an empty list when no trainer hard-negative column is available.
        """
        hn = getattr(self.trainer, "hard_neg_texts", None)
        if not hn:
            return []
        return [t for k in bidx for t in hn[k] if t and t.strip()]

    def emit(
        self,
        se: dict[str, torch.Tensor],
        target_lat: torch.Tensor,
        valid: torch.Tensor,
        lens_l: list[int],
        bidx: list[int],
        b: int,
        lmax: int,
        ep: int,
        ss_mask: Optional[torch.Tensor] = None,
    ) -> EmissionOut:
        m, dev = self.m, self.dev
        d = target_lat.size(-1)
        ids, mask = se["input_ids"], se["attention_mask"]
        # Read unpooled token-level hidden states from the backbone.
        substrate = m._last_hidden(m._run_backbone(m.embed(ids), mask, ids, 0)).float()  # [B, T, d]
        vecs, vlog = self.bridge(substrate, mask.bool())  # [B, nq, d], [B, nq]

        tgt = F.normalize(target_lat.float(), dim=-1)  # [B, lmax, d]
        bank = tgt[
            valid
        ]  # Valid targets lead the InfoNCE bank; positive indices use this ordering.
        # Encode configured hard negatives in the target embedding space and append them to the InfoNCE denominator.
        # They are negatives only, so positive indices remain valid.
        hn_texts = self._hard_negs(bidx)
        if hn_texts:
            with torch.no_grad():
                hn = self.m.encode(hn_texts, convert_to_numpy=False, normalize_embeddings=True)
            bank = torch.cat([bank, hn.to(dev).float()], dim=0)  # ty: ignore[unresolved-attribute]  # encode returns a Tensor
        recon = torch.zeros(b, lmax, d, device=dev)
        vlab = torch.zeros_like(vlog)
        matched_pred: list[torch.Tensor] = []
        pos: list[int] = []
        nq = vecs.size(1)  # Maximum number of targets that can be matched in each row.
        # Transfer all query-target similarities to CPU once, then run SciPy matching per row.
        sims = torch.bmm(vecs, tgt.transpose(1, 2)).detach().float().cpu().numpy()  # [B, nq, lmax]
        off = 0
        for r in range(b):
            mi = lens_l[r]
            if mi == 0:
                continue
            mi_eff = min(
                mi, nq
            )  # At most `nq` targets can be matched; additional targets remain negatives.
            pr, tc = self._match(-sims[r, :, :mi_eff])  # Maximize cosine similarity.
            for p, c in zip(pr, tc):
                recon[r, c] = vecs[r, p]  # Place the matched prediction in its target-aligned slot.
                matched_pred.append(vecs[r, p])
                pos.append(off + int(c))
                vlab[r, p] = 1.0
            off += mi

        matched_predictions = torch.stack(matched_pred) if matched_pred else vecs.new_empty((0, d))
        positive_indices = torch.tensor(pos, device=dev)
        losses = bridge_loss(
            BridgeLossContext(
                matched_predictions=matched_predictions,
                target_bank=bank,
                positive_indices=positive_indices,
                validity_logits=vlog,
                validity_labels=vlab,
                temperature=self.temp,
                validity_weight=self.lam_valid,
                positive_weight=self.pos_weight,
            )
        )
        nce = losses.info_nce.to_tensor()
        vloss = losses.validity.to_tensor()
        base = losses.to_tensor()
        zero = target_lat.new_zeros(())
        return EmissionOut(
            recon=recon,
            base_loss=base,
            # Preserve canonical fields: `recon_loss` is InfoNCE, `loss_stop` is validity BCE, and `loss_dims` is unused.
            logs={"loss_stop": vloss.detach(), "loss_dims": zero, "recon_loss": nce.detach()},
            code_logits=None,
        )

    @torch.no_grad()
    def emit_infer(self, texts: list[str], max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit validity-gated latent vectors for a batch of texts.

        The bridge runs once per batch and keeps query slots whose validity probability exceeds 0.5. If no slot
        passes the threshold, the highest-logit slot is retained. At most `max_steps` vectors are returned per input.

        Returns:
            A pair `(latents, lengths)`. `latents` has shape `[batch, max_emitted, latent_dim]` and is zero-padded;
            `lengths` contains the number of retained vectors for each input.
        """
        m, dev = self.m, self.dev
        e = m.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.a.max_len,  # Match the truncation length used during training.
            padding_side="left",
            return_tensors="pt",
        ).to(dev)
        ids, mask = e["input_ids"], e["attention_mask"]
        substrate = m._last_hidden(m._run_backbone(m.embed(ids), mask, ids, 0)).float()
        vecs, vlog = self.bridge(substrate, mask.bool())  # [B, nq, d], [B, nq]
        keep = vlog.sigmoid() > 0.5
        rows: list[torch.Tensor] = []
        lens: list[int] = []
        for r in range(vecs.size(0)):
            idx = keep[r].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:  # Always retain the most confident slot.
                idx = vlog[r].argmax().unsqueeze(0)
            idx = idx[:max_steps]
            rows.append(vecs[r, idx])
            lens.append(int(idx.numel()))
        lmax = max(lens)
        lat = torch.zeros(vecs.size(0), lmax, vecs.size(-1), device=dev)
        for r, v in enumerate(rows):
            lat[r, : v.size(0)] = v
        return lat, torch.tensor(lens, device=dev)
