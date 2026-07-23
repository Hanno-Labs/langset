"""QueryBridgeEmission — a NON-autoregressive, continuous, parallel-query emission family, ported from the
validated proof (ii3 `bridge_lightning.py`) into langset's `_EmissionObjective` seam.

One forward: N learned query tokens cross-attend the (frozen) backbone's per-token hidden states → N vectors +
a validity/count logit each. Trained by DETR Hungarian matching to the row's target latents, then per-fact
InfoNCE (cross-row negatives) + validity BCE. It reuses the widened seams: `build_targets` (default positional
teacher-forcing supplies the target latents), the objective-before-optimizer order (its module registers on the
model, so `m.parameters()` trains it), and the canonical `EmissionOut` log keys.

Inject with `TrainingArguments(emission=QueryBridgeEmission, freeze_backbone=True)`. Pair with a query/frozen
`_TargetSource` and set `lam_multi_nce=0` (its base loss already does the contrastive term). `n_queries` (default
16) is read off args via getattr — no TrainingArguments change required.

Retrieval is preserved BY CONSTRUCTION: the backbone is frozen, so the base embedder's geometry is untouched;
the bridge is a pure add-on. `emit_infer`/eval delegation (so `model.rollout()` uses one pass instead of the AR
rollout) is the remaining follow-up — see EMISSION_PROTOCOL.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from langset.strategies import EmissionOut, _EmissionObjective, _TargetSource

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from langset.modeling import LangSetModel
    from langset.trainer import Trainer
    from langset.training_args import TrainingArguments


class FrozenEncoderTarget(_TargetSource):
    """Targets = the FROZEN encoder's OWN embedding of the target texts, i.e. E(text). Pair with
    `target_texts = queries` to reproduce the 'query-target' objective (emitted vectors trained to match E(query),
    which is exactly what the bench scores). No EMA twin and no self-distillation — the frozen model IS the target,
    so `twin = model` and `update` is a no-op. Retrieval geometry preserved by construction."""

    suppresses_nce = False

    def __init__(
        self, model: "LangSetModel", args: "TrainingArguments", tok: "PreTrainedTokenizerBase", dev: torch.device
    ) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = model  # eval encodes its retrieval bank with the frozen model itself

    def encode(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            z = self.m.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return z.to(self.dev).float()  # [n, d] L2-normalized frozen-encoder embeddings

    def update(self) -> None:  # nothing to track — the target model is frozen
        pass


def _heads_for(d: int) -> int:
    for h in (8, 4, 2, 1):
        if d % h == 0:
            return h
    return 1


class QueryBridge(nn.Module):
    """N learned queries cross-attend the frozen substrate -> N L2-normalized vectors + per-query validity logit."""

    def __init__(self, d: int, n_queries: int, n_layers: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, _heads_for(d), 4 * d, batch_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d, d)
        self.valid = nn.Linear(d, 1)

    def forward(self, substrate: torch.Tensor, smask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.queries.unsqueeze(0).expand(substrate.size(0), -1, -1)
        q = self.dec(q, substrate, memory_key_padding_mask=~smask)
        return F.normalize(self.out(q), dim=-1), self.valid(q).squeeze(-1)


class QueryBridgeEmission(_EmissionObjective):
    """Parallel-query continuous emission (see module docstring). `codebook=False`: no FSQ digits, no AR rollout."""

    codebook = False

    def __init__(
        self, model: "LangSetModel", args: "TrainingArguments", dev: torch.device, trainer: "Trainer"
    ) -> None:
        super().__init__(model, args, dev, trainer)
        from scipy.optimize import linear_sum_assignment  # local: optional dep, only this family needs it

        self._match = linear_sum_assignment
        d = int(model.h)
        self.n_queries = int(getattr(args, "n_queries", 16))
        self.temp = float(getattr(args, "tau", 0.05))
        self.lam_valid = float(getattr(args, "bridge_lam_valid", 1.0))
        self.pos_weight = float(getattr(args, "bridge_pos_weight", 2.0))
        bridge = QueryBridge(d, self.n_queries).to(dev)
        # Register on the model so `m.parameters()` (built AFTER the objective now) trains it. Persistence via the
        # aux-head plug is a follow-up; for now this makes the module optimizer-visible.
        model.add_module("emission_bridge", bridge)
        pending = getattr(model, "_emission_bridge_state", None)
        if pending is not None:  # restore weights persisted by save_pretrained (serve-time / resumed run)
            bridge.load_state_dict(pending)
        self.bridge = bridge

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.bridge.parameters()

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
        # per-token substrate from the (frozen when freeze_backbone) backbone — same read seed_hidden does, un-pooled
        substrate = m._last_hidden(m._run_backbone(m.embed(ids), mask, ids, 0)).float()  # [B, T, d]
        vecs, vlog = self.bridge(substrate, mask.bool())  # [B, nq, d], [B, nq]

        tgt = F.normalize(target_lat.float(), dim=-1)  # [B, lmax, d]
        bank = tgt[valid]  # [Ntot, d] — cross-row InfoNCE negatives
        recon = torch.zeros(b, lmax, d, device=dev)
        vlab = torch.zeros_like(vlog)
        matched_pred: list[torch.Tensor] = []
        pos: list[int] = []
        off = 0
        for r in range(b):
            mi = lens_l[r]
            if mi == 0:
                continue
            t_i = tgt[r, :mi]  # [mi, d]
            cost = -(vecs[r] @ t_i.T).detach().float().cpu().numpy()  # [nq, mi]
            pr, tc = self._match(cost)  # mi matches (nq >= mi)
            for p, c in zip(pr, tc):
                recon[r, c] = vecs[r, p]  # matched emission -> its target's slot (aligns recon with `valid`)
                matched_pred.append(vecs[r, p])
                pos.append(off + int(c))
                vlab[r, p] = 1.0
            off += mi

        if matched_pred:
            mp = torch.stack(matched_pred)  # [K, d]
            pos_t = torch.tensor(pos, device=dev)
            nce = F.cross_entropy(mp @ bank.T / self.temp, pos_t)  # each emission retrieves its own target
        else:
            nce = target_lat.new_zeros(())
        vloss = F.binary_cross_entropy_with_logits(
            vlog, vlab, pos_weight=torch.tensor(self.pos_weight, device=dev)
        )
        base = nce + self.lam_valid * vloss
        zero = target_lat.new_zeros(())
        return EmissionOut(
            recon=recon,
            base_loss=base,
            # reuse the canonical log keys (semantic map: recon_loss<-InfoNCE, loss_stop<-validity BCE, loss_dims<-0)
            logs={"loss_stop": vloss.detach(), "loss_dims": zero, "recon_loss": nce.detach()},
            dim_lg=None,
            lab_label=None,
        )

    @torch.no_grad()
    def emit_infer(self, texts: list[str], max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One-pass inference: text -> a validity-gated fact-vector SET per row (NO autoregression). Returns
        (lat [B, Lmax, d], lens [B]) padded — the same shape the AR rollout yields, so eval/retrieval is unchanged."""
        m, dev = self.m, self.dev
        e = m.tokenizer(
            texts, padding=True, truncation=True, max_length=m.max_len, padding_side="left", return_tensors="pt"
        ).to(dev)
        ids, mask = e["input_ids"], e["attention_mask"]
        substrate = m._last_hidden(m._run_backbone(m.embed(ids), mask, ids, 0)).float()
        vecs, vlog = self.bridge(substrate, mask.bool())  # [B, nq, d], [B, nq]
        keep = vlog.sigmoid() > 0.5
        rows: list[torch.Tensor] = []
        lens: list[int] = []
        for r in range(vecs.size(0)):
            idx = keep[r].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:  # fallback: never emit zero — keep the single most-confident query
                idx = vlog[r].argmax().unsqueeze(0)
            idx = idx[:max_steps]
            rows.append(vecs[r, idx])
            lens.append(int(idx.numel()))
        lmax = max(lens)
        lat = torch.zeros(vecs.size(0), lmax, vecs.size(-1), device=dev)
        for r, v in enumerate(rows):
            lat[r, : v.size(0)] = v
        return lat, torch.tensor(lens, device=dev)
