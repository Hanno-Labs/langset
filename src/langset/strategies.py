"""Interchangeable strategies for multi-latent training.

A training step combines an emission objective, a target source, optional loss terms, and small callables for
epoch ordering, checkpoint selection, and seed construction. `TrainingArguments` stores the selected
implementations, which are instantiated once and used through shared interfaces.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from langset.loss import (
    CoTLossContext,
    InfoNCELossContext,
    SoftTargetCrossEntropyContext,
    StopLossContext,
    SupervisedContrastiveLossContext,
    cot_loss,
    info_nce_loss,
    soft_target_cross_entropy_loss,
    stop_loss,
    supervised_contrastive_loss,
)
from langset.modeling import LangSetModel
from langset.sigreg import SIGReg

if TYPE_CHECKING:  # annotations only (from __future__ import annotations -> strings);
    from transformers import PreTrainedTokenizerBase

    from langset.trainer import Trainer  # avoids a runtime import cycle trainer <-> strategies
    from langset.training_args import TrainingArguments


# ---- per-step context handed to the aux loss terms ----------------------------------------------
@dataclass
class MultiStepCtx:
    """Inputs shared by auxiliary loss terms for one multi-latent step.

    ``recon`` and ``target_lat`` have shape ``[B, L, d]`` and ``valid`` marks the ``N`` real emission slots.
    Indexing either latent tensor with ``valid`` produces the canonical flattened ``[N, d]`` view.
    ``flat_texts``, ``lens_l``, and ``bidx`` follow the same row-major ordering.
    """

    trainer: Trainer  # the owning Trainer; read its PER-ROW data (indexed by dataset row id):
    #                                     sup_labels / hard_neg_texts / concept and state targets
    args: TrainingArguments  # the run config — a term reads its own weight/temperature here (a.lam_*, a.tau)
    model: LangSetModel  # the online model being trained (rarely needed directly — emit via target_source)
    dev: torch.device  # device every tensor below lives on; build new tensors with device=c.dev
    bidx: list[
        int
    ]  # this batch's DATASET ROW IDS (len B) — index trainer.sup_labels[k], etc. with these
    lens_l: list[int]  # emitted-item count per row (len B): row r produced lens_l[r] items; Σ = N
    flat_texts: list[
        str
    ]  # the N target texts row-major (row0's items, then row1's, ...), aligned to recon[valid]
    valid: (
        torch.Tensor
    )  # [B, L] bool mask of real (non-padding) emission slots; recon[valid] -> [N, d]
    target_lat: (
        torch.Tensor
    )  # [B, L, d] the stop-grad TARGET latents each emission is trained toward
    recon: (
        torch.Tensor
    )  # [B, L, d] the model's EMITTED latents this step — gradient flows through these
    lmax: int  # L above: the padded emitted-item time dim for this batch
    target_source: _TargetSource  # the target provider; call .encode(texts) -> [n, d] normalized latents (hard-neg bank)
    phase_head: Optional[
        torch.nn.Module
    ]  # transient hidden->phase linear classifier, or None when lam_phase == 0
    phase_ids: dict[str, int]  # phase-label string -> class index, the CE targets for phase_head


# ---- aux loss terms -----------------------------------------------------------------------------
class _LossTerm:
    """Interface for an optional weighted term added to the base emission loss.

    Terms are built once, evaluated each step, and return ``None`` when their required inputs or weights are absent.
    """

    key: str = ""  # this term's log/agg name (e.g. "loss_multi_nce"); set by each subclass
    # Isolated terms run after the shared loss backward pass, so their graph does not coexist with the main graph.
    # Their gradients accumulate before the same optimizer step.
    isolated_backward: bool = False

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        """Compute this term for the step described by `c`. Return `(key, raw_unweighted_loss, weight)` — the
        loop then does `loss += weight * raw` and logs `raw` under `key` — or None to skip this term entirely
        (e.g. its weight is 0 or its required column/head is absent), which is a no-op for the step."""
        raise NotImplementedError


def identical_text_mask(c: MultiStepCtx, fn_mask: torch.Tensor) -> None:
    """Mask pairs with identical target text from the in-batch negative set.

    Mutates the ``[N, N]`` boolean ``fn_mask`` in place by setting excluded pairs to ``True``.
    """
    grp: dict[str, list[int]] = {}
    for ii, tx in enumerate(c.flat_texts):  # flat_texts is row-major aligned with recon[valid]
        grp.setdefault(tx, []).append(ii)
    for mem in grp.values():
        if len(mem) > 1:
            for aa in mem:
                for bb in mem:
                    if aa != bb:
                        fn_mask[aa, bb] = True


class MultiNCETerm(_LossTerm):
    """Apply in-batch InfoNCE between emitted latents and target-source latents.

    Each emission is paired with its aligned target. Other targets act as negatives unless excluded by a configured
    masker. The term is controlled by ``lam_multi_nce`` and is skipped when the target source suppresses NCE.
    """

    key = "loss_multi_nce"

    def __init__(self, maskers: list[Callable[[MultiStepCtx, torch.Tensor], None]]) -> None:
        self.maskers = maskers

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a = c.args
        if c.target_source.suppresses_nce:  # e.g. SIGReg replaces the NCE with its regularizer
            return None
        if not (a.lam_multi_nce > 0 and int(c.valid.sum()) > 1):
            return None
        emitted = c.recon[c.valid]
        targets = c.target_lat[c.valid]
        emission_count = emitted.size(0)
        fn_mask = torch.zeros(emission_count, emission_count, dtype=torch.bool, device=c.dev)
        for masker in self.maskers:
            masker(c, fn_mask)
        loss_nce = info_nce_loss(
            InfoNCELossContext(
                anchors=emitted,
                candidates=targets,
                positive_indices=torch.arange(emission_count, device=c.dev),
                temperature=a.tau,
                logit_mask=fn_mask,
                normalize_embeddings=True,
            )
        ).to_tensor()
        return (self.key, loss_nce, a.lam_multi_nce)


class SupConTerm(_LossTerm):
    """Supervised-contrastive shaping over emitted latents by the per-item `sup_field` group labels."""

    key = "loss_sup"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if self_.sup_labels is None or a.lam_sup <= 0:
            return None
        sup_flat = [
            (self_.sup_labels[k][j] if j < len(self_.sup_labels[k]) else "unknown")
            for r, k in enumerate(c.bidx)
            for j in range(c.lens_l[r])
        ]
        loss_sup = supervised_contrastive_loss(
            SupervisedContrastiveLossContext(
                embeddings=c.recon[c.valid], labels=sup_flat, temperature=a.sup_tau
            )
        ).to_tensor()  # pull same-stage, push different-stage
        return (self.key, loss_sup, a.lam_sup)


class PhaseTerm(_LossTerm):
    """CE phase classifier on the emitted reconstruction (non-collapsing SupCon alternative)."""

    key = "loss_phase"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if c.phase_head is None:
            return None
        sup = self_.sup_labels
        assert (
            sup is not None
        )  # set whenever a phase head exists (ty can't see the cross-attr invariant)
        pf = [
            (sup[k][j] if j < len(sup[k]) else "")
            for r, k in enumerate(c.bidx)
            for j in range(c.lens_l[r])
        ]
        pid = torch.tensor([c.phase_ids.get(x, -100) for x in pf], device=c.dev)
        loss_phase = F.cross_entropy(c.phase_head(c.recon[c.valid]), pid, ignore_index=-100)
        return (self.key, loss_phase, a.lam_phase)


def build_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """Build the default auxiliary loss terms.

    Returns an in-batch InfoNCE term with identical-text false-negative masking, followed by the
    supervised-contrastive term. Each term skips itself when its required weight or dataset labels are absent.
    """
    return [MultiNCETerm(maskers=[identical_text_mask]), SupConTerm()]


class CoTGenTerm(_LossTerm):
    """Train next-token generation of each row's ``cot_text`` from its input seed.

    The term uses an isolated backward pass so its long generation graph does not coexist with the latent-emission
    graph. Pair it with `cot_seed_texts` when emissions should also condition on the provided reasoning.
    Batches without reasoning text are skipped.
    """

    key = "loss_cot"
    isolated_backward = True

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, m, dev, self_ = c.args, c.model, c.dev, c.trainer
        if not any(
            self_.cot_texts[k] for k in c.bidx
        ):  # no reasoning in this batch -> nothing to learn
            return None
        tok = m.tokenizer

        def _tokm(texts: list[str], mx: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
            e = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=mx,
                padding_side=side,
                return_tensors="pt",
            )
            return e["input_ids"].to(dev), e["attention_mask"].to(dev)

        # Keep reasoning blocks up to the configured sequence limit. Seeds are left-padded so their final real token
        # predicts the first reasoning token; reasoning targets are right-padded so no padding separates seed and
        # target tokens, and padding remains excluded from cross-entropy.
        di, dm = _tokm([self_.input_text[k] for k in c.bidx], a.max_len, "left")
        ti, tm = _tokm([self_.cot_texts[k] or " " for k in c.bidx], a.max_len, "right")
        loss_cot = cot_loss(
            CoTLossContext(
                model=m,
                args=a,
                seed_ids=di,
                seed_mask=dm,
                cot_ids=ti,
                cot_mask=tm,
            )
        ).to_tensor()
        return (self.key, loss_cot, a.lam_cot)


def build_cot_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """Build the default auxiliary terms plus `CoTGenTerm`.

    Use with ``seed_builder=cot_seed_texts`` and a dataset containing ``cot_text``.
    """
    return [*build_loss_terms(args), CoTGenTerm()]


# ---- emission objective -------------------------------------------------------------------------
@dataclass
class EmissionOut:
    """Result of one emission forward."""

    recon: torch.Tensor  # [B, L, d] the model's emitted latents — gradient flows through these
    base_loss: torch.Tensor  # scalar: the objective's own loss (objective-defined scalar)
    logs: dict[
        str, torch.Tensor
    ]  # UNWEIGHTED scalar components for logging (objective-defined components)
    code_logits: Optional[torch.Tensor] = None


class _EmissionObjective:
    """Interface for multi-latent emission objectives.

    An objective converts seeded inputs and target latents into gradient-bearing emitted latents and a base loss.
    ``codebook`` indicates whether inference uses the model's autoregressive codebook rollout. Implementations share
    a common constructor and can be selected through ``TrainingArguments.emission``.
    """

    codebook: bool = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        self.m, self.a, self.dev, self.trainer = model, args, dev, trainer

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
        """Run the emission forward and its base loss for one step.

        se:         tokenized seed batch (input_ids/attention_mask) already on device — the model reads this.
        target_lat: [B, L, d] stop-grad target latents to reconstruct toward.
        valid:      [B, L] bool mask of real (non-padding) emission slots.
        lens_l:     per-row emitted-item count (len B).
        bidx:       dataset row ids for this batch (len B) — for objectives that read per-row config.
        b:          B, the batch row count (== target_lat.size(0)); passed explicitly to size new tensors.
        lmax:       L, the padded emitted-item time dim.
        ep:         current epoch index (drives e.g. scheduled-sampling warmup).
        """
        raise NotImplementedError

    def build_targets(
        self,
        ent_lists: list[list[str]],
        flat_tgt: torch.Tensor,
        d: int,
        dev: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
        """Arrange encoded target items into padded teacher-forcing tensors.

        Returns ``(target_lat, valid, lens_l, lmax)``. ``target_lat`` has shape ``[B, L, d]``, ``valid`` marks real
        target slots, ``lens_l`` contains per-row lengths, and ``lmax`` is the padded sequence length. The default
        implementation preserves target-item order and performs no model forward.
        """
        lmax = max(len(x) for x in ent_lists)
        b = len(ent_lists)
        target_lat = torch.zeros(b, lmax, d, device=dev)
        valid = torch.zeros(b, lmax, dtype=torch.bool, device=dev)
        lens_l: list[int] = []
        k = 0
        for r, lst in enumerate(ent_lists):
            nl = len(lst)
            lens_l.append(nl)
            target_lat[r, :nl] = flat_tgt[k : k + nl]
            valid[r, :nl] = True
            k += nl
        return target_lat, valid, lens_l, lmax

    def emit_infer(self, texts: list[str], max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit latent sequences for inference.

        Returns zero-padded latents with shape ``[B, L, d]`` and per-row lengths. The default implementation
        delegates to the model's autoregressive codebook rollout; non-autoregressive objectives may override it.
        """
        lat, lens = self.m.rollout(  # ty: ignore[invalid-assignment]  # return_lengths=True -> (lat, lens)
            texts, max_steps=max_steps, return_lengths=True
        )
        return lat, lens

    def z_for_reg(
        self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor, lmax: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``[N, d]`` predicted and target representations passed to target regularization."""
        return em.recon[valid], target_lat[valid]


class CodeSoftmaxObjective(_EmissionObjective):
    """Emit one distribution over a fixed alphabet and commit its codebook mixture.

    Inject with `TrainingArguments(emission=CodeSoftmaxObjective)` and a head built with `code_emit=True`.
    Use this when the target geometry already encodes membership in a known alphabet. One normalized softmax
    allocates a fixed budget of mass, so members compete and the emitted latent is their superposition.

    The target law is read off the target latent itself, no extra column: for an orthonormal codebook and a
    membership target `normalize(multi_hot @ code.T)`, projecting back with `target @ code.T` recovers the members
    (equal weight) and ~0 elsewhere, so an L1 normalize gives the uniform-over-members law to match. Loss is the
    soft-target cross-entropy over the codes, plus an independent sigmoid terminator.
    """

    codebook = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        super().__init__(model, args, dev, trainer)
        assert model.head.code_emit, (
            "CodeSoftmaxObjective needs a codebook head: build the model with code_emit=True, n_codes=<alphabet>"
        )
        assert model.head.res_dim == 0, (
            "CodeSoftmaxObjective requires res_dim=0; use StateResidualObjective for named state plus a residual"
        )
        self.n_codes = int(model.head.n_codes)

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
        m, a = self.m, self.a
        ss_prob = a.ss_prob
        assert ss_prob is not None  # Trainer resolves the None sentinel before any emit
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        code_logits, stop_lg, _digits, recon = m.rollout_train_state(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
        )
        code = m.head.code  # [n_codes, d] fixed
        with torch.no_grad():  # the target's law over codes, recovered from the target latent
            w = F.relu(target_lat.float() @ code.t())  # [b, lmax, n_codes]
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        # Apply soft-target cross-entropy over code membership at real emission slots.
        loss_code = soft_target_cross_entropy_loss(
            SoftTargetCrossEntropyContext(
                logits=code_logits[:, :lmax, 0, :].float(),
                target_probabilities=w,
                selection=valid,
            )
        ).to_tensor()
        # Termination is independent of the member distribution, so set width cannot weaken its supervision.
        loss_stop = stop_loss(StopLossContext(logits=stop_lg, lengths=lens_l)).to_tensor()
        # Auxiliary terms consume `recon` as the gradient-bearing model emission. Return the differentiable committed
        # mixture rather than the teacher-forcing target, which carries no emitter gradient on this path.
        mix = F.normalize(
            code_logits[:, :lmax, 0, :].float().softmax(-1) @ code, dim=-1
        )  # [b, lmax, d]
        with (
            torch.no_grad()
        ):  # diagnostic: how close the emitted mixture lands to the target latent
            emit_cos = F.cosine_similarity(mix[valid], target_lat[valid], dim=-1).mean()
        del recon
        return EmissionOut(
            recon=mix,
            base_loss=loss_code + loss_stop,
            logs={"loss_code": loss_code, "loss_stop": loss_stop, "emit_cos": emit_cos},
            code_logits=code_logits,
        )

    def z_for_reg(
        self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor, lmax: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Regularize predicted and target laws over codes so the penalty acts on codebook usage rather than only on
        # committed latent vectors.
        code_logits = em.code_logits
        assert code_logits is not None
        code = self.m.head.code
        z_tgt = F.relu(target_lat[valid].float() @ code.t())
        z_tgt = z_tgt / z_tgt.sum(-1, keepdim=True).clamp_min(1e-9)
        return code_logits[:, :lmax, 0, :].float().softmax(-1)[valid], z_tgt


class ConceptObjective(_EmissionObjective):
    """Emit a latent with independently normalized named-concept facets.

    The dataset column selected by ``concept_field`` contains one concept mapping per target step. For example::

        {
            "input_text": "...",
            "target_texts": ["..."],
            "concepts": [
                {
                    "vocals": ["yell-singing", "gang-vocals"],
                    "mood": ["angry-but-vulnerable"],
                }
            ],
        }

    LangSet discovers each facet's alphabet from the dataset and builds a fixed codebook using ``code_source``.
    Only facets present in a row are supervised. Each facet is normalized independently, and ``res_dim`` optionally
    reserves dimensions for an unnamed residual representation.
    """

    codebook = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        super().__init__(model, args, dev, trainer)
        assert model.head.code_emit, "ConceptObjective needs code_emit=True on the model"
        self.spans = list(model.head.concept_spans)
        self.facets = list(model.head.concept_names) or [
            f"facet{i}" for i in range(len(self.spans))
        ]
        self.res_dim = int(model.head.res_dim)
        self.laws = getattr(trainer, "concept_laws", None) if trainer is not None else None
        assert self.laws is not None, (
            "ConceptObjective needs a parsed concepts column — set TrainingArguments.concept_field"
        )
        # Build a fixed code vector for every discovered concept, grouped into facet spans.
        alpha = getattr(trainer, "concept_alphabet", None)
        if alpha and not bool(model.head.code.abs().sum()):
            names_per_facet = [(f, alpha[f]) for f in alpha]
            budget = model.head.state_dim
            share = max(1, budget // max(len(names_per_facet), 1))
            facets_in = []
            for f, names in names_per_facet:
                dims = min(share, budget - share * (len(names_per_facet) - 1 - len(facets_in)))
                codes = build_codebook(getattr(args, "code_source", "model"), names, dims, model)
                facets_in.append((f, codes, dims))
            model.head.set_concepts(facets_in)
            self.spans = list(model.head.concept_spans)
            self.facets = list(model.head.concept_names)
            src = getattr(args, "code_source", "model")
            print(
                f"[concepts] codebook from "
                f"{src if isinstance(src, str) else getattr(src, '__name__', src)!r}: "
                + ", ".join(
                    f"{f}={len(n)}c/{d}d" for (f, n), (_, _, d) in zip(names_per_facet, facets_in)
                )
                + f" | residual {self.res_dim} dims",
                flush=True,
            )

    def _target_law(self, b: int, lmax: int, lens_l: list[int], bidx: list[int]) -> tuple:
        """Per-row/per-tick target distribution over ALL members, plus a mask of which facets were stated.
        A row that names only `vocals` trains only those dims — silence about a facet is not evidence."""
        n_codes = self.m.head.n_codes
        tgt = torch.zeros(b, lmax, n_codes, device=self.dev)
        seen = torch.zeros(b, lmax, len(self.spans), dtype=torch.bool, device=self.dev)
        assert self.laws is not None
        for r, k in enumerate(bidx):
            per_tick = self.laws[k]
            for t in range(min(lens_l[r], lmax, len(per_tick))):
                for fi, (m_lo, m_hi, _, _) in enumerate(self.spans):
                    w = per_tick[t].get(fi)
                    if w:
                        for m_idx, weight in w.items():
                            tgt[r, t, m_lo + m_idx] = weight
                        seen[r, t, fi] = True
        return tgt, seen

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
        m, a = self.m, self.a
        ss_prob = a.ss_prob
        assert ss_prob is not None
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        code_logits, stop_lg, _d, _r, emit_hid = m.rollout_train_state(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
            return_emit_hidden=True,
        )
        flat = code_logits[:, :lmax, 0, :].float()  # [b, lmax, n_codes]
        tgt, seen = self._target_law(b, lmax, lens_l, bidx)

        # Normalize and supervise each stated facet independently.
        losses, per_facet = [], {}
        for fi, (m_lo, m_hi, _, _) in enumerate(self.spans):
            sel = seen[:, :, fi] & valid
            if not bool(sel.any()):
                continue
            li = soft_target_cross_entropy_loss(
                SoftTargetCrossEntropyContext(
                    logits=flat[..., m_lo:m_hi],
                    target_probabilities=tgt[..., m_lo:m_hi],
                    selection=sel,
                )
            ).to_tensor()
            losses.append(li)
            per_facet[f"c_{self.facets[fi]}"] = li
        loss_concept = torch.stack(losses).sum() if losses else flat.new_zeros(())

        loss_stop = stop_loss(StopLossContext(logits=stop_lg, lengths=lens_l)).to_tensor()

        # Auxiliary terms require the gradient-bearing committed emission rather than teacher-forcing targets.
        p = m.head.concept_probs(code_logits[:, :lmax])
        state = F.normalize(p @ m.head.code, dim=-1)
        emission = (
            torch.cat([state, m.head.residual(emit_hid[:, :lmax])], -1) * (0.5**0.5)
            if self.res_dim
            else state
        )
        with torch.no_grad():
            emit_cos = F.cosine_similarity(emission[valid], target_lat[valid], dim=-1).mean()
        return EmissionOut(
            recon=emission,
            base_loss=loss_concept + loss_stop,
            logs={
                "loss_concept": loss_concept,
                "loss_stop": loss_stop,
                "emit_cos": emit_cos,
                **per_facet,
            },
        )


class StateResidualObjective(_EmissionObjective):
    """Emit a latent composed of a named codebook state and an unnamed residual.

    The leading ``latent_dim - res_dim`` dimensions contain a normalized mixture over the fixed state codebook.
    The trailing ``res_dim`` dimensions contain a learned residual. State labels come from the dataset column
    selected by ``state_field`` and are represented as a per-step list of active member indices. Target probability
    mass is divided evenly among the active members at each step.

    Construct the model with ``code_emit=True``, set ``n_codes`` to the state alphabet size, and choose a positive
    ``res_dim`` when an unnamed residual is required.
    """

    codebook = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        super().__init__(model, args, dev, trainer)
        assert model.head.code_emit, (
            "StateResidualObjective needs a codebook head: build with code_emit=True, n_codes=<alphabet>"
        )
        self.n_codes = int(model.head.n_codes)
        self.res_dim = int(model.head.res_dim)
        # Build the fixed codebook from member names unless the caller installed one before training.
        names = getattr(args, "code_names", None)
        if names is not None and not bool(model.head.code.abs().sum()):
            codes = build_codebook(
                getattr(args, "code_source", "random"), list(names), model.head.state_dim, model
            )
            model.head.set_code(codes.to(model.head.code.device))
            src = getattr(args, "code_source", "random")
            print(
                f"[state] codebook: {len(names)} members x {model.head.state_dim} dims "
                f"from {src if isinstance(src, str) else getattr(src, '__name__', src)!r}"
                f" | residual {self.res_dim} dims",
                flush=True,
            )
        self.labels = getattr(trainer, "state_labels", None) if trainer is not None else None
        assert self.labels is not None, (
            "StateResidualObjective needs per-tick member labels: set TrainingArguments.state_field to a row "
            "column of per-tick active-index lists"
        )

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
        m, a, dev = self.m, self.a, self.dev
        ss_prob = a.ss_prob
        assert ss_prob is not None
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        code_logits, stop_lg, _digits, _recon, emit_hid = m.rollout_train_state(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
            return_emit_hidden=True,  # the residual is a function of the emit hidden, not of the logits
        )
        code = m.head.code

        # Split target probability mass evenly across each step's active members.
        tgt = torch.zeros(b, lmax, self.n_codes, device=dev)
        has = torch.zeros(b, lmax, dtype=torch.bool, device=dev)
        assert self.labels is not None
        for r, k in enumerate(bidx):
            per_tick = self.labels[k]
            for t in range(min(lens_l[r], lmax, len(per_tick))):
                members = [int(c) for c in per_tick[t] if 0 <= int(c) < self.n_codes]
                if members:
                    tgt[r, t, members] = 1.0 / len(members)
                    has[r, t] = True
        loss_state = soft_target_cross_entropy_loss(
            SoftTargetCrossEntropyContext(
                logits=code_logits[:, :lmax, 0, :].float(),
                target_probabilities=tgt,
                selection=has & valid,
            )
        ).to_tensor()

        # Termination uses an independent sigmoid. Folding it into a diffuse member distribution would make stop
        # supervision depend on the number of active members.
        loss_stop = stop_loss(StopLossContext(logits=stop_lg, lengths=lens_l)).to_tensor()

        # Return the same gradient-bearing state-plus-residual emission that is fed back. Auxiliary losses use this
        # value as their query and must not receive the detached teacher-forcing target.
        p = code_logits[:, :lmax, 0, :].float().softmax(-1)
        state = F.normalize(p @ code, dim=-1)
        if self.res_dim:
            res = m.head.residual(emit_hid[:, :lmax])
            emission = torch.cat([state, res], -1) * (0.5**0.5)
        else:
            emission = state
        with torch.no_grad():
            emit_cos = F.cosine_similarity(emission[valid], target_lat[valid], dim=-1).mean()
        logs = {"loss_state": loss_state, "loss_stop": loss_stop, "emit_cos": emit_cos}
        return EmissionOut(
            recon=emission,
            base_loss=loss_state + loss_stop,
            logs=logs,
        )


# ---- target source ------------------------------------------------------------------------------
class _TargetSource:
    """Interface for target-latent providers and optional anti-collapse regularization.

    Sources share a common constructor and can be selected through ``TrainingArguments.target_source``.
    """

    suppresses_nce: bool = False  # skip in-batch NCE when the target source replaces it
    wants_regularizer: bool = False  # request objective representations for `regularizer`
    twin: Optional[LangSetModel] = None  # encoder used to build the evaluation retrieval bank

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts as ``[n, d]`` L2-normalized target latents.

        The gradient policy is defined by the target source. The trainer uses these values for per-step targets and
        for the hard-negative bank.
        """
        raise NotImplementedError

    def update(self) -> None:
        """Update source state after an optimizer step; the base implementation is a no-op."""

    def regularizer(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Optional[torch.Tensor]:
        """Return an optional regularization loss for predicted and target representations."""
        return None


class EMATwinTarget(_TargetSource):
    """Produce stop-gradient targets from an exponential-moving-average copy of the online model.

    After each optimizer step, `update` moves the target model toward the trainable online parameters using
    ``ema_m``. The lagged target provides a more stable comparison geometry than using the online model on both sides.
    """

    suppresses_nce = False

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        tok: PreTrainedTokenizerBase,
        dev: torch.device,
    ) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = copy.deepcopy(model)
        for p in self.twin.parameters():
            p.requires_grad_(False)
        self.twin.eval()
        self._online = [po for po in model.parameters() if po.requires_grad]
        self._ema = [
            pe for pe, po in zip(self.twin.parameters(), model.parameters()) if po.requires_grad
        ]

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Encode each text as one normalized target latent without gradients. Inputs are truncated to
        # `target_max_len`, which callers should increase for document-length targets.
        a, tok, dev = self.a, self.tok, self.dev
        e = tok(
            texts, padding=True, truncation=True, max_length=a.target_max_len, return_tensors="pt"
        ).to(dev)
        twin = self.twin
        assert twin is not None  # built in __init__ (deepcopy of the model)
        with torch.no_grad():
            z = twin(e["input_ids"], e["attention_mask"])
        return F.normalize(z.float(), dim=-1)

    def update(self) -> None:
        with torch.no_grad():
            torch._foreach_mul_(self._ema, self.a.ema_m)  # ty: ignore[no-matching-overload]  # torch _foreach_ stub overloads
            torch._foreach_add_(self._ema, self._online, alpha=1.0 - self.a.ema_m)  # ty: ignore[no-matching-overload]  # torch _foreach_ stub overloads


class CachedTarget(_TargetSource):
    """Produce targets with a frozen encoder and cache embeddings by input text.

    If ``target_encoder_ckpt`` is configured, the encoder is loaded from that LangSet checkpoint. Otherwise, the
    source is a frozen copy of the online model taken at initialization. Each unique text is encoded once and
    subsequent calls reuse the cached latent. `update` is a no-op because the target geometry is fixed.
    """

    suppresses_nce = False

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        tok: PreTrainedTokenizerBase,
        dev: torch.device,
    ) -> None:
        self.a, self.dev = args, dev
        ckpt = getattr(args, "target_encoder_ckpt", None)
        if ckpt:
            from langset.modeling import (
                LangSetModel as _LSM,
            )  # local import avoids a strategies<->modeling cycle

            enc = _LSM.load(ckpt, device=str(dev))
        else:
            enc = copy.deepcopy(model)  # frozen snapshot of the online model at init
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        self.enc = enc
        self.twin = enc  # eval retrieval bank encodes with the fixed geometry
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: list[str]) -> torch.Tensor:
        miss = [t for t in dict.fromkeys(texts) if t not in self._cache]  # unique, order-preserving
        if miss:
            a, dev = self.a, self.dev
            e = self.enc.tokenizer(
                miss,
                padding=True,
                truncation=True,
                max_length=a.target_max_len,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                z = F.normalize(self.enc(e["input_ids"], e["attention_mask"]).float(), dim=-1)
            for t, v in zip(miss, z):
                self._cache[t] = v.detach()
        return torch.stack([self._cache[t] for t in texts])

    def update(self) -> None:
        return None  # fixed geometry — nothing to track


class SIGRegTarget(_TargetSource):
    """Use the online model for gradient-bearing targets and apply SIGReg regularization.

    This target source does not maintain an EMA copy. Both predicted and target representations remain connected to
    the online model, and independent SIGReg penalties are applied to them. The default in-batch NCE term is
    suppressed. Configure the regularizer with ``sigreg_lambda``, ``sigreg_knots``, and ``sigreg_slices``.
    """

    suppresses_nce = True
    wants_regularizer = True

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        tok: PreTrainedTokenizerBase,
        dev: torch.device,
    ) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = model  # no separate twin — eval encodes with the live model
        self.sig_reg = SIGReg(knots=args.sigreg_knots, slices=args.sigreg_slices).to(dev)

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Keep live target embeddings connected to the online model; SIGReg supplies the configured regularization.
        a, tok, dev = self.a, self.tok, self.dev
        e = tok(
            texts, padding=True, truncation=True, max_length=a.target_max_len, return_tensors="pt"
        ).to(dev)
        z = self.m(e["input_ids"], e["attention_mask"])
        return F.normalize(z.float(), dim=-1)

    def regularizer(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Optional[torch.Tensor]:
        # Two independent Gaussianity penalties, not a match between predicted and target representations.
        return self.sig_reg(z_pred) + self.sig_reg(z_tgt)


# ---- concepts (the text-in format for a named, superposed state) ---------------------------------
# A row's `concepts` column maps facet names to the concepts that are active:
#
#     "concepts": {"vocals": ["yell-singing", "gang-vocals"], "tempo": {"7": 0.1, "8": 0.9}}
#
# A list means "these are all true, equally" (mass split evenly — the superposition). A dict means explicit
# weights, which also encodes a CONTINUOUS value as a mixture over ordered concepts: 7.9 is 0.1 of "7" and 0.9
# of "8", interpolation included, and unlike a regressed scalar you can still read what it says.
#
# The alphabet is discovered by scanning the column. Multi-latent rows provide one mapping per step, aligned with
# `target_texts`.
def parse_concepts(raw: object) -> "dict[str, dict[str, float]]":
    """Parse one concept mapping and normalize positive weights within each facet."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("concepts must be a dictionary")
    out: dict[str, dict[str, float]] = {}
    for facet, members in raw.items():
        if isinstance(members, dict):
            w = {}
            for name, value in members.items():
                if not isinstance(value, (int, float, str)):
                    raise TypeError("concept weights must be numeric")
                weight = float(value)
                if weight > 0:
                    w[str(name)] = weight
        elif isinstance(members, str):
            w = {members: 1.0} if members.strip() else {}
        elif isinstance(members, (list, tuple)):
            names = list(members)
            w = {str(n): 1.0 for n in names if str(n).strip()}
        else:
            raise TypeError("concept members must be a dictionary, list, tuple, or string")
        tot = sum(w.values())
        if tot > 0:
            out[str(facet)] = {k: v / tot for k, v in w.items()}
    return out


def discover_concept_alphabet(rows_concepts: "list[object]") -> "dict[str, list[str]]":
    """Return sorted concept names for each facet discovered in the dataset.

    Sorting facets and members makes the resulting codebook layout deterministic.
    """
    seen: dict[str, set[str]] = {}
    for raw in rows_concepts:
        ticks = raw if isinstance(raw, (list, tuple)) else [raw]
        for tick in ticks:
            for facet, w in parse_concepts(tick).items():
                seen.setdefault(facet, set()).update(w)
    return {f: sorted(v) for f, v in sorted(seen.items())}


# ---- code sources (where a named member's vector comes from) -------------------------------------
# A `code_source` maps member names to a `[n_members, dim]` tensor during setup. The result is frozen for the run.
# Orthonormal codes support exact linear recovery but encode no member relationships; model-derived codes may
# preserve information from token embeddings but are not orthogonal or exactly invertible.
def random_orthonormal_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Return a deterministic random orthonormal codebook with one row per member.

    Requires ``len(names) <= dim`` and produces rows satisfying ``C Cᵀ = I``.
    """
    assert len(names) <= dim, (
        f"random_orthonormal_codes needs dim >= n_members for an orthonormal frame; got dim={dim}, "
        f"n_members={len(names)}. Use model_embedded_codes (no such limit) or widen the state half."
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.linalg.qr(torch.randn(dim, len(names), generator=g)).Q[:, : len(names)]
    return q.t().contiguous()


def model_embedded_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Build code vectors by mean-pooling input embeddings for each member name.

    Each vector is truncated or padded to ``dim`` and normalized. The resulting rows are not guaranteed to be
    orthogonal or exactly invertible, and the alphabet size is not limited by ``dim``.
    """
    emb = model.embed.weight  # [V, h]; tied to the LM head on most small models
    out = []
    for nm in names:
        ids = model.tokenizer(nm, add_special_tokens=False)["input_ids"] or [
            model.tokenizer.eos_token_id
        ]
        v = emb[torch.tensor(ids, device=emb.device)].float().mean(0)
        out.append(v[:dim] if v.numel() >= dim else F.pad(v, (0, dim - v.numel())))
    return F.normalize(torch.stack(out), dim=-1)


def orthogonalized_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Embed member names and apply QR decomposition to produce orthonormal rows.

    This enables an orthogonal readout but generally changes pairwise relationships between the name embeddings.
    """
    e = model_embedded_codes(names, dim, model)
    q = torch.linalg.qr(e.t().float()).Q[:, : len(names)]
    return q.t().contiguous()


def twin_encoded_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Encode member names through the model's emission path and normalize the resulting code vectors.

    This places the codebook in the same output space as model-produced target latents. The emission path must be
    usable during setup.
    """
    with torch.no_grad():
        z = model.emit(list(names))  # [n_members, latent_dim], the target space itself
    z = z.float()[:, :dim] if z.size(-1) >= dim else F.pad(z.float(), (0, dim - z.size(-1)))
    return F.normalize(z, dim=-1)


CODE_SOURCES = {
    "random": random_orthonormal_codes,
    "model": model_embedded_codes,
    "orthogonal": orthogonalized_codes,
    "twin": twin_encoded_codes,
}


def build_codebook(
    source: "str | Callable[[list[str], int, LangSetModel], torch.Tensor]",
    names: list[str],
    dim: int,
    model: LangSetModel,
) -> torch.Tensor:
    """Resolve a `code_source` (name or callable) and build the [n_members, dim] codebook. Validates the shape
    here so a bad custom source fails at setup with a clear message rather than deep inside the first step."""
    fn = CODE_SOURCES[source] if isinstance(source, str) else source
    codes = fn(list(names), int(dim), model)
    assert tuple(codes.shape) == (len(names), dim), (
        f"code_source {getattr(fn, '__name__', source)!r} returned {tuple(codes.shape)}, "
        f"expected ({len(names)}, {dim})"
    )
    return codes


# ---- small function-strategies ------------------------------------------------------------------
def multi_epoch_order(
    tr_idx: list[int], rng_t: torch.Generator, args: TrainingArguments, seeds: list[str]
) -> list[int]:
    """Return a random permutation of the training positions for one epoch."""
    return torch.randperm(len(tr_idx), generator=rng_t).tolist()


def multi_select_metric(mode: str, mrr: float, pur: float, ep: int) -> float:
    """Return the configured retrieval, purity, or blended checkpoint-selection score."""
    return pur if mode == "purity" else (mrr + pur) if mode == "blend" else mrr


def last_epoch_selector(mode: str, mrr: float, pur: float, ep: int) -> float:
    """Return the epoch index so each evaluated epoch supersedes earlier checkpoints.

    The attached ``needs_final_epoch`` flag ensures the final epoch is evaluated even when ``eval_every > 1``.
    """
    return float(ep)


# The trainer evaluates/selects only on `ep % eval_every == 0`; a "keep the last epoch" selector must still see the
# FINAL epoch even when eval_every>1, or it silently restores an earlier (last-evaluated) epoch. This flag tells the
# trainer to always evaluate the final epoch for this selector; the default selector lacks it, so its path is unchanged.
last_epoch_selector.needs_final_epoch = True  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # function-attribute flag


def multi_seed_texts(trainer: Trainer, seeds: list[str], args: TrainingArguments) -> list[str]:
    """Return the raw input seeds used by the default emission forward."""
    return seeds


def cot_seed_texts(trainer: Trainer, seeds: list[str], args: TrainingArguments) -> list[str]:
    """Append each row's reasoning text to its seed before emission.

    Pair with `build_cot_loss_terms` so the same reasoning is trained autoregressively. Targets and evaluation
    continue to use the raw seeds.
    """
    return [f"{s}\n\nReasoning:\n{trainer.cot_texts[i]}" for i, s in enumerate(seeds)]
