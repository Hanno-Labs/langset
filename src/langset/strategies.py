"""Swappable training strategies for the multi-latent trainer (GoF Strategy pattern).

The multi-latent step is assembled from a few interchangeable pieces — the emission objective, the target
source, the aux loss terms, plus small function-strategies (epoch ordering, checkpoint selection, seed
building). Each is a class or callable with a fixed interface and a DEFAULT implementation here that reproduces
the historical behavior byte-for-byte (guarded by tests/test_trainer_multi_characterization.py).

`TrainingArguments` holds these as INJECTABLE fields (defaults below), so selecting a different behavior is
passing a different implementation — `TrainingArguments(target_source=SIGRegTarget)` — not toggling a flag that
the trainer then branches on. The trainer builds each once and uses it with no per-feature `if`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from langset.modeling import LangSetModel
from langset.sigreg import SIGReg

if TYPE_CHECKING:  # annotations only (from __future__ import annotations -> strings);
    from transformers import PreTrainedTokenizerBase

    from langset.trainer import Trainer  # avoids a runtime import cycle trainer <-> strategies
    from langset.training_args import TrainingArguments


def supcon_loss(z: torch.Tensor, labels: list[str], tau: float) -> torch.Tensor:
    """Supervised-contrastive (Khosla et al.) over emitted latents: same-label items are positives (pulled together),
    all others negatives (pushed apart) — a few group labels SHAPE the geometry into SEPARATE REGIONS. Being a proper
    contrastive loss (each anchor has both positives and negatives) it separates without collapsing. Items whose label
    is ''/'unknown'/'none'/'nan' are dropped. Returns 0 if fewer than two labelled items or no positive pair exists."""
    dev = z.device
    keep = [
        k
        for k, l in enumerate(labels)
        if str(l).strip().lower() not in ("", "unknown", "none", "nan")
    ]
    if len(keep) < 2:
        return z.new_zeros(())
    zz = F.normalize(z[keep], p=2, dim=-1)
    lab = [labels[k] for k in keep]
    b = len(keep)
    sim = (zz @ zz.t() / tau).masked_fill(torch.eye(b, device=dev, dtype=torch.bool), -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = torch.tensor(
        [[1.0 if (i != j and lab[i] == lab[j]) else 0.0 for j in range(b)] for i in range(b)],
        device=dev,
    )
    npos = pos.sum(1)
    has = npos > 0
    if not bool(has.any()):
        return z.new_zeros(())
    return (-(pos * logp).sum(1)[has] / npos[has]).mean()


# ---- per-step context handed to the aux loss terms ----------------------------------------------
@dataclass
class MultiStepCtx:
    """Read-only snapshot of ONE multi-latent training step, handed to every aux `_LossTerm`. Assembled right
    after the emission forward; a term pulls the few fields it needs and returns its contribution (or None).

    Shape legend used below: B = rows in this batch · L = `lmax` (max emitted items across the batch's rows,
    the padded time dim) · N = number of VALID emissions in the batch (= Σ lens_l, since rows differ in length)
    · d = latent dim · V = `fsq_levels`. The canonical flattened view a term works in is `recon[valid]` -> [N, d],
    and `flat_texts` / `lens_l` / `bidx` are all aligned to that same row-major order.
    """

    trainer: Trainer  # the owning Trainer; read its PER-ROW data (indexed by dataset row id):
    #                                     sup_labels / hard_neg_texts / label_plan + label_cols + label_codewords
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
    dim_lg: Optional[
        torch.Tensor
    ]  # [B, L+1, fsq_dim, V] FSQ per-dim digit logits; None for a non-FSQ objective
    lmax: int  # L above: the padded emitted-item time dim for this batch
    fsq_levels: int  # V above: FSQ quantization levels per digit
    lab_label: Optional[
        torch.Tensor
    ]  # [B, L, n_reserved] reserved-dim label targets; None unless FSQ label subspace on
    target_source: _TargetSource  # the target provider; call .encode(texts) -> [n, d] normalized latents (hard-neg bank)
    phase_head: Optional[
        torch.nn.Module
    ]  # transient hidden->phase linear classifier, or None when lam_phase == 0
    phase_ids: dict[str, int]  # phase-label string -> class index, the CE targets for phase_head


# ---- aux loss terms -----------------------------------------------------------------------------
class _LossTerm:
    """Strategy for one weighted, optional term added on top of the base emission loss (one per historical
    `if a.lam_x > 0:` block). Terms are built once and iterated each step; each self-skips when inapplicable."""

    key: str = ""  # this term's log/agg name (e.g. "loss_multi_nce"); set by each subclass
    isolated_backward: bool = (
        False  # if True the term is NOT summed into the shared loss; instead the trainer runs
    )
    #                                     its forward+backward SEPARATELY, AFTER the main loss.backward() has freed its
    #                                     graph, so the two graphs never coexist (peak activation = max, not sum). Grads
    #                                     accumulate into .grad before the single opt.step() -> same step, batch unchanged.

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        """Compute this term for the step described by `c`. Return `(key, raw_unweighted_loss, weight)` — the
        loop then does `loss += weight * raw` and logs `raw` under `key` — or None to skip this term entirely
        (e.g. its weight is 0 or its required column/head is absent), which is a no-op for the step."""
        raise NotImplementedError


def identical_text_mask(c: MultiStepCtx, fn_mask: torch.Tensor) -> None:
    """A negative-mask: MUTATES `fn_mask` ([N, N] bool, N = valid emissions) in place, setting [i, j] = True for
    pairs that must NOT be treated as negatives of each other. Default policy: two emissions with IDENTICAL
    target text share the same true geometry, so they aren't negatives (mirrors the single-latent mask_keys path)."""
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
    """IN-BATCH-NEGATIVE InfoNCE: each emitted recon vs the batch's EMA targets, own target = positive, others
    = negatives, minus the `maskers`' false-negatives. On by default (lam_multi_nce). Ported from the
    single-latent self-contrastive loss."""

    key = "loss_multi_nce"

    def __init__(self, maskers: list[Callable[[MultiStepCtx, torch.Tensor], None]]) -> None:
        self.maskers = maskers

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a = c.args
        if c.target_source.suppresses_nce:  # e.g. SIGReg replaces the NCE with its regularizer
            return None
        if not (a.lam_multi_nce > 0 and int(c.valid.sum()) > 1):
            return None
        rvn = F.normalize(c.recon[c.valid], dim=-1)  # [N, d] emitted (gradient flows here)
        tvn = F.normalize(c.target_lat[c.valid], dim=-1)  # [N, d] EMA targets (already stop-grad)
        nce_logits = (rvn @ tvn.t()) / a.tau  # [N, N] query x key cosine / temp
        n_nce = rvn.size(0)
        fn_mask = torch.zeros(n_nce, n_nce, dtype=torch.bool, device=c.dev)
        for masker in self.maskers:
            masker(c, fn_mask)
        nce_logits = nce_logits.masked_fill(
            fn_mask, float("-inf")
        )  # diagonal (positive) never masked
        loss_nce = F.cross_entropy(nce_logits, torch.arange(n_nce, device=c.dev))
        return (self.key, loss_nce, a.lam_multi_nce)


class HardNegTerm(_LossTerm):
    """Each emitted recon: own EMA target (positive) vs a shared bank of the batch's mined hard-negative texts."""

    key = "loss_hard_neg"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if self_.hard_neg_texts is None or a.lam_hard_neg <= 0:
            return None
        hn_flat = [t for k in c.bidx for t in self_.hard_neg_texts[k]]
        if not hn_flat:
            return None
        hn_bank = c.target_source.encode(hn_flat)  # [Nhn, d] stop-grad normalized hard-neg latents
        rv = F.normalize(c.recon[c.valid], dim=-1)  # [Nvalid, d] emitted reconstructions
        pos = (rv * c.target_lat[c.valid]).sum(-1, keepdim=True)  # [Nvalid, 1] cos to own target
        neg = rv @ hn_bank.t()  # [Nvalid, Nhn] cos to every hard neg
        logits_hn = torch.cat([pos, neg], dim=1) / a.tau
        loss_hn = F.cross_entropy(
            logits_hn, torch.zeros(logits_hn.size(0), dtype=torch.long, device=c.dev)
        )
        return (self.key, loss_hn, a.lam_hard_neg)


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
        loss_sup = supcon_loss(
            c.recon[c.valid], sup_flat, a.sup_tau
        )  # pull same-stage, push different-stage
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


class LabelDimsTerm(_LossTerm):
    """FSQ LABEL SUBSPACE: full-strength CE on the reserved digit dims so the label lives AS coordinates of the
    emitted code. FSQ-only (reads dim_lg); skipped when the objective produces no digit logits or no label plan."""

    key = "loss_label"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if c.lab_label is None or c.dim_lg is None or a.lam_label_dims <= 0:
            return None
        plan = self_.label_plan
        assert plan is not None  # a label plan is what produced lab_label upstream
        rcols = [cj for (cj, _, _) in plan]
        lab_lg = c.dim_lg[:, : c.lmax, 1:, :][:, :, rcols, :]  # [b, lmax, n_reserved, fsq_levels]
        loss_label = F.cross_entropy(
            lab_lg.reshape(-1, c.fsq_levels), c.lab_label.reshape(-1), ignore_index=-100
        )
        return (self.key, loss_label, a.lam_label_dims)


def build_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """DEFAULT loss-term set, built once from the args. Fixed order (label -> multi_nce -> hard_neg -> sup) so the
    float summation is byte-identical; each term self-skips when its weight/column is absent. Inject
    `TrainingArguments(loss_terms=...)` with your own builder (or add terms like CoTGenTerm) to change the set.
    NOTE: the phase head is no longer a term here — it is the `phase` instance of the generic AUXILIARY-HEAD plug
    (langset.heads), applied inline in `_train_multi` right after this loop (the same summation position the old
    `PhaseTerm` held, so lam_phase>0 stays byte-identical). `PhaseTerm` is kept for back-compat injection."""
    return [
        LabelDimsTerm(),
        MultiNCETerm(maskers=[identical_text_mask]),
        HardNegTerm(),
        SupConTerm(),
    ]


class CoTGenTerm(_LossTerm):
    """Exp-B: teach the model to GENERATE the row's chain-of-thought from the clean seed (doc=seed -> target=
    cot_text) via the tied embedding — the SAME CE machinery the latents use, co-trained in the same step.
    `isolated_backward` so its (long) CoT graph never coexists with the latent graph. Pairs with the seed+CoT
    conditioning that `cot_seed_texts` applies to the emission forward — inject BOTH (see build_cot_loss_terms).
    Self-skips when the batch's rows carry no CoT text (so it's inert if injected without a `cot_text` column)."""

    key = "loss_cot"
    isolated_backward = True

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, m, dev, self_ = c.args, c.model, c.dev, c.trainer
        if not any(
            self_.cot_texts[k] for k in c.bidx
        ):  # no reasoning in this batch -> nothing to learn
            return None
        tok, vsz = m.tokenizer, m.vocab_size

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

        # CoT blocks are long (p50~726, p90~1541 tok) -> keep full a.max_len, don't truncate hard. Pin padding sides
        # EXPLICITLY (not the tokenizer's mutable default): the SEED is LEFT-padded so its last real token lands at
        # index sd-1 (the position that predicts the first CoT token), and the CoT is RIGHT-padded so its real tokens
        # sit adjacent to the seed with pads trailing -> no padding between seed and CoT, and the CE never conditions
        # on a pad hidden. (Mirrors the emission forward's left-pad; a right-defaulting tokenizer would otherwise
        # silently condition short seeds' CoT on padding.)
        di, dm = _tokm([self_.input_text[k] for k in c.bidx], a.max_len, "left")
        ti, tm = _tokm([self_.cot_texts[k] or " " for k in c.bidx], a.max_len, "right")
        seq = torch.cat([di, ti], dim=1)
        am = torch.cat([dm, tm], dim=1)
        hid = m._last_hidden(m._run_backbone(m.embed(seq), am, seq, 0))
        sd = di.size(1)
        ph = hid[:, sd - 1 : sd - 1 + ti.size(1), :]  # hidden that predicts each CoT token
        # bf16 vocab projection (NOT .float()): the fp32 [b, T, |V|] logits were the OOM driver on the 80GB A100;
        # CE reduces the softmax in fp32 internally, so bf16 logits are a numerically fine training signal.
        lg = F.linear(ph, m.embed.weight)
        loss_cot = F.cross_entropy(
            lg.reshape(-1, vsz), ti.masked_fill(tm == 0, -100).reshape(-1), ignore_index=-100
        )
        return (self.key, loss_cot, a.lam_cot)


def build_cot_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """Exp-B loss set — INJECT via `TrainingArguments(loss_terms=build_cot_loss_terms)` (pair with
    `seed_builder=cot_seed_texts`). The DEFAULT terms plus CoTGenTerm (isolated-backward), so the model is
    co-trained to generate the row's reasoning while it emits the latents. Needs a `cot_text` dataset column."""
    return [*build_loss_terms(args), CoTGenTerm()]


# ---- emission objective -------------------------------------------------------------------------
@dataclass
class EmissionOut:
    """Result of one emission forward (shapes as in MultiStepCtx: B rows · L=lmax · d latent · V=fsq_levels).
    The trainer sets `loss = base_loss`, folds `logs` into its running averages, and passes `recon`/`dim_lg`/
    `lab_label` on to the aux terms via MultiStepCtx."""

    recon: torch.Tensor  # [B, L, d] the model's emitted latents — gradient flows through these
    base_loss: (
        torch.Tensor
    )  # scalar: the objective's own loss (FSQ: loss_stop + loss_dims + recon_loss)
    logs: dict[
        str, torch.Tensor
    ]  # UNWEIGHTED scalar components for logging (FSQ: loss_stop / loss_dims / recon_loss)
    dim_lg: Optional[
        torch.Tensor
    ]  # [B, L+1, fsq_dim, V] FSQ per-dim digit logits; None for a non-FSQ objective
    lab_label: Optional[
        torch.Tensor
    ]  # [B, L, n_reserved] reserved-dim label targets; None unless FSQ label subspace on


class _EmissionObjective:
    """Strategy for turning the seeded forward into emitted latents + the base emission loss. Default = FSQ
    (token-native digit CE + folded STOP + cosine recon). Selected ONCE per run so the step loop has no
    emission `if`s. `codebook` is a class flag the free-run rollout reads to pick which emission head to use
    (True = FSQ digit head, False = a raw-vector head). All objectives share the __init__ signature
    (model, args, dev, trainer) so they are interchangeable as an injected `TrainingArguments.emission`."""

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
        """Shape the per-row target item lists + their encoded latents into the emission's teacher-forcing tensors
        `(target_lat [B,L,d], valid [B,L], lens_l, lmax)`. EXTRACTED verbatim from the trainer's inline block so the
        STRATEGY owns target↔slot shaping: the AR/FSQ default is positional teacher forcing (item i -> slot i); a
        future matching family (DETR/parallel-query) overrides this (e.g. all-valid, assignment deferred to its own
        Hungarian match inside emit()). Pure tensor assembly — no model forward."""
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
        """Inference emission: texts -> (lat [B, Lmax, d], lens [B]), zero-padding halted rows. The eval and
        `model.rollout` route through here so a non-AR family (parallel-query) can emit in one pass. Default
        delegates to the model's autoregressive rollout (FSQ)."""
        lat, lens = self.m.rollout(  # ty: ignore[invalid-assignment]  # return_lengths=True -> (lat, lens)
            texts, max_steps=max_steps, return_lengths=True
        )
        return lat, lens

    def z_for_reg(
        self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor, lmax: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The (predicted, target) latents a TargetSource.regularizer (e.g. SIGReg) constrains. Default =
        the emitted vs target latents directly ([N, d]); FSQ overrides this to use the pre-quantization z."""
        return em.recon[valid], target_lat[valid]


class FSQObjective(_EmissionObjective):
    """DEFAULT emission: predict each item's per-dim FSQ digits (a STOP folded into dim-0's softmax) + a cosine
    reconstruction to the target. Byte-identical to the historical inline FSQ block. Reads the FSQ grid geometry
    (fsq_dim/fsq_levels) off model.head, so it takes the uniform (model, args, dev, trainer) signature."""

    codebook = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        super().__init__(model, args, dev, trainer)
        head = model.head
        self.fsq_dim = int(head.fsq_dim)
        self.fsq_levels = int(head.fsq_levels)
        self.stop_idx = self.fsq_levels  # STOP is the extra class folded into dim-0's softmax

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
        m, a, dev, self_ = self.m, self.a, self.dev, self.trainer
        fsq_dim, fsq_levels = self.fsq_dim, self.fsq_levels
        ss_prob = a.ss_prob
        assert ss_prob is not None  # Trainer resolves the None sentinel to a float before any emit
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        dim_lg, stop_lg, digits, recon = m.rollout_train_codebook(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            a.tau,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_sample=a.ss_sample,
            ss_mask=ss_mask,  # GradCache: shared per-(row,hop) self-feed decisions (deterministic replay)
            kv_cache=a.kv_cache,  # forward prompt once + single-token hops vs full-prefix recompute (no grad_ckpt)
        )
        dim0 = torch.cat([dim_lg[:, :, 0, :], stop_lg], -1)  # [b, lmax+1, L+1] — digit-0 + STOP
        lab0 = torch.full((b, lmax + 1), -100, dtype=torch.long, device=dev)
        lab_rest = torch.full((b, lmax, fsq_dim - 1), -100, dtype=torch.long, device=dev)
        for r, nl in enumerate(lens_l):
            lab0[r, :nl] = digits[r, :nl, 0]
            lab0[r, nl] = self.stop_idx  # emit digit-0 per item, then STOP after the last
            lab_rest[r, :nl] = digits[r, :nl, 1:]
        lab_label = None  # FSQ LABEL SUBSPACE: reserved dims -> a SEPARATE
        if self_.label_plan is not None:  # weighted label CE (NOT diluted inside loss_dims)
            plan = self_.label_plan
            cols = self_.label_cols
            assert cols is not None  # label_cols is populated alongside label_plan
            lab_label = torch.full((b, lmax, len(plan)), -100, dtype=torch.long, device=dev)
            for s_i, (col_j, field, pos) in enumerate(plan):
                labs, cw = cols[field], self_.label_codewords[field]
                for r, kk in enumerate(bidx):
                    row_labs = labs[kk]
                    for j in range(lens_l[r]):
                        code = cw.get(row_labs[j] if j < len(row_labs) else "")
                        lab_label[r, j, s_i] = code[pos] if code is not None else -100
                lab_rest[:, :, col_j] = -100  # reserved dims leave the reconstruction CE
        loss_stop = F.cross_entropy(
            dim0.reshape(-1, fsq_levels + 1), lab0.reshape(-1), ignore_index=-100
        )
        loss_dims = F.cross_entropy(
            dim_lg[:, :lmax, 1:, :].reshape(-1, fsq_levels), lab_rest.reshape(-1), ignore_index=-100
        )
        recon_loss = (1.0 - F.cosine_similarity(recon[valid], target_lat[valid], dim=-1)).mean()
        return EmissionOut(
            recon=recon,
            base_loss=loss_stop + loss_dims + recon_loss,
            logs={"loss_stop": loss_stop, "loss_dims": loss_dims, "recon_loss": recon_loss},
            dim_lg=dim_lg,
            lab_label=lab_label,
        )

    def z_for_reg(
        self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor, lmax: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Regularize the PRE-QUANTIZATION z = down_proj(latent) — the actual FSQ input, before the tanh+round —
        # so the penalty spreads the encoder's codes across the whole grid. z_pred = predicted E[digit].
        assert self.m.head.down_proj is not None
        z_tgt = self.m.head.down_proj(target_lat[valid].float())  # [N, fsq_dim]
        lvls = torch.arange(self.fsq_levels, device=self.dev, dtype=torch.float32)
        dim_lg = em.dim_lg
        assert dim_lg is not None  # FSQ path always populates dim_lg
        soft = (dim_lg[:, :lmax].float().softmax(-1) * lvls).sum(
            -1
        )  # predicted E[digit] [b, lmax, fsq_dim]
        return soft[valid], z_tgt


class CodeSoftmaxObjective(_EmissionObjective):
    """CODEBOOK emission: ONE softmax over the head's fixed codebook (plus STOP), trained toward the target's own
    distribution over codes. Inject with `TrainingArguments(emission=CodeSoftmaxObjective)` and a head built with
    `code_emit=True` (see EmitHead.set_code).

    Use this when an emission is a SET drawn from a known alphabet (maze frontier: which of 256 cells) rather than
    a point to quantize. The FSQ default spends `fsq_dim` independent softmaxes on a learned grid, and a per-member
    readout off that grid is `n_codes` INDEPENDENT decisions, so nothing forces the emission to choose among
    members: every member can be scored high at once. A single normalized softmax has to ALLOCATE its mass, so the
    members compete, and the emitted latent is the mixture they form.

    The target law is read off the target latent itself, no extra column: for an orthonormal codebook and a
    membership target `normalize(multi_hot @ code.T)`, projecting back with `target @ code.T` recovers the members
    (equal weight) and ~0 elsewhere, so an L1 normalize gives the uniform-over-members law to match. Loss is the
    soft-target cross-entropy over the codes, plus an INDEPENDENT sigmoid terminator (see `emit` for why STOP
    must not be folded into the membership softmax the way the FSQ path folds it into dim-0).
    """

    codebook = True

    def __init__(
        self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer
    ) -> None:
        super().__init__(model, args, dev, trainer)
        assert model.head.code_emit, (
            "CodeSoftmaxObjective needs a codebook head: build the model with code_emit=True, n_codes=<alphabet>"
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
        m, a, dev = self.m, self.a, self.dev
        ss_prob = a.ss_prob
        assert ss_prob is not None  # Trainer resolves the None sentinel before any emit
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        dim_lg, stop_lg, _digits, recon = m.rollout_train_codebook(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            a.tau,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_sample=a.ss_sample,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
        )
        code = m.head.code  # [n_codes, d] fixed
        with torch.no_grad():  # the target's law over codes, recovered from the target latent
            w = F.relu(target_lat.float() @ code.t())  # [b, lmax, n_codes]
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        # MEMBERSHIP: soft-target CE over the codes ALONE, at real emission slots.
        cell_logp = F.log_softmax(dim_lg[:, :lmax, 0, :].float(), -1)  # [b, lmax, n_codes]
        loss_code = -(w * cell_logp).sum(-1)[valid].mean()
        # TERMINATION: an INDEPENDENT sigmoid, not a class inside the membership softmax. Folding STOP in (the
        # FSQ default) is fair when dim-0's target is itself one-hot, but a set-valued target is 1/k-diffuse and
        # the two then share one normalizer: at a member position the target is 0 on STOP, so the gradient
        # pushing the stop logit DOWN is proportional to P(STOP) -- which shrinks as the set widens, because the
        # denominator carries k members. Wider sets suppress the terminator more weakly and it ratchets up.
        # Factored out, "continue vs stop" is trained by its own signal and never rides on set width.
        stop_lab = torch.zeros(b, lmax + 1, device=dev)
        keep = torch.zeros(b, lmax + 1, dtype=torch.bool, device=dev)
        for r, nl in enumerate(lens_l):
            stop_lab[r, nl] = 1.0  # continue through the tick's members, then stop after the last
            keep[r, : nl + 1] = True
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_lg.squeeze(-1).float()[keep], stop_lab[keep]
        )
        # `recon` is the contract every loss term reads as "the model's emission, with gradient" (MultiNCETerm
        # and HardNegTerm both build their query from it). The rollout's own recon is the TARGET here -- this
        # path is lossless, so it carries no gradient at all, and a term querying it would compare each target
        # against itself: near-zero loss, exactly zero gradient, silently inert. Hand back the DIFFERENTIABLE
        # committed mixture instead, which is both the true emission and the vector fed back each tick.
        mix = F.normalize(dim_lg[:, :lmax, 0, :].float().softmax(-1) @ code, dim=-1)  # [b, lmax, d]
        with (
            torch.no_grad()
        ):  # diagnostic: how close the emitted mixture lands to the target latent
            emit_cos = F.cosine_similarity(mix[valid], target_lat[valid], dim=-1).mean()
        del recon
        return EmissionOut(
            recon=mix,
            base_loss=loss_code + loss_stop,
            logs={"loss_code": loss_code, "loss_stop": loss_stop, "emit_cos": emit_cos},
            dim_lg=dim_lg,
            lab_label=None,  # the FSQ label subspace has no analogue here: one softmax, no dims to reserve
        )

    def z_for_reg(
        self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor, lmax: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Regularize the DISTRIBUTIONS, the codebook analogue of FSQ's pre-quantization z: predicted law vs target
        # law over codes, so a penalty spreads usage across the alphabet instead of collapsing onto a few codes.
        dim_lg = em.dim_lg
        assert dim_lg is not None
        code = self.m.head.code
        z_tgt = F.relu(target_lat[valid].float() @ code.t())
        z_tgt = z_tgt / z_tgt.sum(-1, keepdim=True).clamp_min(1e-9)
        return dim_lg[:, :lmax, 0, :].float().softmax(-1)[valid], z_tgt


class ConceptObjective(_EmissionObjective):
    """Emit a state as a superposition over NAMED CONCEPTS, per facet, with a residual for the unnamed.

    The row format is text the whole way down:

        {"input_text": …, "target_text": …,
         "concepts": {"vocals": ["yell-singing", "gang-vocals"], "mood": ["angry-but-vulnerable"]}}

    langset discovers each facet's alphabet by scanning the column, so no index is ever written by hand, and
    the emission's leading dims are CONSTITUTED from those concepts — project them back on the codebook and you
    get what the latent is made of, in your own vocabulary, with no probe. Each facet is normalized separately,
    so a wide `vocals` mixture does not make `tempo` look uncertain.

    Reach for this when the twin's geometry is good but WRONG SOMEWHERE. The twin gives a usable space for free
    and in-batch negatives separate it, but neither lets you say along WHICH axis. Naming one facet overwrites
    that neighbourhood and leaves the rest to the residual — so the normal configuration is a narrow state half
    and a wide residual, not the reverse. You are patching a geometry, not adopting an ontology.
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
        # Build the codebook from the DISCOVERED alphabet: every concept becomes a vector via `code_source`,
        # laid out per facet. Once, at setup, over the concept names — then frozen for the run.
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
        m, a, dev = self.m, self.a, self.dev
        ss_prob = a.ss_prob
        assert ss_prob is not None
        eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
        dim_lg, stop_lg, _d, _r, emit_hid = m.rollout_train_codebook(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            a.tau,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_sample=a.ss_sample,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
            return_emit_hidden=True,
        )
        flat = dim_lg[:, :lmax, 0, :].float()  # [b, lmax, n_codes]
        tgt, seen = self._target_law(b, lmax, lens_l, bidx)

        # one soft-target CE PER FACET, over that facet's members only
        losses, per_facet = [], {}
        for fi, (m_lo, m_hi, _, _) in enumerate(self.spans):
            sel = seen[:, :, fi] & valid
            if not bool(sel.any()):
                continue
            logp = F.log_softmax(flat[..., m_lo:m_hi], -1)
            li = -(tgt[..., m_lo:m_hi] * logp).sum(-1)[sel].mean()
            losses.append(li)
            per_facet[f"c_{self.facets[fi]}"] = li
        loss_concept = torch.stack(losses).sum() if losses else flat.new_zeros(())

        stop_lab = torch.zeros(b, lmax + 1, device=dev)
        keep = torch.zeros(b, lmax + 1, dtype=torch.bool, device=dev)
        for r, nl in enumerate(lens_l):
            stop_lab[r, nl] = 1.0
            keep[r, : nl + 1] = True
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_lg.squeeze(-1).float()[keep], stop_lab[keep]
        )

        # the emission the rest of the trainer sees — WITH gradient, or every aux term silently idles
        p = m.head.concept_probs(dim_lg[:, :lmax])
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
            dim_lg=dim_lg,
            lab_label=None,
        )


class StateResidualObjective(_EmissionObjective):
    """STATE + RESIDUAL emission: a named half and an unnamed half, in one latent.

    The latent splits at `head.res_dim`. The STATE half is a mixture over a fixed alphabet of members you can
    name, trained by soft-target cross-entropy against the members that are actually live this tick. The
    RESIDUAL half is trained by whatever the surrounding terms already do (recon to the target, in-batch NCE),
    so it carries what the alphabet cannot say.

    Why both. A named alphabet is the strongest signal available -- the emission is CONSTITUTED from the members
    rather than having them decoded out of it -- but it is also a ceiling: anything the alphabet cannot name,
    the emission cannot hold, and a world model whose ontology has gaps will silently refuse to represent them.
    The residual is where the unnamed goes, and how much it carries is measurable: ablate it at eval and the
    drop is how incomplete the alphabet was.

    It also keeps a self-supervised component in the design. The state half's meaning is externally specified
    (you chose the members); the residual half's is not, so the latent is not wholly hand-defined.

    Labels come from a per-row column named by `state_field`: a per-tick list of ACTIVE member indices (sparse,
    e.g. `[[3, 47], [12], ...]`). Mass is split evenly over each tick's live members, so a tick holding four
    members asks the emission to spread over four -- the superposition, written as the target.

      TrainingArguments(emission=StateResidualObjective, state_field="frontier", state_classes=256)
      LangSetModel.from_pretrained(..., code_emit=True, n_codes=256, res_dim=64)
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
        # Build the codebook ONCE, here, from the injected `code_source` — over the alphabet's NAMES, not over
        # the training data. Skipped when a codebook was already installed by hand (head.set_code before train).
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
        dim_lg, stop_lg, _digits, _recon, emit_hid = m.rollout_train_codebook(
            se["input_ids"],
            se["attention_mask"],
            target_lat,
            a.tau,
            train_hops=a.train_hops,
            ss_prob=eff_ss,
            ss_sample=a.ss_sample,
            ss_mask=ss_mask,
            kv_cache=a.kv_cache,
            return_emit_hidden=True,  # the residual is a function of the emit hidden, not of the logits
        )
        code = m.head.code

        # STATE: soft-target CE over the alphabet, mass split evenly across each tick's live members.
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
        cell_logp = F.log_softmax(dim_lg[:, :lmax, 0, :].float(), -1)
        sel = has & valid
        loss_state = (
            -(tgt * cell_logp).sum(-1)[sel].mean() if bool(sel.any()) else cell_logp.new_zeros(())
        )

        # TERMINATION: its own sigmoid, never folded into the member softmax. Folding is only fair when the
        # member target is one-hot; against a 1/k-diffuse target the gradient suppressing STOP scales with
        # P(STOP), which weakens as the set widens, so the rollout truncates exactly where sets get wide.
        stop_lab = torch.zeros(b, lmax + 1, device=dev)
        keep = torch.zeros(b, lmax + 1, dtype=torch.bool, device=dev)
        for r, nl in enumerate(lens_l):
            stop_lab[r, nl] = 1.0
            keep[r, : nl + 1] = True
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_lg.squeeze(-1).float()[keep], stop_lab[keep]
        )

        # THE EMISSION the rest of the trainer sees: state mixture ++ residual, exactly what feeds back. It must
        # be the emission WITH gradient -- every aux term (in-batch NCE, hard negatives) builds its query from
        # `recon`, and handing back the stop-grad target instead leaves them running but training nothing.
        p = dim_lg[:, :lmax, 0, :].float().softmax(-1)
        state = F.normalize(p @ code, dim=-1)
        if self.res_dim:
            res = m.head.residual(emit_hid[:, :lmax])
            emission = torch.cat([state, res], -1) * (0.5**0.5)
        else:
            emission = state
        with torch.no_grad():
            emit_cos = F.cosine_similarity(emission[valid], target_lat[valid], dim=-1).mean()
            res_share = (
                (emission[valid][:, -self.res_dim :].norm(dim=-1) ** 2).mean()
                if self.res_dim
                else torch.zeros((), device=dev)
            )
        logs = {"loss_state": loss_state, "loss_stop": loss_stop, "emit_cos": emit_cos}
        if self.res_dim:
            logs["res_share"] = res_share  # how much of the emission the alphabet could NOT name
        return EmissionOut(
            recon=emission,
            base_loss=loss_state + loss_stop,
            logs=logs,
            dim_lg=dim_lg,
            lab_label=None,
        )


# ---- target source ------------------------------------------------------------------------------
class _TargetSource:
    """Strategy for the TARGET latents the emission trains toward, plus any anti-collapse regularization.
    Default = stop-grad EMA twin. Selected ONCE per run. All sources share the __init__ signature
    (model, args, tok, dev) so they are interchangeable as an injected `TrainingArguments.target_source`."""

    suppresses_nce: bool = (
        False  # if True the trainer skips the in-batch NCE term (a live-target source that
    )
    #                                     already prevents collapse via `regularizer` doesn't need — and fights — it)
    wants_regularizer: bool = (
        False  # if True the trainer computes objective.z_for_reg and adds `regularizer` to the
    )
    #                                     loss; keeps that (non-trivial) work off the default path when False
    twin: Optional[LangSetModel] = (
        None  # the model the EVAL block encodes its retrieval bank with (the EMA copy for the
    )
    #                                      default; the online model itself for a live-target source). Set by subclasses.

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Emit each text -> [n, d] L2-normalized target latents (no grad for the EMA default). Used both for the
        per-step targets and, via MultiStepCtx.target_source, for the hard-negative bank."""
        raise NotImplementedError

    def update(self) -> None:
        """Called once AFTER each opt.step(). The EMA default nudges the twin toward the online weights; a
        live-target source has nothing to track, so this is a no-op."""

    def regularizer(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional extra anti-collapse loss on the emitted (`z_pred`) and target (`z_tgt`) latents, added to the
        step loss. None for the EMA default (the stop-grad twin is what prevents collapse there)."""
        return None


class EMATwinTarget(_TargetSource):
    """DEFAULT: a stop-grad EMA copy of the online model supplies the target latents (BYOL/JEPA) so both sides
    don't move together and collapse. Byte-identical to the historical inline twin + emit_texts + ema_update."""

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
        # Single-latent emission of each text -> [N, d] normalized, no_grad. Truncated to target_max_len
        # (default 64: targets are short descriptors; raise it when a target is a DOCUMENT, e.g. emit_seed's
        # phase-0 target is a full abstract). Short future strings are already < 64 so unaffected.
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
    """FROZEN-ENCODER target source with an encode-once cache — the two-stage (V-JEPA) split for the multi-latent
    world model. The target geometry is a FIXED encoder, so every text->latent map is CONSTANT for the whole run:
    encode each unique text ONCE, memoize, then serve lookups. The step loop stops re-encoding BOTH the per-step
    targets (trainer flat_texts) and the batch-pooled hard-negative bank (HardNegTerm) — bringing the 'train only
    the head on cached vectors, epochs in seconds' win that langset's single-latent frozen-pool path already has
    to the multi-latent rollout. It's the fix for re-encoding fixed data every epoch.

    Encoder source: `args.target_encoder_ckpt` (a saved LangSetModel dir) when set — the intended use, a SEPARATE
    already-good geometry (e.g. a trained affordance/embedding model) that the emitter learns to roll FORWARD in,
    so no encoder is co-trained at all. Otherwise a frozen snapshot of the online model at init (only meaningful
    when it already starts from a good encoder). INJECT via
    `TrainingArguments(target_source=CachedTarget, target_encoder_ckpt="path/to/encoder")`. `update()` is a no-op
    (the geometry is fixed) and the eval retrieval bank encodes through the same frozen geometry (`twin = enc`)."""

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
    """EMA-free anti-collapse (LeJEPA, arXiv:2511.08544). INJECT via `TrainingArguments(target_source=SIGRegTarget)`.
    Targets come from the LIVE model WITH gradient (no stop-grad twin); collapse is prevented by an isotropic-Gaussian
    SIGReg penalty on the pre-quant z (via `regularizer`) instead of by a twin. So it drops the twin's VRAM + target
    forward, the in-batch NCE is suppressed (the regularizer replaces it), and eval encodes with the live model itself.
    Reads scalar knobs off args: sigreg_lambda (loss weight, applied in the trainer), sigreg_knots, sigreg_slices."""

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
        # LIVE target WITH gradient (no no_grad, no twin): both the emitted and target latents move, and SIGReg —
        # not a stop-grad twin — is what stops them collapsing together.
        a, tok, dev = self.a, self.tok, self.dev
        e = tok(
            texts, padding=True, truncation=True, max_length=a.target_max_len, return_tensors="pt"
        ).to(dev)
        z = self.m(e["input_ids"], e["attention_mask"])
        return F.normalize(z.float(), dim=-1)

    def regularizer(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Optional[torch.Tensor]:
        # Two INDEPENDENT Gaussianity penalties (predicted E[digit] and target z), NOT a match between them —
        # each is pushed toward isotropic Gaussian, spreading codes across the FSQ grid.
        return self.sig_reg(z_pred) + self.sig_reg(z_tgt)


# ---- concepts (the text-in format for a named, superposed state) ---------------------------------
# A row carries a `concepts` column of NAMED FACETS, each holding the concepts that are true of it:
#
#     "concepts": {"vocals": ["yell-singing", "gang-vocals"], "tempo": {"7": 0.1, "8": 0.9}}
#
# A list means "these are all true, equally" (mass split evenly — the superposition). A dict means explicit
# weights, which also encodes a CONTINUOUS value as a mixture over ordered concepts: 7.9 is 0.1 of "7" and 0.9
# of "8", interpolation included, and unlike a regressed scalar you can still read what it says.
#
# Everything is text. The alphabet is not configured — it is DISCOVERED by scanning the column, the way a
# tokenizer's vocabulary is, so nobody writes or maintains an index. Multi-latent rows pass a LIST of these
# dicts, one per tick, aligned with `target_texts`.
def parse_concepts(raw: object) -> "dict[str, dict[str, float]]":
    """One row's (or tick's) concepts -> {facet: {concept: weight}}, weights normalized within each facet."""
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
    """Scan the corpus's concept column -> {facet: sorted concept names}. Sorted for determinism, so the same
    corpus always yields the same layout and a checkpoint stays readable."""
    seen: dict[str, set[str]] = {}
    for raw in rows_concepts:
        ticks = raw if isinstance(raw, (list, tuple)) else [raw]
        for tick in ticks:
            for facet, w in parse_concepts(tick).items():
                seen.setdefault(facet, set()).update(w)
    return {f: sorted(v) for f, v in sorted(seen.items())}


# ---- code sources (where a named member's vector comes from) -------------------------------------
# A `code_source` maps the alphabet's member NAMES to their vectors: (names, dim, model) -> [n_members, dim].
# Called ONCE at setup, over the alphabet (hundreds to a few thousand short strings), never per batch. The
# result is frozen into a buffer for the run — a codebook that re-embeds while the model trains is a learned
# codebook in disguise, and re-opens the collapse problem the fixed FSQ grid was chosen to avoid.
#
# The choice is a real trade, not a default: an orthonormal codebook decodes losslessly by plain matmul (so a
# recall number measures the EMISSION, with no probe as a confound) but its members carry no relation to each
# other; embedded codes put `ph2` near `ph3` and `melanoma` near `breast_neoplasms`, at the cost of exact
# recovery. Benchmarks want the first, real domains usually want the second.
def random_orthonormal_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Arbitrary but perfectly decodable: a seeded orthonormal frame, one row per member (`C Cᵀ = I`).

    Member vectors are unrelated by construction — cell 10 and cell 11 are as orthogonal as cell 10 and cell
    200 — so nothing about the geometry can flatter a result. That is exactly why it belongs on a benchmark."""
    assert len(names) <= dim, (
        f"random_orthonormal_codes needs dim >= n_members for an orthonormal frame; got dim={dim}, "
        f"n_members={len(names)}. Use model_embedded_codes (no such limit) or widen the state half."
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.linalg.qr(torch.randn(dim, len(names), generator=g)).Q[:, : len(names)]
    return q.t().contiguous()


def model_embedded_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """The base model's own reading of each member's NAME, mean-pooled over its tokens.

    Semantically related members land near each other, which is information the random frame throws away, and
    the codebook becomes model-derived rather than hand-designed. Not orthonormal, so a mixture no longer
    recovers its members exactly — read out with top-k over member scores rather than expecting a clean inverse.
    Unlimited alphabet size (no dim >= n_members constraint)."""
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
    """Embed the names, then orthogonalize — keeps a lossless readout and as much of the semantic arrangement
    as an orthonormal frame can hold. The distortion is the price; the members are no longer purely the model's
    own vectors."""
    e = model_embedded_codes(names, dim, model)
    q = torch.linalg.qr(e.t().float()).Q[:, : len(names)]
    return q.t().contiguous()


def twin_encoded_codes(names: list[str], dim: int, model: LangSetModel) -> torch.Tensor:
    """Encode each member name with the model's own emit path, so the codebook lands in the SAME space as the
    training targets.

    This is the coherent option: the state mixture and the twin's target then share one geometry, so the state
    loss and the recon loss are talking about the same thing, and the residual is precisely the part of the
    target the named members cannot span. Requires the emit path to be usable at setup."""
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
    """DEFAULT epoch ordering: a plain shuffle of the training positions. Inject a different `epoch_order` to
    change it."""
    return torch.randperm(len(tr_idx), generator=rng_t).tolist()


def multi_select_metric(mode: str, mrr: float, pur: float, ep: int) -> float:
    """DEFAULT checkpoint-selection signal from the epoch's metrics. retr_mrr (default) / purity / blend. Inject
    a different `selector` to change it (e.g. one that keeps the last epoch)."""
    return pur if mode == "purity" else (mrr + pur) if mode == "blend" else mrr


def last_epoch_selector(mode: str, mrr: float, pur: float, ep: int) -> float:
    """SUPERPOSITION selector — INJECT via `TrainingArguments(selector=last_epoch_selector)`. No early-stop signal;
    keeps the LAST epoch (returns float(ep)). Under superposition training retr_mrr selects for a collapsed
    one-future-per-seed geometry — exactly the wrong target when you WANT the latent to spread over a seed's
    alternative futures, so retr_mrr is meant to fall and must not gate selection."""
    return float(ep)


# The trainer evaluates/selects only on `ep % eval_every == 0`; a "keep the last epoch" selector must still see the
# FINAL epoch even when eval_every>1, or it silently restores an earlier (last-evaluated) epoch. This flag tells the
# trainer to always evaluate the final epoch for this selector; the default selector lacks it, so its path is unchanged.
last_epoch_selector.needs_final_epoch = True  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # function-attribute flag


def multi_seed_texts(trainer: Trainer, seeds: list[str], args: TrainingArguments) -> list[str]:
    """DEFAULT texts fed to the EMISSION forward — what the model reads before emitting its latents = the raw
    input seeds. Inject a different `seed_builder` to change it (e.g. append per-row CoT); targets/eval keep raw seeds."""
    return seeds


def cot_seed_texts(trainer: Trainer, seeds: list[str], args: TrainingArguments) -> list[str]:
    """Exp-B seed-builder — INJECT via `TrainingArguments(seed_builder=cot_seed_texts)` (pair with
    `loss_terms=build_cot_loss_terms`). Conditions the emission forward on each row's teacher-forced reasoning
    (seed + CoT) so the latents are emitted AFTER the reasoning; targets and eval keep the raw seeds, and
    CoTGenTerm trains the model to produce that reasoning itself."""
    return [f"{s}\n\nReasoning:\n{trainer.cot_texts[i]}" for i, s in enumerate(seeds)]
