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
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.nn.functional as F

from langset.modeling import LangSetModel
from langset.sigreg import SIGReg

if TYPE_CHECKING:                                    # annotations only (from __future__ import annotations -> strings);
    from langset.trainer import Trainer              # avoids a runtime import cycle trainer <-> strategies
    from langset.training_args import TrainingArguments


def supcon_loss(z: torch.Tensor, labels: list[str], tau: float) -> torch.Tensor:
    """Supervised-contrastive (Khosla et al.) over emitted latents: same-label items are positives (pulled together),
    all others negatives (pushed apart) — a few group labels SHAPE the geometry into SEPARATE REGIONS. Being a proper
    contrastive loss (each anchor has both positives and negatives) it separates without collapsing. Items whose label
    is ''/'unknown'/'none'/'nan' are dropped. Returns 0 if fewer than two labelled items or no positive pair exists."""
    dev = z.device
    keep = [k for k, l in enumerate(labels) if str(l).strip().lower() not in ("", "unknown", "none", "nan")]
    if len(keep) < 2:
        return z.new_zeros(())
    zz = F.normalize(z[keep], p=2, dim=-1)
    lab = [labels[k] for k in keep]
    b = len(keep)
    sim = (zz @ zz.t() / tau).masked_fill(torch.eye(b, device=dev, dtype=torch.bool), -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = torch.tensor([[1.0 if (i != j and lab[i] == lab[j]) else 0.0 for j in range(b)] for i in range(b)],
                       device=dev)
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
    trainer: Trainer                    # the owning Trainer; read its PER-ROW data (indexed by dataset row id):
    #                                     sup_labels / hard_neg_texts / label_plan + label_cols + label_codewords
    args: TrainingArguments             # the run config — a term reads its own weight/temperature here (a.lam_*, a.tau)
    model: LangSetModel                 # the online model being trained (rarely needed directly — emit via target_source)
    dev: torch.device                   # device every tensor below lives on; build new tensors with device=c.dev
    bidx: list[int]                     # this batch's DATASET ROW IDS (len B) — index trainer.sup_labels[k], etc. with these
    lens_l: list[int]                   # emitted-item count per row (len B): row r produced lens_l[r] items; Σ = N
    flat_texts: list[str]               # the N target texts row-major (row0's items, then row1's, ...), aligned to recon[valid]
    valid: torch.Tensor                 # [B, L] bool mask of real (non-padding) emission slots; recon[valid] -> [N, d]
    target_lat: torch.Tensor            # [B, L, d] the stop-grad TARGET latents each emission is trained toward
    recon: torch.Tensor                 # [B, L, d] the model's EMITTED latents this step — gradient flows through these
    dim_lg: Optional[torch.Tensor]      # [B, L+1, fsq_dim, V] FSQ per-dim digit logits; None for a non-FSQ objective
    lmax: int                           # L above: the padded emitted-item time dim for this batch
    fsq_levels: int                     # V above: FSQ quantization levels per digit
    lab_label: Optional[torch.Tensor]   # [B, L, n_reserved] reserved-dim label targets; None unless FSQ label subspace on
    target_source: _TargetSource        # the target provider; call .encode(texts) -> [n, d] normalized latents (hard-neg bank)
    phase_head: Optional[torch.nn.Module]  # transient hidden->phase linear classifier, or None when lam_phase == 0
    phase_ids: dict[str, int]           # phase-label string -> class index, the CE targets for phase_head


# ---- aux loss terms -----------------------------------------------------------------------------
class _LossTerm:
    """Strategy for one weighted, optional term added on top of the base emission loss (one per historical
    `if a.lam_x > 0:` block). Terms are built once and iterated each step; each self-skips when inapplicable."""
    key: str = ""                       # this term's log/agg name (e.g. "loss_multi_nce"); set by each subclass
    isolated_backward: bool = False     # if True the term is NOT summed into the shared loss; instead the trainer runs
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
    for ii, tx in enumerate(c.flat_texts):               # flat_texts is row-major aligned with recon[valid]
        grp.setdefault(tx, []).append(ii)
    for mem in grp.values():
        if len(mem) > 1:
            for aa in mem:
                for bb in mem:
                    if aa != bb:
                        fn_mask[aa, bb] = True


def same_seed_mask(c: MultiStepCtx, fn_mask: torch.Tensor) -> None:
    """SUPERPOSITION negative-mask (an extra masker — inject via build_superposition_loss_terms): also MUTATES
    `fn_mask` in place so the alternative futures of ONE seed aren't negatives of each other — then the emitted
    latent can settle at their centroid (the mixture) instead of being pushed apart. Seed per emission = its row's
    input_text, repeated lens_l[r] times to stay aligned with recon[valid]."""
    seed_flat = [c.trainer.input_text[k] for r, k in enumerate(c.bidx) for _ in range(c.lens_l[r])]
    grp: dict[str, list[int]] = {}
    for ii, sd in enumerate(seed_flat):
        grp.setdefault(sd, []).append(ii)
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
        if c.target_source.suppresses_nce:                   # e.g. SIGReg replaces the NCE with its regularizer
            return None
        if not (a.lam_multi_nce > 0 and int(c.valid.sum()) > 1):
            return None
        rvn = F.normalize(c.recon[c.valid], dim=-1)          # [N, d] emitted (gradient flows here)
        tvn = F.normalize(c.target_lat[c.valid], dim=-1)     # [N, d] EMA targets (already stop-grad)
        nce_logits = (rvn @ tvn.t()) / a.tau                 # [N, N] query x key cosine / temp
        n_nce = rvn.size(0)
        fn_mask = torch.zeros(n_nce, n_nce, dtype=torch.bool, device=c.dev)
        for masker in self.maskers:
            masker(c, fn_mask)
        nce_logits = nce_logits.masked_fill(fn_mask, float("-inf"))   # diagonal (positive) never masked
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
        hn_bank = c.target_source.encode(hn_flat)            # [Nhn, d] stop-grad normalized hard-neg latents
        rv = F.normalize(c.recon[c.valid], dim=-1)           # [Nvalid, d] emitted reconstructions
        pos = (rv * c.target_lat[c.valid]).sum(-1, keepdim=True)   # [Nvalid, 1] cos to own target
        neg = rv @ hn_bank.t()                               # [Nvalid, Nhn] cos to every hard neg
        logits_hn = torch.cat([pos, neg], dim=1) / a.tau
        loss_hn = F.cross_entropy(logits_hn, torch.zeros(logits_hn.size(0), dtype=torch.long, device=c.dev))
        return (self.key, loss_hn, a.lam_hard_neg)


class SupConTerm(_LossTerm):
    """Supervised-contrastive shaping over emitted latents by the per-item `sup_field` group labels."""
    key = "loss_sup"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if self_.sup_labels is None or a.lam_sup <= 0:
            return None
        sup_flat = [(self_.sup_labels[k][j] if j < len(self_.sup_labels[k]) else "unknown")
                    for r, k in enumerate(c.bidx) for j in range(c.lens_l[r])]
        loss_sup = supcon_loss(c.recon[c.valid], sup_flat, a.sup_tau)   # pull same-stage, push different-stage
        return (self.key, loss_sup, a.lam_sup)


class PhaseTerm(_LossTerm):
    """CE phase classifier on the emitted reconstruction (non-collapsing SupCon alternative)."""
    key = "loss_phase"

    def contribute(self, c: MultiStepCtx) -> Optional[tuple[str, torch.Tensor, float]]:
        a, self_ = c.args, c.trainer
        if c.phase_head is None:
            return None
        pf = [(self_.sup_labels[k][j] if j < len(self_.sup_labels[k]) else "")
              for r, k in enumerate(c.bidx) for j in range(c.lens_l[r])]
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
        rcols = [cj for (cj, _, _) in self_.label_plan]
        lab_lg = c.dim_lg[:, :c.lmax, 1:, :][:, :, rcols, :]     # [b, lmax, n_reserved, fsq_levels]
        loss_label = F.cross_entropy(lab_lg.reshape(-1, c.fsq_levels), c.lab_label.reshape(-1), ignore_index=-100)
        return (self.key, loss_label, a.lam_label_dims)


def build_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """DEFAULT loss-term set, built once from the args. Fixed order (label -> multi_nce -> hard_neg -> sup ->
    phase) so the float summation is byte-identical; each term self-skips when its weight/column is absent. Inject
    `TrainingArguments(loss_terms=...)` with your own builder (or add terms like CoTGenTerm) to change the set."""
    return [
        LabelDimsTerm(),
        MultiNCETerm(maskers=[identical_text_mask]),
        HardNegTerm(),
        SupConTerm(),
        PhaseTerm(),
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
        if not any(self_.cot_texts[k] for k in c.bidx):        # no reasoning in this batch -> nothing to learn
            return None
        tok, vsz = m.tokenizer, m.vocab_size

        def _tokm(texts: list[str], mx: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
            e = tok(texts, padding=True, truncation=True, max_length=mx, padding_side=side, return_tensors="pt")
            return e["input_ids"].to(dev), e["attention_mask"].to(dev)

        # CoT blocks are long (p50~726, p90~1541 tok) -> keep full a.max_len, don't truncate hard. Pin padding sides
        # EXPLICITLY (not the tokenizer's mutable default): the SEED is LEFT-padded so its last real token lands at
        # index sd-1 (the position that predicts the first CoT token), and the CoT is RIGHT-padded so its real tokens
        # sit adjacent to the seed with pads trailing -> no padding between seed and CoT, and the CE never conditions
        # on a pad hidden. (Mirrors the emission forward's left-pad; a right-defaulting tokenizer would otherwise
        # silently condition short seeds' CoT on padding.)
        di, dm = _tokm([self_.input_text[k] for k in c.bidx], a.max_len, "left")
        ti, tm = _tokm([self_.cot_texts[k] or " " for k in c.bidx], a.max_len, "right")
        seq = torch.cat([di, ti], dim=1); am = torch.cat([dm, tm], dim=1)
        hid = m._last_hidden(m._run_backbone(m.embed(seq), am, seq, 0))
        sd = di.size(1)
        ph = hid[:, sd - 1: sd - 1 + ti.size(1), :]                     # hidden that predicts each CoT token
        # bf16 vocab projection (NOT .float()): the fp32 [b, T, |V|] logits were the OOM driver on the 80GB A100;
        # CE reduces the softmax in fp32 internally, so bf16 logits are a numerically fine training signal.
        lg = F.linear(ph, m.embed.weight)
        loss_cot = F.cross_entropy(lg.reshape(-1, vsz), ti.masked_fill(tm == 0, -100).reshape(-1), ignore_index=-100)
        return (self.key, loss_cot, a.lam_cot)


def build_cot_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """Exp-B loss set — INJECT via `TrainingArguments(loss_terms=build_cot_loss_terms)` (pair with
    `seed_builder=cot_seed_texts`). The DEFAULT terms plus CoTGenTerm (isolated-backward), so the model is
    co-trained to generate the row's reasoning while it emits the latents. Needs a `cot_text` dataset column."""
    return [*build_loss_terms(args), CoTGenTerm()]


def build_superposition_loss_terms(args: TrainingArguments) -> list[_LossTerm]:
    """SUPERPOSITION loss set — INJECT via `TrainingArguments(loss_terms=build_superposition_loss_terms)`. Same as
    the default, but the in-batch InfoNCE also gets `same_seed_mask`, so two branches of ONE seed aren't pushed
    apart and the emitted latent can settle at their mixture. Pair with `epoch_order=grouped_epoch_order` (branches
    contiguous) and `selector=last_epoch_selector` (retr_mrr is meant to fall here, so it must not gate selection)."""
    return [
        LabelDimsTerm(),
        MultiNCETerm(maskers=[identical_text_mask, same_seed_mask]),
        HardNegTerm(),
        SupConTerm(),
        PhaseTerm(),
    ]


# ---- emission objective -------------------------------------------------------------------------
@dataclass
class EmissionOut:
    """Result of one emission forward (shapes as in MultiStepCtx: B rows · L=lmax · d latent · V=fsq_levels).
    The trainer sets `loss = base_loss`, folds `logs` into its running averages, and passes `recon`/`dim_lg`/
    `lab_label` on to the aux terms via MultiStepCtx."""
    recon: torch.Tensor                 # [B, L, d] the model's emitted latents — gradient flows through these
    base_loss: torch.Tensor             # scalar: the objective's own loss (FSQ: loss_stop + loss_dims + recon_loss)
    logs: dict[str, torch.Tensor]       # UNWEIGHTED scalar components for logging (FSQ: loss_stop / loss_dims / recon_loss)
    dim_lg: Optional[torch.Tensor]      # [B, L+1, fsq_dim, V] FSQ per-dim digit logits; None for a non-FSQ objective
    lab_label: Optional[torch.Tensor]   # [B, L, n_reserved] reserved-dim label targets; None unless FSQ label subspace on


class _EmissionObjective:
    """Strategy for turning the seeded forward into emitted latents + the base emission loss. Default = FSQ
    (token-native digit CE + folded STOP + cosine recon). Selected ONCE per run so the step loop has no
    emission `if`s. `codebook` is a class flag the free-run rollout reads to pick which emission head to use
    (True = FSQ digit head, False = a raw-vector head). All objectives share the __init__ signature
    (model, args, dev, trainer) so they are interchangeable as an injected `TrainingArguments.emission`."""
    codebook: bool = True

    def __init__(self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer) -> None:
        self.m, self.a, self.dev, self.trainer = model, args, dev, trainer

    def emit(self, se: dict[str, torch.Tensor], target_lat: torch.Tensor, valid: torch.Tensor,
             lens_l: list[int], bidx: list[int], b: int, lmax: int, ep: int) -> EmissionOut:
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

    def z_for_reg(self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor,
                  lmax: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (predicted, target) latents a TargetSource.regularizer (e.g. SIGReg) constrains. Default =
        the emitted vs target latents directly ([N, d]); FSQ overrides this to use the pre-quantization z."""
        return em.recon[valid], target_lat[valid]


class FSQObjective(_EmissionObjective):
    """DEFAULT emission: predict each item's per-dim FSQ digits (a STOP folded into dim-0's softmax) + a cosine
    reconstruction to the target. Byte-identical to the historical inline FSQ block. Reads the FSQ grid geometry
    (fsq_dim/fsq_levels) off model.head, so it takes the uniform (model, args, dev, trainer) signature."""
    codebook = True

    def __init__(self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer) -> None:
        super().__init__(model, args, dev, trainer)
        head = model.head
        self.fsq_dim = int(head.fsq_dim)
        self.fsq_levels = int(head.fsq_levels)
        self.stop_idx = self.fsq_levels                      # STOP is the extra class folded into dim-0's softmax

    def emit(self, se: dict[str, torch.Tensor], target_lat: torch.Tensor, valid: torch.Tensor,
             lens_l: list[int], bidx: list[int], b: int, lmax: int, ep: int) -> EmissionOut:
        m, a, dev, self_ = self.m, self.a, self.dev, self.trainer
        fsq_dim, fsq_levels = self.fsq_dim, self.fsq_levels
        eff_ss = a.ss_prob if a.ss_warmup <= 0 else a.ss_prob * min(1.0, ep / a.ss_warmup)
        dim_lg, stop_lg, digits, recon = m.rollout_train_codebook(
            se["input_ids"], se["attention_mask"], target_lat, a.tau,
            train_hops=a.train_hops, ss_prob=eff_ss, ss_sample=a.ss_sample)
        dim0 = torch.cat([dim_lg[:, :, 0, :], stop_lg], -1)  # [b, lmax+1, L+1] — digit-0 + STOP
        lab0 = torch.full((b, lmax + 1), -100, dtype=torch.long, device=dev)
        lab_rest = torch.full((b, lmax, fsq_dim - 1), -100, dtype=torch.long, device=dev)
        for r, nl in enumerate(lens_l):
            lab0[r, :nl] = digits[r, :nl, 0]
            lab0[r, nl] = self.stop_idx                          # emit digit-0 per item, then STOP after the last
            lab_rest[r, :nl] = digits[r, :nl, 1:]
        lab_label = None                                         # FSQ LABEL SUBSPACE: reserved dims -> a SEPARATE
        if self_.label_plan is not None:                         # weighted label CE (NOT diluted inside loss_dims)
            lab_label = torch.full((b, lmax, len(self_.label_plan)), -100, dtype=torch.long, device=dev)
            for s_i, (col_j, field, pos) in enumerate(self_.label_plan):
                labs, cw = self_.label_cols[field], self_.label_codewords[field]
                for r, kk in enumerate(bidx):
                    row_labs = labs[kk]
                    for j in range(lens_l[r]):
                        code = cw.get(row_labs[j] if j < len(row_labs) else "")
                        lab_label[r, j, s_i] = code[pos] if code is not None else -100
                lab_rest[:, :, col_j] = -100                     # reserved dims leave the reconstruction CE
        loss_stop = F.cross_entropy(dim0.reshape(-1, fsq_levels + 1), lab0.reshape(-1), ignore_index=-100)
        loss_dims = F.cross_entropy(dim_lg[:, :lmax, 1:, :].reshape(-1, fsq_levels),
                                    lab_rest.reshape(-1), ignore_index=-100)
        recon_loss = (1.0 - F.cosine_similarity(recon[valid], target_lat[valid], dim=-1)).mean()
        return EmissionOut(recon=recon, base_loss=loss_stop + loss_dims + recon_loss,
                           logs={"loss_stop": loss_stop, "loss_dims": loss_dims, "recon_loss": recon_loss},
                           dim_lg=dim_lg, lab_label=lab_label)

    def z_for_reg(self, em: EmissionOut, target_lat: torch.Tensor, valid: torch.Tensor,
                  lmax: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Regularize the PRE-QUANTIZATION z = down_proj(latent) — the actual FSQ input, before the tanh+round —
        # so the penalty spreads the encoder's codes across the whole grid. z_pred = predicted E[digit].
        assert self.m.head.down_proj is not None
        z_tgt = self.m.head.down_proj(target_lat[valid].float())            # [N, fsq_dim]
        lvls = torch.arange(self.fsq_levels, device=self.dev, dtype=torch.float32)
        soft = (em.dim_lg[:, :lmax].float().softmax(-1) * lvls).sum(-1)     # predicted E[digit] [b, lmax, fsq_dim]
        return soft[valid], z_tgt


class ContinuousObjective(_EmissionObjective):
    """Raw continuous-vector emission — INJECT via `TrainingArguments(emission=ContinuousObjective)`. Emit via
    out_proj, cosine to the target, STOP as a BCE on stop_proj. No FSQ digits, so `dim_lg`/`lab_label` are None
    and the FSQ-only aux terms (label subspace) self-skip. The emitted latent can settle at the CENTROID of several
    admissible futures (calibrated superposition) — the mixture a discrete FSQ argmax cannot represent. REQUIRES the
    model built with `continuous_emit=True` (the head architecture is a MODEL property, chosen at from_pretrained)."""
    codebook = False

    def __init__(self, model: LangSetModel, args: TrainingArguments, dev: torch.device, trainer: Trainer) -> None:
        super().__init__(model, args, dev, trainer)
        # fail fast: the continuous forward (rollout_train_continuous) needs the continuous head; without it the user
        # would hit a low-signal AssertionError deep in the rollout. Point them at the from_pretrained flag instead.
        if not (getattr(model.head, "multi_latent", False) and getattr(model.head, "continuous_emit", False)):
            raise ValueError(
                "emission=ContinuousObjective requires a model built with multi_latent=True and continuous_emit=True; "
                "pass LangSetModel.from_pretrained(..., multi_latent=True, continuous_emit=True).")

    def emit(self, se: dict[str, torch.Tensor], target_lat: torch.Tensor, valid: torch.Tensor,
             lens_l: list[int], bidx: list[int], b: int, lmax: int, ep: int) -> EmissionOut:
        m, dev = self.m, self.dev
        preds, stop_lg = m.rollout_train_continuous(
            se["input_ids"], se["attention_mask"], target_lat)       # [b, lmax+1, d], [b, lmax+1]
        recon = preds[:, :lmax]                                      # emitted continuous latents (gradient flows here)
        stop_lab = torch.zeros(b, lmax + 1, device=dev)
        stop_msk = torch.zeros(b, lmax + 1, dtype=torch.bool, device=dev)
        for r, nl in enumerate(lens_l):
            stop_msk[r, :nl + 1] = True                              # supervise positions 0..nl (STOP at nl); rest ignore
            stop_lab[r, nl] = 1.0
        loss_stop = F.binary_cross_entropy_with_logits(stop_lg[stop_msk], stop_lab[stop_msk])
        recon_loss = (1.0 - F.cosine_similarity(recon[valid], target_lat[valid], dim=-1)).mean()
        loss_dims = recon_loss.new_zeros(())                        # no FSQ digit CE in continuous mode (logged as 0)
        return EmissionOut(recon=recon, base_loss=loss_stop + recon_loss,
                           logs={"loss_stop": loss_stop, "loss_dims": loss_dims, "recon_loss": recon_loss},
                           dim_lg=None, lab_label=None)


# ---- target source ------------------------------------------------------------------------------
class _TargetSource:
    """Strategy for the TARGET latents the emission trains toward, plus any anti-collapse regularization.
    Default = stop-grad EMA twin. Selected ONCE per run. All sources share the __init__ signature
    (model, args, tok, dev) so they are interchangeable as an injected `TrainingArguments.target_source`."""

    suppresses_nce: bool = False        # if True the trainer skips the in-batch NCE term (a live-target source that
    #                                     already prevents collapse via `regularizer` doesn't need — and fights — it)
    wants_regularizer: bool = False     # if True the trainer computes objective.z_for_reg and adds `regularizer` to the
    #                                     loss; keeps that (non-trivial) work off the default path when False
    twin: Optional[LangSetModel] = None  # the model the EVAL block encodes its retrieval bank with (the EMA copy for the
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

    def __init__(self, model: LangSetModel, args: TrainingArguments, tok: Any, dev: torch.device) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = copy.deepcopy(model)
        for p in self.twin.parameters():
            p.requires_grad_(False)
        self.twin.eval()
        self._online = [po for po in model.parameters() if po.requires_grad]
        self._ema = [pe for pe, po in zip(self.twin.parameters(), model.parameters()) if po.requires_grad]

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Single-latent emission of each text -> [N, d] normalized, no_grad. Truncated to target_max_len
        # (default 64: targets are short descriptors; raise it when a target is a DOCUMENT, e.g. emit_seed's
        # phase-0 target is a full abstract). Short future strings are already < 64 so unaffected.
        a, tok, dev = self.a, self.tok, self.dev
        e = tok(texts, padding=True, truncation=True, max_length=a.target_max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            z = self.twin(e["input_ids"], e["attention_mask"])
        return F.normalize(z.float(), dim=-1)

    def update(self) -> None:
        with torch.no_grad():
            torch._foreach_mul_(self._ema, self.a.ema_m)
            torch._foreach_add_(self._ema, self._online, alpha=1.0 - self.a.ema_m)


class SIGRegTarget(_TargetSource):
    """EMA-free anti-collapse (LeJEPA, arXiv:2511.08544). INJECT via `TrainingArguments(target_source=SIGRegTarget)`.
    Targets come from the LIVE model WITH gradient (no stop-grad twin); collapse is prevented by an isotropic-Gaussian
    SIGReg penalty on the pre-quant z (via `regularizer`) instead of by a twin. So it drops the twin's VRAM + target
    forward, the in-batch NCE is suppressed (the regularizer replaces it), and eval encodes with the live model itself.
    Reads scalar knobs off args: sigreg_lambda (loss weight, applied in the trainer), sigreg_knots, sigreg_slices."""
    suppresses_nce = True
    wants_regularizer = True

    def __init__(self, model: LangSetModel, args: TrainingArguments, tok: Any, dev: torch.device) -> None:
        self.m, self.a, self.tok, self.dev = model, args, tok, dev
        self.twin = model                                        # no separate twin — eval encodes with the live model
        self.sig_reg = SIGReg(knots=args.sigreg_knots, slices=args.sigreg_slices).to(dev)

    def encode(self, texts: list[str]) -> torch.Tensor:
        # LIVE target WITH gradient (no no_grad, no twin): both the emitted and target latents move, and SIGReg —
        # not a stop-grad twin — is what stops them collapsing together.
        a, tok, dev = self.a, self.tok, self.dev
        e = tok(texts, padding=True, truncation=True, max_length=a.target_max_len, return_tensors="pt").to(dev)
        z = self.m(e["input_ids"], e["attention_mask"])
        return F.normalize(z.float(), dim=-1)

    def regularizer(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> Optional[torch.Tensor]:
        # Two INDEPENDENT Gaussianity penalties (predicted E[digit] and target z), NOT a match between them —
        # each is pushed toward isotropic Gaussian, spreading codes across the FSQ grid.
        return self.sig_reg(z_pred) + self.sig_reg(z_tgt)


# ---- small function-strategies ------------------------------------------------------------------
def multi_epoch_order(tr_idx: list[int], rng_t: torch.Generator, args: TrainingArguments,
                      seeds: list[str]) -> list[int]:
    """DEFAULT epoch ordering: a plain shuffle of the training positions. Inject a different `epoch_order` to
    change it (e.g. a variant that keeps a seed's branches contiguous)."""
    return torch.randperm(len(tr_idx), generator=rng_t).tolist()


def grouped_epoch_order(tr_idx: list[int], rng_t: torch.Generator, args: TrainingArguments,
                        seeds: list[str]) -> list[int]:
    """SUPERPOSITION epoch ordering — INJECT via `TrainingArguments(epoch_order=grouped_epoch_order)`. Keeps a
    seed's branches CONTIGUOUS (shuffle the seed GROUPS, not the positions) so all of one seed's futures land in
    the same batch -> their per-target digit-CE sums to a soft-CE toward the branch mixture P_mix, a clean per-seed
    uncertainty signal instead of branches scattered across batches. Pair with build_superposition_loss_terms."""
    grp: dict[str, list[int]] = {}
    for pos in range(len(tr_idx)):
        grp.setdefault(seeds[tr_idx[pos]], []).append(pos)
    gkeys = list(grp.keys())
    gperm = torch.randperm(len(gkeys), generator=rng_t).tolist()
    return [pos for gi in gperm for pos in grp[gkeys[gi]]]


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
