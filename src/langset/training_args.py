"""Training configuration for langset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from langset.strategies import (
    EMATwinTarget,
    FSQObjective,
    build_loss_terms,
    multi_epoch_order,
    multi_seed_texts,
    multi_select_metric,
)


@dataclass
class TrainingArguments:
    # optimization
    epochs: int = 40
    batch_size: int = 32
    lr: float = 5e-4
    tau: float = 0.07                     # contrastive temperature
    max_len: int = 512

    # loss weights. The self-contrastive term (emit(input) <-> emit(target_text)) is the primary at weight 1.0;
    # these are light aux terms: recon grounds the latent in the target text, uniform keeps the space spread.
    lam_recon: float = 0.3                # aux: the latent must also DECODE target_text
    lam_uniform: float = 0.1              # aux: light uniformity (spread latents on the sphere)
    lam_hard_neg: float = 0.0             # MULTI-LATENT hard-neg InfoNCE weight (0 = off, byte-identical to before)
    # MULTI-LATENT in-batch-negative InfoNCE — the separation term the multi-latent path was MISSING. The base
    # multi-latent loss (loss_stop + loss_dims + recon) is pure reconstruction: it MATCHES each emitted latent to
    # its EMA target but never pushes DIFFERENT items apart, so the geometry is capped by the base embedding and
    # organizes by whatever dominates the target text (boilerplate). This ports single-latent's primary objective:
    # each emitted recon vs the batch's targets, own target = positive, others = in-batch negatives (identical target
    # text masked as false-neg). ON by default — this is the fix, not an opt-in. Set 0.0 for the old behavior.
    # WEIGHT: unlike single-latent (where InfoNCE IS the whole loss, weight 1.0), here it COMPETES with the FSQ-recon
    # base (loss_stop+loss_dims+recon ~0.9); at 1.0 the InfoNCE (~ln(N_inbatch)≈4.85) overwhelms recon and SCRAMBLES
    # the geometry (retr_mrr stuck at chance, nce loss never descends). 0.3 balances the two → healthy geometry that
    # organizes by real structure (FDA smol: disease decodes held-out, area_fsq +0.58 / indication_fsq +0.82, no labels).
    lam_multi_nce: float = 0.3

    # SPEED (big models): stop_grad_target forwards the target view under no_grad (BYOL/MoCo-style) so it anchors
    # the geometry but takes NO backward -> drops one full backbone backward+recompute per step (~30%). Behavioral
    # (only the input view gets gradient); default False = byte-identical to before. SINGLE-latent path only; the
    # MULTI-latent path already sources targets from the no_grad EMA twin, so it is inherently stop-grad.
    stop_grad_target: bool = False

    # SPEED (INCOMPLETE - NOT numerically identical, do not use as-is): fuse input+target into ONE backbone call over
    # [input;target]. Intent = 1 launch + 1 grad-ckpt recompute instead of two. BUT the emit query is appended AFTER
    # the tokens, so its RoPE position depends on the PADDED length; padding the shorter view up to the common max
    # shifts its query -> emit differs (identity test: param delta ~2e-3 >> 1e-4 tol). A correct fuse needs per-sequence
    # position RESET (cu_seqlens / reset position_ids, like FlashAttention varlen packing) in the emit path = a model
    # change, not this trainer concat. Left off by default; flag kept only to document the dead end.
    fuse_views: bool = False

    # false-negative masking: when many rows share the same true geometry (e.g. two cases on the same legal
    # issue), in-batch contrastive would wrongly push them apart. Name a dataset column of group keys (a string
    # or a list of facet tokens per row); in-batch pairs that SHARE any key are masked out of the negatives
    # (the diagonal positive is always kept). None = vanilla in-batch (every off-diagonal is a negative).
    mask_field: Optional[str] = None

    # hard negatives: name a dataset column of hard-negative text(s) per row (MINED near-miss targets the emitted
    # latent should be pushed AWAY from — e.g. a boilerplate-similar case, or the WRONG lifecycle outcome).
    # SINGLE-latent: a scalar hard-neg text per row, appended as an EXTRA always-negative column to the in-batch
    # contrastive logits. MULTI-latent: a LIST of hard-neg texts per row; the batch's hard-neg latents form a shared
    # bank and each emitted item's reconstruction runs an InfoNCE (own target vs the bank) at weight `lam_hard_neg`.
    # Encoded under no_grad (memory-safe: no extra backward). None = no hard negs (byte-identical to before).
    hard_neg_field: Optional[str] = None

    # supervised-contrastive label shaping (Khosla et al.): name a dataset column of GROUP LABELS the emitted latents
    # should organize into SEPARATE REGIONS by. SINGLE-latent: one scalar label per row. MULTI-latent: a per-row LIST
    # of labels aligned 1:1 with `target_texts` (each emitted item's group, e.g. its lifecycle STAGE). Within a batch,
    # same-label emissions are pulled together and different-label pushed apart at weight `lam_sup` / temperature
    # `sup_tau`. Being proper SupCon (positives + negatives) it separates without collapsing. Labels "unknown"/""/"none"
    # are dropped from the term. None / lam_sup=0 = off (byte-identical to before). This is the region lever.
    sup_field: Optional[str] = None
    lam_sup: float = 0.0
    sup_tau: float = 0.1

    # PHASE HEAD — the non-collapsing alternative to SupCon. Reuses the same `sup_field` per-item labels but trains a
    # transient linear CE classifier (emitted recon -> phase) at weight `lam_phase`, instead of the contrastive pull.
    # CE only carves a separating hyperplane, so phase becomes linearly decodable WITHOUT collapsing within-phase
    # event identity (retr_mrr survives). lam_phase=0 = off. Use INSTEAD of lam_sup (set lam_sup=0).
    lam_phase: float = 0.0

    # FSQ LABEL SUBSPACE — a FORMAL label space inside the emitted code, with NO head per label. Map each facet to
    # reserved FSQ digit indices (each >=1; dim 0 is STOP-coupled). Those dims' reconstruction targets are REPLACED
    # by the label's codeword (one digit if n_classes<=fsq_levels, else little-endian base-fsq_levels across the
    # group), so the emitter is FORCED to encode the label AS coordinates of the token: reading a label = argmax the
    # reserved digit (no probe), writing = clamp it (controllable generation). Remaining dims reconstruct as usual.
    # Needs per-item label columns (lists aligned 1:1 with target_texts) named by the dict keys. Unknown/""/"none"
    # labels -> those dims ignored (-100) for that item. None = off (byte-identical). e.g. {"label_stage":[1],
    # "label_area":[2,3], "label_indication":[4,5,6]}.
    label_dims: Optional[dict[str, list[int]]] = None
    # weight on the reserved-dim label CE. It's a SEPARATE term (not folded into loss_dims) so the ~9 label dims are
    # not diluted among the ~1000 free reconstruction dims — at weight 1 it sits at parity with loss_dims/recon.
    lam_label_dims: float = 1.0

    # CONTINUOUS EMBEDDING SLOTS — the continuous analog of `label_dims` (which reserves FSQ *digit* indices). Reserve
    # a CONTIGUOUS slice of dims of the single emitted embedding for a named facet and bind it with a small transient
    # decoder head (CE for "classify", MSE for "regress") that reads ONLY those dims — so gradient forces the encoder
    # to route the facet INTO that slice. The facet then lives AS a readable sub-vector (emb[lo:hi] decodes to it)
    # instead of smeared across all d dims, and WITHOUT a persisted model: like the phase head the slot head is not
    # saved — its only job is to inject gradient (eval re-fits its own probe on the now-informative slice). The slot
    # dims still participate in the retrieval InfoNCE; they just ALSO carry a legible facet. Per-item labels come from
    # a dataset column named by the dict key; unknown/""/"none"/"nan" -> that item ignored for the slot. Works on BOTH
    # the single-latent path (slots the one emitted embedding) and the multi-latent path (MEAN-POOLS the emitted latent
    # SET over its valid positions -> one per-row vector, then slots that — same per-row semantics).
    # None = off (byte-identical). e.g. {"material_bucket": (0, 64, "classify"), "side": (64, 80, "classify")}.
    emb_slots: Optional[dict[str, tuple[int, int, str]]] = None
    lam_emb_slots: float = 1.0

    # MULTI-HOP training (scheduled sampling). ss_prob=0 = pure teacher forcing, gradients flow 1 hop (default,
    # byte-identical). ss_prob>0: for the first `train_hops` emitted positions (None = ALL), feed the model's OWN
    # predicted latent back with probability ss_prob instead of ground truth — the exposure-bias fix that makes
    # multi-hop rollout trained rather than emergent. ss_sample samples the self-fed digits instead of argmax.
    train_hops: Optional[int] = None
    ss_prob: float = 0.0
    ss_sample: bool = False
    # scheduled-sampling WARMUP: linearly ramp the effective ss_prob 0 -> ss_prob over the first `ss_warmup` epochs.
    # 0 = constant ss_prob (no warmup). REQUIRED for deep train_hops (feeding own predictions for many hops from
    # epoch 0, when they're garbage, destabilizes training — the phase head sticks at chance). Ramp lets the model
    # learn 1-hop first (teacher-forced), then phase in self-feeding as its predictions become worth feeding.
    ss_warmup: int = 0

    # EMIT SEED as step-0 (phase-0 as a generated node). Off (default) = byte-identical: the model only emits latents
    # for the FUTURE events (the first emission is the first future). On: each seed's OWN text is prepended as target
    # position 0, so the emitter learns to produce its start-state latent (head.encode(emit(seed))) BEFORE the futures.
    # This puts phase-0 into the emitted geometry as a real generated node (the rollout's step 0 becomes phase-0, the
    # futures shift to steps 1..L). sup_field labels get a leading "phase0" class (so phase-0 is phase-decodable too).
    # NOTE: downstream evals that assume "emission step 0 = first future" must offset by one when this is on.
    emit_seed: bool = False

    # knowledge-injection ([LEARN] rows): name a column tagging each row's task. Rows tagged "learn" are trained with
    # next-token CE (generate `target_text` given `input_text`, via the tied embedding — teaches the backbone domain
    # SUBSTANCE) instead of contrastive emit; all other rows stay the self-contrastive retrieval objective. Mixed in
    # one run, routed by tag = curriculum-as-multitask. `learn_ratio` = P(a learn step precedes each embed batch);
    # anneal high->low to "teach first". None / 0.0 = off (byte-identical: no learn rows, no learn steps).
    learn_field: Optional[str] = None
    learn_ratio: float = 0.0

    # multi-latent (variable-length FSQ latent-set emission). Active only when the model is `multi_latent=True`:
    # the Trainer reads a `target_texts` (list[str] per row) column, emits an EMA-twin target for each item, and
    # trains the token-native FSQ emitter (per-dim digits + a learned STOP). Ignored on the single-latent path.
    # `fsq_dim`/`fsq_levels` are MODEL internals (read from `model.head`), NOT args.
    ema_m: float = 0.99                   # EMA-twin momentum supplying the (stop-grad) target latents
    max_target_items: int = 12            # cap on target latents emitted/supervised per row
    max_steps: int = 16                   # free-rollout cap used in multi-latent eval
    # target-text encode length for the EMA target latents. 64 (default) suits SHORT targets (labels/tags/one-liners
    # — the library's premise). Raise it when a target is a DOCUMENT: emit_seed's phase-0 target is a full science
    # abstract, and 64 tokens keeps only its (often boilerplate) intro -> blurry phase-0 identity. Short targets
    # (e.g. future-event strings) are already < 64 so a higher cap only enriches the long ones (mild padding cost).
    target_max_len: int = 64
    # CachedTarget (two-stage, frozen-encoder) — path to a saved LangSetModel that supplies a FIXED target geometry
    # the emitter learns to roll forward in. Set together with `target_source=CachedTarget`: the targets and the
    # hard-negative bank are encoded ONCE and cached, so epochs stop re-encoding fixed data. None (default) = the
    # co-training EMA twin path (unchanged). Inert unless target_source is CachedTarget.
    target_encoder_ckpt: Optional[str] = None

    # SIGReg (LeJEPA, arXiv:2511.08544) scalar knobs — used only when the target source is SIGRegTarget (inject via
    # `TrainingArguments(target_source=SIGRegTarget)`, see the injection block below). SIGRegTarget replaces the EMA
    # twin with a LIVE encoder + an isotropic-Gaussian penalty on the pre-quant z; these tune that penalty. Inert
    # under the default EMATwinTarget. See langset/sigreg.py.
    sigreg_lambda: float = 0.3            # weight on the SIGReg isotropic-Gaussian loss (0.3 balances vs recon; higher
    #                                       over-diversifies and washes out global structure, lower under-constrains)
    sigreg_knots: int = 17               # Epps-Pulley quadrature knots (>= 2)
    sigreg_slices: int = 256             # random 1-D projection directions, resampled each step (>= 1)

    # Exp-B — CoT-conditioned emission. Activated by INJECTING the pair `loss_terms=build_cot_loss_terms` +
    # `seed_builder=cot_seed_texts` (see the injection block below) with a `cot_text` dataset column: the model is
    # co-trained to GENERATE each row's chain-of-thought (its own isolated backward, so the two autograd graphs never
    # coexist) and the latent forward is conditioned on seed+CoT. No injection / absent cot_text column = OFF
    # (byte-identical FSQ path). Not a flag — inject the strategies; this scalar just weights the CoT next-token CE.
    lam_cot: float = 1.0                  # weight on the CoT next-token CE (used only with the CoT strategies)

    # SUPERPOSITION (multi-latent) — to make one emitted latent hold a calibrated MIXTURE of several possible next
    # states, supervise it DIRECTLY: describe the whole SET of futures in a target text (see examples/maze-superposition)
    # and inject `selector=last_epoch_selector`, because retr_mrr is meant to FALL as the latent spreads over the set
    # (so it must not gate checkpoint selection). Read it back with `rollout(..., return_soft=True)`, whose entropy
    # tracks how many futures are live.
    # `snapshot_every` below is the one plain scalar: an ONLINE-weights snapshot to `{output_dir}_ep{N}`, `_ep{2N}`,
    # ... after every N epochs (1-based, independent of the eval cadence; separate from the best-so-far restore and
    # the preempt-resume checkpoint) so you can eval the trajectory offline. 0 = off (no snapshots) = byte-identical.
    snapshot_every: int = 0

    # ---- MULTI-LATENT STRATEGY INJECTION (dependency injection, not flags) --------------------------------------
    # The multi-latent step is assembled from swappable pieces (see langset/strategies.py). Each field below holds
    # the STRATEGY ITSELF — a class or callable — and the trainer just calls it; there is NO `if use_x:` selection.
    # The defaults reproduce the historical behavior byte-for-byte (guarded by test_trainer_multi_characterization).
    # To change a behavior, INJECT a different implementation, e.g.
    #     TrainingArguments(target_source=SIGRegTarget)          # EMA-free anti-collapse
    # rather than toggling a boolean the trainer then branches on. See strategies.py for the interface each must meet.
    emission: Callable = FSQObjective              # (model, args, dev, trainer) -> _EmissionObjective : seed->latents + base loss
    target_source: Callable = EMATwinTarget        # (model, args, tok, dev) -> _TargetSource : the target latents + anti-collapse
    loss_terms: Callable = build_loss_terms        # (args) -> list[_LossTerm] : the weighted aux separation/shaping terms
    epoch_order: Callable = multi_epoch_order      # (tr_idx, rng, args, seeds) -> list[int] : per-epoch visiting order
    selector: Callable = multi_select_metric       # (mode, mrr, purity, ep) -> float : the checkpoint-selection signal
    seed_builder: Callable = multi_seed_texts      # (trainer, seeds, args) -> list[str] : the texts the emission reads

    # JEPA MASKED-SELF-PREDICTION mode (single-latent). You give a `text` column of RAW, UNMASKED text; the
    # Trainer masks it FRESH EVERY EPOCH at runtime (input_text=visible, target_text=full) — no external label,
    # no pre-built masked training data, and unlimited mask diversity from a small corpus. It auto-activates when
    # the dataset has a `text` column and no `input_text` (zero-config). `masker` selects the algorithm:
    #   None / "word"  -> TokenMasker(mask_ratio)  : hide scattered words (the reasonable default)
    #   "span"         -> SpanMasker(mask_ratio)   : hide one contiguous span
    #   "field"        -> FieldMasker(ratio=mask_ratio) : hide whole delimited fields
    #   a Callable (text, rng) -> (visible, hidden) : your own masker (e.g. TokenMasker with a protect predicate)
    # Set masker explicitly to force masking even when input_text/target_text are present.
    masker: Optional[Callable[..., Any] | str] = None
    mask_ratio: float = 0.15

    # PREEMPT-RESUME (long big-model runs on preemptible GPUs). Epochs are the natural checkpoint boundary, so instead of
    # fragile mid-epoch state we make epochs SMALL: `max_steps_per_epoch` caps each epoch to N steps (~<=30min of wall
    # clock); run proportionally MORE epochs to cover the same data (optimization is identical — the optimizer only sees
    # batches, "epoch" is just when we stop to eval+save). Each mini-epoch draws a fresh random subset (standard minibatch
    # SGD). When resume_dir is set (a PERSISTENT path, e.g. a Modal Volume) the trainer writes a FULL training-state
    # checkpoint (trainable weights + optimizer + epoch + best-so-far + rng) to {resume_dir}/resume.pt at every epoch
    # boundary and LOADS it at train() start -> a preempt/retry continues from the last mini-epoch instead of restarting
    # from ep0. Atomic (tmp+rename) so a mid-write preempt can't corrupt it. Works on BOTH the single- and multi-latent
    # paths. None/0 = OFF (byte-identical).
    resume_dir: Optional[str] = None
    max_steps_per_epoch: int = 0          # >0: cap each epoch to N steps (bound the max work lost on preempt to ~N*step_s)
    run_sig: Optional[str] = None         # identity fingerprint (base+data+config); resume REFUSES on mismatch -> fresh,
                                          # so a DIFFERENT model/run can never wrongly load THIS run's checkpoint

    # validation / early-stop
    val_frac: float = 0.2
    eval_every: int = 1
    # In-loop eval cost scales with the val-set size THREE ways: the EMA bank re-encode, the free-rollout over every val
    # seed, and a Python loop doing a GPU-synced argsort PER emission. On big/long-text val sets that makes eval minutes,
    # not seconds. Cap the in-loop eval to a fixed cohort so it stays a fast training-time signal (~<=10s). 0 = no cap
    # (use the full val set). The final/best model is still selected on this proxy; run a full eval offline for the number.
    eval_max_chains: int = 64
    patience: int = 10
    seed: int = 0
    # checkpoint-selection signal (multi-latent). "retr_mrr" = event-identity retrieval (default, unchanged).
    # "purity" = kNN-purity of the emitted geometry by the SupCon `sup_field` label (selects the most STAGE-SEPARATED
    # epoch — required when lam_sup>0, since SupCon suppresses retr_mrr by pulling same-label items together).
    # "blend" = retr_mrr + purity (keep event identity AND gain stage-geometry). purity/blend need sup labels.
    select: str = "retr_mrr"

    # io / logging
    output_dir: str = "langset-out"
    report_to: Optional[str] = None       # "wandb" or None
    wandb_project: str = "langset"
    verbose: bool = True
