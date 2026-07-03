"""Training configuration for langset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

    # validation / early-stop
    val_frac: float = 0.2
    eval_every: int = 1
    patience: int = 10
    seed: int = 0

    # io / logging
    output_dir: str = "langset-out"
    report_to: Optional[str] = None       # "wandb" or None
    wandb_project: str = "langset"
    verbose: bool = True
