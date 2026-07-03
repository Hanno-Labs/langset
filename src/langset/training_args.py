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

    # false-negative masking: when many rows share the same true geometry (e.g. two cases on the same legal
    # issue), in-batch contrastive would wrongly push them apart. Name a dataset column of group keys (a string
    # or a list of facet tokens per row); in-batch pairs that SHARE any key are masked out of the negatives
    # (the diagonal positive is always kept). None = vanilla in-batch (every off-diagonal is a negative).
    mask_field: Optional[str] = None

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
