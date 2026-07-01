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
