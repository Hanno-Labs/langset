"""Training configuration for langset. Defaults encode the lessons from the validated recipe:
EMA-specialize (ema_m + low anchor), select on geometry not loss, restore-best."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainingArguments:
    # optimization
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    tau: float = 0.07                     # InfoNCE temperature
    max_len: int = 512

    # bootstrap -> specialize
    ema: bool = True                      # EMA self-distillation target (the "specialize" step)
    ema_m: float = 0.99                   # EMA momentum (lower = faster drift off the bootstrap)
    lam_anchor: float = 0.1               # cosine pull toward the FROZEN bootstrap target (0 = fully specialize)

    # validation / early-stop  (see selection.py)
    val_frac: float = 0.2
    eval_every: int = 1
    patience: int = 6                     # epochs without val improvement before stopping
    select: Literal["auto", "retrieval", "purity", "loss"] = "auto"  # auto = purity if labels else retrieval
    seed: int = 0

    # io / logging
    output_dir: str = "langset-out"
    report_to: Optional[str] = None       # "wandb" or None
    wandb_project: str = "langset"
    verbose: bool = True
