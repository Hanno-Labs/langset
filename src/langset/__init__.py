"""langset — a short path to a world model in your LLM.

Few-shot fine-tune a pretrained LLM to predict in latent space: emit a sequence of latents (a JEPA world model)
that holds a calibrated superposition of next states — or a single latent as a bespoke embedding model, which is
Sentence-Transformer-shaped and drops into SetFit as a `model_body`.

## Module reference

- [Modeling](langset/modeling.html) — model and emission heads
- [Training](langset/trainer.html) — training loop and orchestration
- [Training arguments](langset/training_args.html) — run configuration
- [Losses](langset/loss.html) — composable training objectives
- [Strategies](langset/strategies.html) — emission and target strategies
- [Heads](langset/heads.html) — auxiliary supervised heads
- [Masking](langset/masking.html) — input masking helpers
- [Probes](langset/probes.html) — representation-quality probes
- [Bridge emission](langset/bridge_emission.html) — continuous vector-set emission
- [Data](langset/data.html) — dataset helpers
- [Selection](langset/selection.html) — model selection utilities
- [SIGReg](langset/sigreg.html) — isotropic regularization
- [Sentence-transformer integration](langset/st_module.html) — embedding-model adapter
"""

from langset.heads import Head
from langset.masking import (
    FieldMasker,
    SpanMasker,
    TokenMasker,
    build_masked,
    build_masked_pairs,
    mask_view,
    resolve_masker,
)
from langset.modeling import EmitHead, LangSetModel
from langset.probes import calibration_corr, linear_decodability
from langset.trainer import Trainer
from langset.training_args import TrainingArguments

__all__ = [
    "LangSetModel",
    "EmitHead",
    "Head",
    "Trainer",
    "TrainingArguments",
    "calibration_corr",
    "linear_decodability",
    "build_masked",
    "build_masked_pairs",
    "mask_view",
    "resolve_masker",
    "SpanMasker",
    "TokenMasker",
    "FieldMasker",
]
__version__ = "0.13.1"  # keep in sync with pyproject [project].version
