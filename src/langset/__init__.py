"""langset — a short path to a world model in your LLM.

Few-shot fine-tune a pretrained LLM to predict in latent space: emit a sequence of latents (a JEPA world model)
that holds a calibrated superposition of next states — or a single latent as a bespoke embedding model, which is
Sentence-Transformer-shaped and drops into SetFit as a `model_body`.
"""
from langset.modeling import EmitHead, LangSetModel
from langset.probes import calibration_corr, linear_decodability
from langset.trainer import Trainer
from langset.training_args import TrainingArguments

__all__ = ["LangSetModel", "EmitHead", "Trainer", "TrainingArguments",
           "calibration_corr", "linear_decodability"]
__version__ = "0.11.0"
