"""langset — few-shot fine-tune an LLM to emit a latent into a bespoke geometry you define.

The trained model is Sentence-Transformer-shaped, so it drops into SetFit as a `model_body`.
"""
from langset.modeling import EmitHead, LangSetModel
from langset.trainer import Trainer
from langset.training_args import TrainingArguments

__all__ = ["LangSetModel", "EmitHead", "Trainer", "TrainingArguments"]
__version__ = "0.9.1"
