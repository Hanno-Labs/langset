"""Wrap a trained LangSetModel as a `sentence_transformers.SentenceTransformer` so it drops into SetFit as the
`model_body`:

    body = langset_model.as_sentence_transformer()
    setfit_model = SetFitModel(model_body=body, labels=[...])
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LangSetSTModule(nn.Module):
    """A Sentence-Transformers custom module: tokenizes text and emits the langset latent as `sentence_embedding`."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model                                  # registered submodule -> .to(device) propagates
        # attributes SentenceTransformer / SetFit's trainer expect a body to expose
        self.tokenizer = model.tokenizer
        self.auto_model = model.backbone
        self.max_seq_length = model.max_len

    def forward(self, features: dict[str, Any]) -> dict[str, Any]:
        z = self.model(features["input_ids"], features["attention_mask"])
        features["sentence_embedding"] = z
        return features

    def tokenize(self, texts: list[str], **kw: Any) -> dict[str, torch.Tensor]:
        enc = self.model.tokenizer(list(texts), padding=True, truncation=True,
                                   max_length=self.model.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.model.latent_dim)

    def get_embedding_dimension(self) -> int:        # ST 4.x name
        return int(self.model.latent_dim)

    def get_config_dict(self) -> dict[str, Any]:
        return {}

    def save(self, output_path: str, **kw: Any) -> None:
        self.model.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str, **kw: Any) -> "LangSetSTModule":
        from langset.modeling import LangSetModel
        return LangSetSTModule(LangSetModel.load(input_path))


def to_sentence_transformer(model: Any) -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    st = SentenceTransformer(modules=[LangSetSTModule(model)], device=str(model.device))
    for attr, val in (("tokenizer", model.tokenizer), ("max_seq_length", model.max_len)):
        try:                                                 # some ST versions expect these on the ST object
            setattr(st, attr, val)
        except Exception:                                    # noqa: BLE001 - it's a read-only property -> already resolves
            pass
    return st
