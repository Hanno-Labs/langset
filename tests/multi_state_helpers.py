"""Shared deterministic fixtures for named-state multi-vector trainer tests."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from langset import LangSetModel, Trainer, TrainingArguments
from langset.strategies import CodeSoftmaxObjective

TINY_MODEL = os.environ.get(
    "LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"
)


def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_model() -> LangSetModel:
    _seed()
    model = LangSetModel.from_pretrained(
        TINY_MODEL,
        bf16=False,
        device="cpu",
        multi_latent=True,
        code_emit=True,
        n_codes=8,
    )
    model.head.set_code(torch.randn(8, model.latent_dim))
    return model


def _dataset() -> list[dict]:
    topics = ["late payment", "theft", "property tax"]
    stages = ["filing", "hearing", "ruling"]
    return [
        {
            "input_text": f"case {i}: a dispute concerning {topics[i % 3]}",
            "target_texts": [
                f"{stage}: the {topics[i % 3]} matter at the {stage} stage" for stage in stages
            ],
            "stage": list(stages),
        }
        for i in range(8)
    ]


def _args(out_dir: str, **over: object) -> TrainingArguments:
    values: dict[str, object] = {
        "epochs": 2,
        "batch_size": 4,
        "lr": 1e-3,
        "max_len": 64,
        "report_to": None,
        "verbose": False,
        "eval_every": 99,
        "patience": 99,
        "val_frac": 0.25,
        "seed": 0,
        "sup_field": "stage",
        "lam_sup": 0.2,
        "lam_phase": 0.1,
        "emission": CodeSoftmaxObjective,
        "output_dir": out_dir,
    }
    values.update(over)
    return TrainingArguments(**values)  # type: ignore[arg-type]


def _flat_trainable(model: LangSetModel) -> np.ndarray:
    return np.concatenate(
        [
            p.detach().float().cpu().numpy().ravel()
            for _, p in sorted(model.named_parameters(), key=lambda item: item[0])
            if p.requires_grad
        ]
    ).astype(np.float64)


def _run(**arg_over: object) -> dict[str, np.ndarray]:
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, **arg_over), _dataset()).train()
    return {"params_post": _flat_trainable(model)}
