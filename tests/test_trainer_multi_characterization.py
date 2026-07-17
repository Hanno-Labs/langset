"""GOLDEN characterization tests for the MULTI-LATENT (FSQ latent-set) trainer.

Purpose: pin the EXACT numerics of `Trainer._train_multi` so the strategy-seams refactor (extracting the
emission objective / target source / aux loss terms into swappable strategies) is provably MECHANICALLY
IDENTICAL when wired with its DEFAULT strategies. The single-latent golden (test_trainer_characterization.py)
does NOT cover this path — all the feature branching lives here.

The golden exercises, in one deterministic run, every DEFAULT code path the refactor touches:
  * FSQ emission objective (per-dim digit CE + folded STOP + cosine recon)
  * EMA-twin target source (stop-grad targets + ema_update)
  * the aux LossTerms that are on by default or lit by these args: multi_nce (+ its identical-text
    negative mask), supcon, phase head, hard negatives
So an accidental change to ANY of those terms / their weighting / their order fails loudly.

Everything runs on CPU/fp32 with a tiny model so it's deterministic and fast.

Regenerate the golden ONLY when you intentionally change behavior:
    .venv/bin/python tests/test_trainer_multi_characterization.py --update
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from langset import LangSetModel, Trainer, TrainingArguments

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_trainer_multi.npz"
TINY_MODEL = os.environ.get(
    "LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"
)
RTOL, ATOL = 1e-4, 1e-6


def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_model() -> LangSetModel:
    _seed()  # seed BEFORE building: LoRA-A / FSQ head init are random -> must be pinned
    return LangSetModel.from_pretrained(
        TINY_MODEL, bf16=False, device="cpu", multi_latent=True, fsq_dim=32, fsq_levels=8
    )


def _dataset() -> list[dict]:
    """8 rows, each a seed -> a SET of target descriptions, with per-item stage labels (sup/phase) and a
    per-row hard negative — so one run lights multi_nce, supcon, phase, and hard_neg together."""
    topics = ["late payment", "theft", "property tax"]
    stages = ["filing", "hearing", "ruling"]
    rows = []
    for i in range(8):
        t = topics[i % 3]
        rows.append(
            {
                "input_text": f"case {i}: a dispute concerning {t}",
                "target_texts": [f"{s}: the {t} matter at the {s} stage" for s in stages],
                "stage": list(stages),  # per-item labels (sup_field / phase)
                "hardneg": [f"an unrelated {topics[(i + 1) % 3]} document"],
            }
        )
    return rows


def _args(out_dir: str, **over: object) -> TrainingArguments:
    d: dict[str, object] = dict(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        max_len=64,
        report_to=None,
        verbose=False,
        eval_every=99,
        patience=99,
        val_frac=0.25,
        seed=0,  # no eval/early-stop -> deterministic weights
        sup_field="stage",
        lam_sup=0.2,
        lam_phase=0.1,  # light: light up supcon + phase-head terms
        hard_neg_field="hardneg",
        lam_hard_neg=0.2,  # light up the hard-negative term
        label_dims={"stage": [1]},
        lam_label_dims=0.3,  # light up the FSQ label-subspace term (reserved dim 1)
        output_dir=out_dir,
    )
    d.update(over)
    return TrainingArguments(**d)  # type: ignore[arg-type]


def _flat_trainable(model: LangSetModel) -> np.ndarray:
    return np.concatenate(
        [
            p.detach().float().cpu().numpy().ravel()
            for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0])
            if p.requires_grad
        ]
    ).astype(np.float64)


def _run(**arg_over: object) -> dict[str, np.ndarray]:
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, **arg_over), _dataset()).train()
    return {"params_post": _flat_trainable(model)}


def generate_golden() -> None:
    g = _run()
    np.savez(GOLDEN, **g)
    print(f"[golden] wrote {GOLDEN}  (params={g['params_post'].size})")


def test_train_multi_identity() -> None:
    assert GOLDEN.exists(), "golden missing -> run with --update"
    g = np.load(GOLDEN)
    out = _run()
    np.testing.assert_allclose(
        out["params_post"],
        g["params_post"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="MULTI-LATENT train math changed (FSQ objective / EMA target / aux terms differ)",
    )


def test_golden_is_sensitive_to_nce() -> None:
    """Prove the golden actually GUARDS behavior: dropping the multi_nce term (a real math change) must
    diverge from the golden beyond tolerance, so the identity test would catch a silent term change."""
    assert GOLDEN.exists(), "golden missing -> run with --update"
    g = np.load(GOLDEN)
    out = _run(lam_multi_nce=0.0)
    max_delta = float(np.max(np.abs(out["params_post"] - g["params_post"])))
    assert max_delta > 1e-4, (
        f"golden is NOT sensitive to multi_nce (max param delta {max_delta:.2e}) -> tighten the snapshot"
    )


if __name__ == "__main__":
    if "--update" in sys.argv:
        generate_golden()
    else:
        test_train_multi_identity()
        print("train_multi_identity OK")
        test_golden_is_sensitive_to_nce()
        print("golden_is_sensitive OK")
        print("ALL PASS")
