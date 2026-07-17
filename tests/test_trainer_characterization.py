"""GOLDEN characterization tests for the langset contrastive trainer.

Purpose: pin the EXACT numerics of the single-latent self-contrastive path so a "faster but identical"
refactor for big models (e.g. avoid the grad-ckpt recompute, cache the target-view forward, fuse ops)
is provably MECHANICALLY IDENTICAL. Any accidental change to the loss / gradients / emitted embedding
fails loudly instead of silently degrading every future training run.

Two tiers:
  * test_forward_identity / test_train_identity  -> tight golden (rtol 1e-4). Numerics-preserving
    perf refactors MUST pass these. This is the safety net.
  * test_smoke_retrieval                         -> behavioral floor. For math-CHANGING optimizations
    (drop recon, EMA/stop-grad target) identity is NOT expected; this only asserts training still works.
  * test_golden_is_sensitive_to_recon            -> proves the golden actually GUARDS behavior: running
    with lam_recon=0 (the optimization we're considering) diverges from the golden beyond tolerance.

Everything runs on CPU/fp32 with a tiny model so it's deterministic and fast (seconds).

Regenerate the golden ONLY when you intentionally change behavior:
    PYTHONPATH=src_proto python tests/test_trainer_characterization.py --update
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
GOLDEN = HERE / "golden_trainer.npz"
# a tiny published checkpoint: fixed weights -> deterministic. Override to characterize a different arch.
TINY_MODEL = os.environ.get(
    "LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"
)
LATENT_DIM = 32
PROBE = [
    "a contract dispute about late payment terms",
    "a criminal appeal concerning theft",
    "a tax case over property transfer",
]
RTOL, ATOL = (
    1e-4,
    1e-5,
)  # tight enough to catch any real math change (those move by >1e-3), loose enough for
#    cross-platform float reassociation (macOS-generated golden vs Linux CI drifts ~1e-6/element)


def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_model(grad_ckpt: bool = False) -> LangSetModel:
    _seed()  # seed BEFORE building: LoRA-A / EmitHead init are random -> must be pinned
    return LangSetModel.from_pretrained(
        TINY_MODEL, latent_dim=LATENT_DIM, bf16=False, device="cpu", grad_ckpt=grad_ckpt
    )


def _dataset() -> list[dict[str, str]]:
    topics = ["late payment", "theft", "property tax"]
    return [
        {
            "input_text": f"case {i}: a dispute concerning {topics[i % 3]}",
            "target_text": f"holding: the {topics[i % 3]} provisions govern this matter",
        }
        for i in range(8)
    ]


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


def _embed(model: LangSetModel, texts: list[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float64)


def _run(grad_ckpt: bool = False, **arg_over: object) -> dict[str, np.ndarray]:
    """Deterministic end-to-end: build -> emit BEFORE training (pins forward) -> train -> snapshot
    trainable weights (pins the full 3-forward loss+backward+optimizer math) + emit AFTER training."""
    model = _build_model(grad_ckpt=grad_ckpt)
    emb_pre = _embed(model, PROBE)
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, **arg_over), _dataset()).train()
    return {
        "emb_pre": emb_pre,
        "params_post": _flat_trainable(model),
        "emb_post": _embed(model, PROBE),
    }


# ---- golden generation --------------------------------------------------------------------------
def generate_golden() -> None:
    g = _run()
    np.savez(GOLDEN, **g)
    print(f"[golden] wrote {GOLDEN}  (params={g['params_post'].size}, emb={g['emb_pre'].shape})")


# ---- Tier 1: identity (the safety net) ----------------------------------------------------------
def test_forward_identity() -> None:
    assert GOLDEN.exists(), (
        "golden missing -> run: python tests/test_trainer_characterization.py --update"
    )
    g = np.load(GOLDEN)
    model = _build_model()
    np.testing.assert_allclose(
        _embed(model, PROBE),
        g["emb_pre"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="FORWARD / emit path changed (encode() numerics differ)",
    )


def test_train_identity() -> None:
    assert GOLDEN.exists(), "golden missing -> run with --update"
    g = np.load(GOLDEN)
    out = _run()
    np.testing.assert_allclose(
        out["params_post"],
        g["params_post"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="TRAIN math changed (3-forward loss / backward / optimizer differ)",
    )
    np.testing.assert_allclose(
        out["emb_post"],
        g["emb_post"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="post-train embeddings changed",
    )


def test_grad_ckpt_is_identity() -> None:
    """Gradient checkpointing (mandatory for big-model batches) recomputes the forward during backward to
    save memory — and it must be NUMERICALLY EXACT. Proving grad_ckpt=True reproduces the grad_ckpt=False
    golden underpins the whole optimization: the recompute is pure overhead, so a refactor that avoids/
    reworks it is provably safe as long as it stays on this golden."""
    assert GOLDEN.exists(), "golden missing -> run with --update"
    g = np.load(GOLDEN)
    out = _run(grad_ckpt=True)
    np.testing.assert_allclose(
        out["params_post"],
        g["params_post"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="grad_ckpt changed the numerics (should be exact) -> recompute is not free-to-optimize",
    )


# ---- Tier 2: behavioral floor (for math-CHANGING opts) ------------------------------------------
def _contrastive_margin(model: LangSetModel) -> float:
    """Mean matched-pair similarity MINUS mean mismatched-pair similarity. An untrained/collapsed model has
    every embedding ~identical -> margin ~0; a WORKING contrastive trainer widens this gap (pull matched
    together, push mismatched apart). Model-agnostic, computed outside the trainer, robust to the tiny
    backbone's anisotropy (absolute similarities are all ~1.0, only the GAP carries the signal)."""
    ds = _dataset()
    I = _embed(model, [r["input_text"] for r in ds])
    T = _embed(model, [r["target_text"] for r in ds])
    sim = I @ T.T
    n = len(ds)
    diag = float(np.mean(np.diag(sim)))
    off = float((sim.sum() - np.trace(sim)) / (n * n - n))
    return diag - off


def test_smoke_training_works() -> None:
    """Behavioral floor for math-CHANGING opts (drop recon, EMA target) where bit-identity is NOT expected.
    A tiny-random backbone can't learn retrieval outright, but a WORKING contrastive trainer must widen the
    matched-vs-mismatched margin. A broken step (NaN, sign flip, dropped primary loss) leaves it ~0 or
    negative -> caught here without depending on exact numerics."""
    model = _build_model()
    pre = _contrastive_margin(model)
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, epochs=12, lr=2e-3), _dataset()).train()
    post = _contrastive_margin(model)
    print(f"[smoke] contrastive margin pre={pre:.4f} post={post:.4f}")
    assert np.isfinite(post), f"non-finite embeddings after training (post={post})"
    assert post > pre + 0.008, (
        f"contrastive training did not widen the margin: pre={pre:.4f} post={post:.4f}"
    )


# ---- proof the golden actually guards behavior --------------------------------------------------
def test_golden_is_sensitive_to_recon() -> None:
    """The optimization we're considering = drop the recon aux (lam_recon=0) to kill one of the 3 forwards.
    That CHANGES the math, so the golden MUST diverge from it -> proves the identity test would catch a
    silent removal of recon (or any other loss-term change)."""
    assert GOLDEN.exists(), "golden missing -> run with --update"
    g = np.load(GOLDEN)
    out = _run(lam_recon=0.0)
    max_delta = float(np.max(np.abs(out["params_post"] - g["params_post"])))
    assert max_delta > 1e-3, (
        f"golden is NOT sensitive to recon (max param delta {max_delta:.2e}) -> the identity test would "
        "miss a silent behavior change; tighten the snapshot"
    )


if __name__ == "__main__":
    if "--update" in sys.argv:
        generate_golden()
    else:
        test_forward_identity()
        print("forward_identity OK")
        test_train_identity()
        print("train_identity OK")
        test_grad_ckpt_is_identity()
        print("grad_ckpt_is_identity OK")
        test_smoke_training_works()
        print("smoke_training_works OK")
        test_golden_is_sensitive_to_recon()
        print("golden_is_sensitive OK")
        print("ALL PASS")
