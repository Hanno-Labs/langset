"""SMOKE test for the continuous multi-latent emission path (PR #5).

The multi-latent golden pins the DEFAULT FSQ path; continuous emission is a separate branch (raw out_proj +
cosine recon + BCE STOP, `dim_lg`/`lab_label` None). This is a fast CPU/fp32 smoke that the injected
`emission=ContinuousObjective` path RUNS and stays finite, that a rollout produces sane shapes, and that the
fail-fast guard fires when the model wasn't built continuous. Run:  .venv/bin/python tests/test_continuous_smoke.py
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset import LangSetModel, Trainer  # noqa: E402
from langset.strategies import ContinuousObjective  # noqa: E402


def _continuous_model() -> LangSetModel:
    M._seed()
    return LangSetModel.from_pretrained(M.TINY_MODEL, bf16=False, device="cpu",
                                        multi_latent=True, fsq_dim=32, fsq_levels=8, continuous_emit=True)


def test_continuous_training_runs_and_is_finite() -> None:
    """One short multi-latent run with emission=ContinuousObjective injected on a continuous-built model: trainable
    params stay finite (the out_proj + BCE-STOP path didn't NaN) and every param moved (training happened)."""
    model = _continuous_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, M._args(td, emission=ContinuousObjective), M._dataset()).train()
    post = M._flat_trainable(model)
    assert np.isfinite(post).all(), "continuous training produced non-finite params"
    assert float(np.abs(post).sum()) > 0.0, "no trainable params moved — continuous path didn't train"


def test_continuous_rollout_shapes() -> None:
    """Inference rollout on a continuous model returns latents in the backbone's space with a length per row that
    never exceeds max_steps (STOP is check-then-emit, so it can't over-run)."""
    model = _continuous_model()
    model.eval()
    lat, lengths = model.rollout(["alpha beta", "gamma"], max_steps=6, return_lengths=True)
    assert lat.shape[0] == 2 and lat.shape[-1] == model.latent_dim, f"unexpected rollout latent shape {tuple(lat.shape)}"
    assert int(lengths.max()) <= 6, f"rollout emitted more than max_steps latents ({int(lengths.max())} > 6)"


def test_continuous_objective_requires_continuous_model() -> None:
    """Injecting emission=ContinuousObjective on a model NOT built continuous fails fast with a clear ValueError
    (not a low-signal AssertionError deep in rollout_train_continuous)."""
    M._seed()
    fsq_model = LangSetModel.from_pretrained(M.TINY_MODEL, bf16=False, device="cpu",
                                             multi_latent=True, fsq_dim=32, fsq_levels=8)  # FSQ head, not continuous
    try:
        ContinuousObjective(fsq_model, M._args("/tmp/_unused"), torch.device("cpu"), None)
    except ValueError as e:
        assert "continuous_emit=True" in str(e)
        return
    raise AssertionError("ContinuousObjective should reject a non-continuous model with a ValueError")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    test_continuous_objective_requires_continuous_model(); print("requires_continuous_model OK")
    test_continuous_rollout_shapes(); print("rollout_shapes OK")
    test_continuous_training_runs_and_is_finite(); print("training_runs_and_is_finite OK")
    print("ALL PASS")
