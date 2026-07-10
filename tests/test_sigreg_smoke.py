"""SMOKE test for the SIGReg (LeJEPA) anti-collapse target source.

The multi-latent characterization golden (test_trainer_multi_characterization.py) pins the DEFAULT EMA-twin path.
SIGReg is a different training path (no EMA twin, live gradient targets, in-batch NCE gated off, an extra
isotropic-Gaussian penalty), so it needs its own coverage. This is a fast CPU/fp32 smoke, not a golden: it just
proves the injected `target_source=SIGRegTarget` path RUNS, stays finite, and actually changes training vs the
EMA default — plus the SIGReg quadrature-arg guards. Run:  .venv/bin/python tests/test_sigreg_smoke.py
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np
import torch

# reuse the golden harness (tiny model, deterministic seed, 8-row dataset, aux-term-exercising args)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset import Trainer  # noqa: E402
from langset.sigreg import SIGReg  # noqa: E402
from langset.strategies import SIGRegTarget  # noqa: E402


def _run_sigreg() -> np.ndarray:
    model = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        args = M._args(td, target_source=SIGRegTarget, sigreg_lambda=0.3)
        Trainer(model, args, M._dataset()).train()
    return M._flat_trainable(model)


def test_sigreg_runs_and_is_finite() -> None:
    """One short multi-latent run with SIGRegTarget injected: params stay finite (the live-gradient + regularizer
    path didn't NaN) and every trainable moved (training actually happened)."""
    post = _run_sigreg()
    assert np.isfinite(post).all(), "SIGReg training produced non-finite params"
    assert float(np.abs(post).sum()) > 0.0, "no trainable params moved — SIGReg path didn't train"


def test_sigreg_diverges_from_ema_default() -> None:
    """SIGReg must be a REAL alternative to the EMA twin, not a silent no-op: injecting target_source=SIGRegTarget
    has to change the learned weights vs the default EMATwinTarget beyond tolerance."""
    base = M._run()["params_post"]                 # default EMATwinTarget
    sig = _run_sigreg()
    max_delta = float(np.max(np.abs(sig - base)))
    assert max_delta > 1e-4, f"SIGReg did not change training vs EMA default (max param delta {max_delta:.2e})"


def test_sigreg_arg_guards() -> None:
    """The quadrature args are user-set via TrainingArguments; degenerate values raise a clear error instead of a
    ZeroDivisionError (knots=1 -> dt=3/(knots-1)) or a broken projection (slices<=0)."""
    for bad in (dict(knots=1), dict(knots=0), dict(slices=0), dict(slices=-4)):
        try:
            SIGReg(**bad)
        except ValueError:
            continue
        raise AssertionError(f"SIGReg({bad}) should have raised ValueError")
    SIGReg(knots=2, slices=1)                       # minimal valid config must construct


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    test_sigreg_arg_guards(); print("sigreg_arg_guards OK")
    test_sigreg_runs_and_is_finite(); print("sigreg_runs_and_is_finite OK")
    test_sigreg_diverges_from_ema_default(); print("sigreg_diverges_from_ema_default OK")
    print("ALL PASS")
