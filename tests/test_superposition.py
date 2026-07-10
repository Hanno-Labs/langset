"""REPRODUCING test for the superposition selector bug surfaced in PR #7 review (Copilot comment 2).

`last_epoch_selector` returns float(ep) so it should keep the FINAL epoch's weights. But checkpoint selection runs
only inside the `if ep % a.eval_every: continue` gate, so when eval_every > 1 and the last epoch isn't an eval
epoch, the selector never sees it and an earlier epoch is restored. Since evaluation is RNG-neutral (verified: with
no restore, eval_every doesn't change the trajectory), the result must be INVARIANT to eval_every under
last_epoch_selector. It currently is not -> RED until the trainer always evaluates the final epoch for such a
selector. Run:  .venv/bin/python tests/test_superposition.py
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset import Trainer  # noqa: E402
from langset.strategies import last_epoch_selector  # noqa: E402


def _run(**over) -> np.ndarray:
    model = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, M._args(td, epochs=5, verbose=False, selector=last_epoch_selector, **over),
                M._dataset()).train()
    return M._flat_trainable(model)


def test_last_epoch_selector_invariant_to_eval_every() -> None:
    """last_epoch_selector keeps the final epoch, so its result cannot depend on the eval cadence. epochs=5 with
    eval_every=3 evaluates ep0,ep3 and would restore ep3 (not the true final ep4); eval_every=1 restores ep4."""
    every1 = _run(eval_every=1)                            # final epoch (ep4) is always an eval epoch -> correct
    every3 = _run(eval_every=3)                            # ep4 skipped -> restores ep3 unless the trainer is fixed
    max_delta = float(np.max(np.abs(every1 - every3)))
    assert max_delta == 0.0, (
        f"last_epoch_selector result depends on eval_every (max param delta {max_delta:.2e}) — the final epoch is "
        f"skipped by the eval gate, so an earlier epoch is kept instead of the last")


def test_superposition_triple_activates() -> None:
    """Injecting the three superposition strategies changes training vs the default (grouped order + same_seed mask
    + last-epoch selection are really wired in) and stays finite."""
    from langset.strategies import build_superposition_loss_terms, grouped_epoch_order
    base = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(base, M._args(td, verbose=False), M._dataset()).train()
    base_p = M._flat_trainable(base)
    sup = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(sup, M._args(td, verbose=False, loss_terms=build_superposition_loss_terms,
                             epoch_order=grouped_epoch_order, selector=last_epoch_selector), M._dataset()).train()
    sup_p = M._flat_trainable(sup)
    assert np.isfinite(sup_p).all(), "superposition training went non-finite"
    assert float(np.max(np.abs(sup_p - base_p))) > 1e-4, "superposition triple did not change training"


def test_snapshot_every_is_one_based() -> None:
    """Copilot #3: snapshot_every=N snapshots AFTER epochs N, 2N, ... (1-based) — not at ep0. epochs=4, N=2 -> only
    {output_dir}_ep2 and _ep4 exist; _ep0 must not."""
    import os
    model = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/run"
        Trainer(model, M._args(td, epochs=4, verbose=False, output_dir=out, snapshot_every=2), M._dataset()).train()
        assert not os.path.isdir(f"{out}_ep0"), "snapshot fired at ep0 (should be 1-based)"
        assert os.path.isdir(f"{out}_ep2") and os.path.isdir(f"{out}_ep4"), "expected snapshots after epochs 2 and 4"


if __name__ == "__main__":
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("test_last_epoch_selector_invariant_to_eval_every",
                 "test_superposition_triple_activates",
                 "test_snapshot_every_is_one_based"):
        try:
            globals()[name]()
            print(f"{name} PASS")
        except AssertionError as e:
            print(f"{name} FAIL -> {e}")
