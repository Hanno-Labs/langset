"""Repro for the LANGSET_PROFILE_STEPS crash on a CPU box (Copilot review, PR #9).

The step-level profiler is env-gated (`LANGSET_PROFILE_STEPS=N` → profile N single-latent steps, dump the
table, and sys.exit(0)). But the harness configures the profiler with a CUDA activity and calls
`torch.cuda.synchronize()` UNCONDITIONALLY whenever profiling is on — so enabling it on a machine without CUDA
raises a CUDA error instead of profiling. Intended behavior: profile on whatever device is present, then exit(0).
"""
from __future__ import annotations

import tempfile

import pytest

from test_trainer_characterization import _args, _build_model, _dataset

from langset import Trainer


def test_profiling_on_cpu_profiles_then_exits(monkeypatch):
    # profile exactly 1 step; the intended terminal behavior is a clean sys.exit(0) after the dump
    monkeypatch.setenv("LANGSET_PROFILE_STEPS", "1")
    model = _build_model()                          # CPU tiny model
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(SystemExit) as ei:       # profile-then-exit, NOT a CUDA RuntimeError
            Trainer(model, _args(td), _dataset()).train()
    assert ei.value.code == 0
