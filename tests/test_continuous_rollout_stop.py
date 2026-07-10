"""REPRODUCING test for the continuous multi-latent rollout STOP off-by-one (PR #5 review, Copilot comment 1).

Training (`rollout_train_continuous`) and the FSQ inference branch use CHECK-THEN-EMIT STOP semantics: the STOP
head decides at a position whether that position is a real emission or the terminal, and the terminal emits
NOTHING (FSQ: `emit_now = alive & (dim0 != stop_idx)`; continuous training: STOP is supervised at the dedicated
terminal position L, whose prediction is not a reconstruction target).

The continuous branch of `rollout()` instead does EMIT-THEN-CHECK: it emits `z`, counts it in `lengths`, feeds it
back, and only THEN consults `stop_proj`. So the step at which the model wants to stop still produces a spurious
latent and over-counts the length by one -> a train/inference mismatch.

This test forces the (trained) STOP head to fire at the very first decision point, so the correct emitted length
is 0 (nothing before the first STOP). It currently returns 1 -> RED until the rollout is fixed to check-then-emit.
Run:  .venv/bin/python tests/test_continuous_rollout_stop.py
"""
from __future__ import annotations

import numpy as np
import torch

from langset import LangSetModel

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _continuous_model() -> LangSetModel:
    torch.manual_seed(0)
    np.random.seed(0)
    return LangSetModel.from_pretrained(TINY, bf16=False, device="cpu",
                                        multi_latent=True, fsq_dim=32, fsq_levels=8, continuous_emit=True)


def test_continuous_rollout_stops_before_emitting() -> None:
    """With the STOP head forced to fire at step 0, no latent should be emitted (check-then-emit, matching the FSQ
    branch and continuous training). The buggy emit-then-check path emits one spurious latent instead."""
    m = _continuous_model()
    m.eval()
    with torch.no_grad():                                  # force STOP at the first decision point (logit >> 0)
        assert m.head.stop_proj is not None
        m.head.stop_proj.weight.zero_()
        m.head.stop_proj.bias.fill_(1e4)
    lat, lengths = m.rollout(["a document that should emit ZERO latents once STOP fires first"],
                             max_steps=8, return_lengths=True)
    emitted = int(lengths.reshape(-1)[0])
    assert emitted == 0, (
        f"continuous rollout emitted {emitted} latent(s) although STOP fired at step 0 — emit-then-check "
        f"off-by-one: the terminal position should emit nothing (check-then-emit, as in FSQ + training)")


if __name__ == "__main__":
    test_continuous_rollout_stops_before_emitting()
    print("PASS")
