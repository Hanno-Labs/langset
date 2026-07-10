"""Repro for the multi-latent text-replay bugs flagged in Copilot review of PR #8.

#1  all rows tagged `learn` -> the emit/latent split (`_emb`) is empty, so tr_idx/val_idx are empty and the
    `or perm[:1]` fallback can't help (perm itself is empty). Training should fail LOUDLY (there are no emit
    rows for the latent objective), not silently no-op or crash cryptically deep in eval.
"""
from __future__ import annotations

import tempfile

import pytest
import torch

from test_trainer_multi_characterization import _args, _build_model

from langset import Trainer
from langset.trainer import _replay_ce, _tokenize_replay


def _row0_replay_loss(m, docs, tgts, doc_side):
    """Replay CE for the FIRST row, tokenized in a batch with `docs`/`tgts` (doc padded to the batch max)."""
    di, dm = _tokenize_replay(m.tokenizer, docs, 64, doc_side, m.device)
    ti, tm = _tokenize_replay(m.tokenizer, tgts, 32, "right", m.device)
    with torch.no_grad():
        return float(_replay_ce(m, di[:1], dm[:1], ti[:1], tm[:1], m.vocab_size))


def test_replay_ce_is_padding_invariant():
    # Copilot #2: the short row's replay loss must not depend on a longer BATCHMATE forcing it to be padded.
    m = _build_model()
    short_d, short_t = "hi there", "a short answer"
    long_d = "a considerably longer conditioning document that forces the shorter row to be padded in the batch"
    long_t = "a much longer target answer with many more tokens than the short one to also force target padding"
    solo = _row0_replay_loss(m, [short_d], [short_t], "left")                        # short row alone, no doc pad
    left = _row0_replay_loss(m, [short_d, long_d], [short_t, long_t], "left")         # doc LEFT-padded (the fix)
    right = _row0_replay_loss(m, [short_d, long_d], [short_t, long_t], "right")       # doc RIGHT-padded (the bug)
    d_left, d_right = abs(solo - left), abs(solo - right)
    print(f"\nsolo={solo:.5f} left={left:.5f} right={right:.5f} | d_left={d_left:.2e} d_right={d_right:.2e}")
    assert d_right > 1e-2, f"right-pad should shift the loss (the bug): d_right={d_right:.2e}"
    assert d_left < 1e-4, f"left-pad must keep the row's loss invariant (the fix): d_left={d_left:.2e}"


def test_replay_training_runs_multi_latent():
    # integration: emit rows + a couple varying-length learn rows, replay firing every step -> the rewired
    # multi-latent learn path must train end-to-end (exercises _tokenize_replay + _replay_ce on real batches).
    from test_trainer_multi_characterization import _flat_trainable
    m = _build_model()
    rows = [{"input_text": f"case {i}: a dispute concerning topic {i % 3}",
             "target_texts": [f"stage A of {i}", f"stage B of {i}"], "tag": "emit"} for i in range(6)]
    rows += [{"input_text": f"domain fact {i} to keep the backbone fluent and fluent and fluent and fluent",
              "target_texts": [f"restated fact {i}"], "tag": "learn"} for i in range(2)]
    before = _flat_trainable(m)
    with tempfile.TemporaryDirectory() as td:
        args = _args(td, learn_field="tag", learn_ratio=1.0, epochs=2,
                     sup_field=None, lam_sup=0.0, hard_neg_field=None, lam_hard_neg=0.0,
                     label_dims=None, lam_label_dims=0.0)
        Trainer(m, args, rows).train()                  # must not raise; learn_loss fires each step
    assert float(((_flat_trainable(m) - before) ** 2).sum()) > 0.0    # training actually moved the weights


def _all_learn_rows() -> list[dict]:
    return [{"input_text": f"domain fact {i} to keep the backbone fluent",
             "target_texts": [f"the fact {i} restated"],
             "tag": "learn"} for i in range(6)]


def test_all_rows_learn_tagged_errors_clearly():
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        args = _args(td, learn_field="tag", learn_ratio=0.5,
                     sup_field=None, lam_sup=0.0, hard_neg_field=None, lam_hard_neg=0.0,
                     label_dims=None, lam_label_dims=0.0)
        with pytest.raises(ValueError):        # no emit rows left -> a clear, upfront error
            Trainer(model, args, _all_learn_rows()).train()
