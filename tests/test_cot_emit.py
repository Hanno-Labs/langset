"""REPRODUCING tests for the CoT-emit (Exp-B) bugs surfaced in PR #6 review (Copilot comments 1 & 2).

These assert the CORRECT behavior, so they are RED on the current code until the fixes land. Run:
    .venv/bin/python tests/test_cot_emit.py
"""
from __future__ import annotations

import sys
import tempfile

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset import Trainer, TrainingArguments  # noqa: E402
from langset.strategies import CoTGenTerm, MultiStepCtx  # noqa: E402


class _FakeTrainer:                                        # only the fields CoTGenTerm reads off the trainer
    def __init__(self, input_text: list[str], cot_texts: list[str]) -> None:
        self.input_text, self.cot_texts = input_text, cot_texts


def _cot_loss(model, padding_side: str) -> float:
    """CoTGenTerm loss on a MIXED-length-seed batch with the tokenizer forced to `padding_side`."""
    tok = model.tokenizer
    rows_seed = ["short", "a considerably longer seed sentence with many more tokens than the first row"]
    cots = ["reason about row 0 step by step and conclude", "reason about row 1 step by step and conclude"]
    tr = _FakeTrainer(rows_seed, cots)
    c = MultiStepCtx(trainer=tr, args=M._args("/tmp/_x"), model=model, dev=torch.device("cpu"), bidx=[0, 1],
                     lens_l=[1, 1], flat_texts=[], valid=None, target_lat=None, recon=None, dim_lg=None,
                     lmax=1, fsq_levels=8, lab_label=None, target_source=None, phase_head=None, phase_ids={})
    torch.manual_seed(0)
    tok.padding_side = padding_side
    return float(CoTGenTerm().contribute(c)[1].item())


def test_cot_loss_is_padding_side_invariant() -> None:
    """Copilot #2: CoTGenTerm must pin left-padding internally (like the emission forward), so the last real seed
    token lands at index sd-1 regardless of the tokenizer's default. Otherwise a right-padding tokenizer conditions
    the CoT CE on padding for shorter seeds. The loss must therefore be INVARIANT to tok.padding_side."""
    model = M._build_model()
    right = _cot_loss(model, "right")
    left = _cot_loss(model, "left")
    assert abs(right - left) < 1e-6, (
        f"CoT loss depends on tokenizer padding side (right={right:.6f} left={left:.6f}) — _tokm reads the mutable "
        f"tokenizer default instead of pinning left; short seeds get conditioned on padding")


def _cot_texts_for(rows: list[dict], column_mapping=None) -> list[str]:
    model = M._build_model()
    with tempfile.TemporaryDirectory() as td:                          # bare args: exercise only the __init__ cot_text load
        args = TrainingArguments(epochs=1, batch_size=4, output_dir=td, report_to=None, verbose=False)
        tr = Trainer(model, args, rows, column_mapping=column_mapping)
    return tr.cot_texts


def test_cot_text_none_is_empty_not_literal_none() -> None:
    """Copilot #1: a row with cot_text=None must load as "" (CoT off for that row), not the truthy literal "None"
    (which would train the model to generate the word 'None')."""
    rows = [{"input_text": f"seed {i}", "target_texts": ["a", "b"], "cot_text": (None if i == 0 else "reason here")}
            for i in range(4)]
    cot = _cot_texts_for(rows)
    assert cot[0] == "", f'cot_text=None should load as "" but got {cot[0]!r}'


def test_cot_text_respects_column_mapping() -> None:
    """Copilot #1: cot_text loading must honor column_mapping (go through the canonical-name lookup), not index the
    raw "cot_text" column directly — otherwise a renamed reasoning column is silently ignored (CoT never trains)."""
    rows = [{"input_text": f"seed {i}", "target_texts": ["a", "b"], "reasoning": f"think about {i}"} for i in range(4)]
    cot = _cot_texts_for(rows, column_mapping={"reasoning": "cot_text"})
    assert cot[2] == "think about 2", f"column_mapping to cot_text was ignored; got {cot[2]!r}"


def _run(rows, **over) -> "list":
    import numpy as np
    model = M._build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, M._args(td, **over), rows).train()
    return M._flat_trainable(model)


def test_cot_training_runs_and_is_finite() -> None:
    """Copilot #3: injecting the CoT strategy pair on rows with a cot_text column trains end-to-end (the isolated
    CoT backward runs) and stays finite."""
    import numpy as np
    from langset.strategies import build_cot_loss_terms, cot_seed_texts
    rows = M._dataset()
    for i, r in enumerate(rows):
        r["cot_text"] = f"reason about case {i}: identify the topic, then each stage, then conclude"
    post = _run(rows, loss_terms=build_cot_loss_terms, seed_builder=cot_seed_texts, lam_cot=1.0)
    assert np.isfinite(post).all() and float(np.abs(post).sum()) > 0.0, "CoT training didn't run / went non-finite"


def test_cot_column_alone_is_a_noop() -> None:
    """Copilot #3: a cot_text column WITHOUT injecting the CoT strategies must not change training — the default
    strategies ignore it (byte-identical to the same rows with no cot_text column)."""
    import numpy as np
    base_rows = M._dataset()
    cot_rows = M._dataset()
    for r in cot_rows:
        r["cot_text"] = "some reasoning that must be ignored when the CoT strategies are not injected"
    base = _run(base_rows)
    with_col = _run(cot_rows)
    assert np.array_equal(base, with_col), "a cot_text column changed default training — it must be inert when not injected"


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("test_cot_loss_is_padding_side_invariant",
                 "test_cot_text_none_is_empty_not_literal_none",
                 "test_cot_text_respects_column_mapping",
                 "test_cot_training_runs_and_is_finite",
                 "test_cot_column_alone_is_a_noop"):
        try:
            globals()[name]()
            print(f"{name} PASS")
        except AssertionError as e:
            print(f"{name} FAIL -> {e}")
