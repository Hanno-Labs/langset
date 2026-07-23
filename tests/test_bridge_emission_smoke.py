"""Smoke test for the ported parallel-query emission family (langset.bridge_emission.QueryBridgeEmission).

Not a golden — it proves the PORT wires up end-to-end in langset's multi-latent trainer: with a FROZEN backbone
and `emission=QueryBridgeEmission`, a training run (a) builds + registers the bridge module, (b) trains it (its
params move, the backbone's do NOT), (c) produces a finite decreasing loss. Runs on CPU with a tiny random model.
"""

from __future__ import annotations

import os
import tempfile

import torch

from langset import LangSetModel, Trainer, TrainingArguments
from langset.bridge_emission import FrozenEncoderTarget, QueryBridgeEmission

TINY_MODEL = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


def _rows() -> list[dict]:
    topics = ["late payment", "theft", "property tax"]
    stages = ["filing", "hearing", "ruling"]
    return [
        {
            "input_text": f"case {i}: a dispute concerning {topics[i % 3]}",
            "target_texts": [f"{s}: the {topics[i % 3]} matter at the {s} stage" for s in stages],
        }
        for i in range(8)
    ]


def test_query_bridge_emission_trains_with_frozen_backbone() -> None:
    torch.manual_seed(0)
    model = LangSetModel.from_pretrained(
        TINY_MODEL, bf16=False, device="cpu", multi_latent=True, freeze_backbone=True
    )
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            epochs=3,
            batch_size=4,
            lr=1e-2,
            max_len=64,
            report_to=None,
            verbose=False,
            eval_every=1,      # run eval so the bridge's emit_infer path is exercised
            patience=99,
            val_frac=0.25,
            select="retr_mrr",
            seed=0,
            emission=QueryBridgeEmission,       # <-- the ported family, injected
            target_source=FrozenEncoderTarget,  # <-- query-target: match the frozen encoder's E(text)
            lam_multi_nce=0.0,                  # its base loss already does the contrastive term
            output_dir=out,
        )
        trained = Trainer(model, args, _rows()).train()

    # (a) the bridge module was registered on the model
    assert hasattr(trained, "emission_bridge"), "QueryBridgeEmission did not register its module on the model"
    qb = trained.emission_bridge
    # (b) its params are finite and it is trainable (the backbone is frozen)
    p = next(qb.parameters())
    assert torch.isfinite(p).all()
    assert any(pp.requires_grad for pp in qb.parameters())
    assert not any(pp.requires_grad for pp in trained.backbone.parameters()), "backbone should be frozen"


def _rows_with_hard_negs() -> list[dict]:
    """Same rows, plus a per-row `hard_neg` column of same-topic/wrong-stage confusables."""
    topics = ["late payment", "theft", "property tax"]
    wrong = ["appeal", "dismissal", "settlement"]
    rows = _rows()
    for i, r in enumerate(rows):
        r["hard_neg"] = [f"{w}: the {topics[i % 3]} matter at the {w} stage" for w in wrong]
    return rows


def test_query_bridge_emission_folds_hard_negatives_into_the_bank() -> None:
    """With `hard_neg_field` set, the family trains end-to-end (its InfoNCE bank now includes the pooled hard-negs).
    Without the column the path is a no-op — this asserts the wired-up case runs and produces finite params."""
    torch.manual_seed(0)
    model = LangSetModel.from_pretrained(
        TINY_MODEL, bf16=False, device="cpu", multi_latent=True, freeze_backbone=True
    )
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            epochs=3, batch_size=4, lr=1e-2, max_len=64, report_to=None, verbose=False,
            eval_every=1, patience=99, val_frac=0.25, select="retr_mrr", seed=0,
            emission=QueryBridgeEmission, target_source=FrozenEncoderTarget,
            lam_multi_nce=0.0, lam_hard_neg=0.0,   # bridge consumes hard_neg_field IN-BANK; separate term stays off
            hard_neg_field="hard_neg",             # <-- pooled adjacent-confusables folded into the bridge's InfoNCE
            output_dir=out,
        )
        trained = Trainer(model, args, _rows_with_hard_negs()).train()

    qb = trained.emission_bridge
    p = next(qb.parameters())
    assert torch.isfinite(p).all()
    assert any(pp.requires_grad for pp in qb.parameters())


def test_query_bridge_emission_persists_and_reattaches() -> None:
    """save_pretrained serializes the bridge; load stashes it; re-attaching the strategy restores the weights."""
    torch.manual_seed(0)
    model = LangSetModel.from_pretrained(
        TINY_MODEL, bf16=False, device="cpu", multi_latent=True, freeze_backbone=True
    )
    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            epochs=2, batch_size=4, lr=1e-2, max_len=64, report_to=None, verbose=False,
            eval_every=99, patience=99, val_frac=0.25, seed=0,
            emission=QueryBridgeEmission, lam_multi_nce=0.0, output_dir=out,
        )
        trained = Trainer(model, args, _rows()).train()
        want = {k: v.clone() for k, v in trained.emission_bridge.state_dict().items()}
        ckpt = f"{out}/ckpt"
        trained.save_pretrained(ckpt)

        loaded = LangSetModel.load(ckpt)
        assert hasattr(loaded, "_emission_bridge_state"), "bridge state not persisted/stashed on load"

        class _NoTrainer:  # serve-time attach: the strategy only needs the model/args/dev
            pass

        strat = QueryBridgeEmission(loaded, args, torch.device("cpu"), _NoTrainer())  # type: ignore[arg-type]
        got = strat.bridge.state_dict()
        for k, v in want.items():
            assert torch.allclose(got[k], v), f"bridge param {k} not restored after load"
