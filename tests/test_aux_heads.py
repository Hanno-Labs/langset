"""Tests for the PLUGGABLE AUXILIARY-HEAD plug (langset.heads.Head) — the generalization of the phase head.

Three properties, matching the plug's contract:
  (a) BACKWARD-COMPAT / SHIM: `lam_phase>0` is byte-identical to passing the equivalent
      `Head.phase_shim(sup_field, lam_phase)` explicitly (with lam_phase=0). The phase head IS now just a Head.
  (b) PERSISTED regression head (mse, reads="hidden"): trains, is saved with the checkpoint, reloads, and its
      output is READABLE at inference via `model.head_output(name, ...)` — the value/time head you query.
  (c) TRANSIENT head: shapes the geometry (its gradient changes the trained weights) but is NOT persisted.

Tiny random model on CPU/fp32 so it's deterministic and fast (mirrors test_trainer_multi_characterization)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from langset import Head, LangSetModel, Trainer, TrainingArguments
from langset.strategies import last_epoch_selector

TINY_MODEL = os.environ.get(
    "LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"
)


def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_model() -> LangSetModel:
    _seed()
    return LangSetModel.from_pretrained(
        TINY_MODEL, bf16=False, device="cpu", multi_latent=True, fsq_dim=32, fsq_levels=8
    )


# 9 rows: seed -> a SET of stage descriptions, with per-item stage labels (phase / recon-ce), a per-ROW scalar
# `value` (hidden-read regression target, a clean function of the topic), and a per-ITEM `dense` value.
_TOPICS = ["late payment", "theft", "property tax"]
_STAGES = ["filing", "hearing", "ruling"]
_TOPIC_VALUE = {"late payment": 0.1, "theft": 0.5, "property tax": 0.9}
_STAGE_VALUE = {"filing": 0.0, "hearing": 0.5, "ruling": 1.0}


def _dataset() -> list[dict]:
    rows = []
    for i in range(9):
        t = _TOPICS[i % 3]
        rows.append(
            {
                "input_text": f"case {i}: a dispute concerning {t}",
                "target_texts": [f"{s}: the {t} matter at the {s} stage" for s in _STAGES],
                "stage": list(_STAGES),
                "value": _TOPIC_VALUE[t],  # per-row scalar (hidden read site)
                "dense": [_STAGE_VALUE[s] for s in _STAGES],  # per-item scalar (recon read site)
                # per-ROW VECTOR target (Δt, event-observed) — the survival/time-head seam (hidden read site).
                "dt_pair": [_TOPIC_VALUE[t] + 0.05, float(i % 2)],
                # per-ITEM VECTOR target: one (Δt, event) pair PER emitted stage (recon read site).
                "dt_seq": [[_STAGE_VALUE[s] + 0.05, float(j % 2)] for j, s in enumerate(_STAGES)],
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
        seed=0,
        sup_field="stage",
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


def _run(**over: object) -> np.ndarray:
    """Train a fresh tiny model to convergence-of-the-moment and return its flat trainable params."""
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, **over), _dataset()).train()
    return _flat_trainable(model)


# --- (a) BACKWARD-COMPAT: lam_phase is exactly the phase-shim Head -------------------------------------------
def test_lam_phase_equals_phase_shim() -> None:
    """`lam_phase=x` must be byte-identical to `heads=[Head.phase_shim(sup_field, x)]` with lam_phase=0 — proving
    the historical arg is a genuine shim over the generic plug (same class map, same RNG, same summation slot)."""
    via_arg = _run(lam_phase=0.1)
    via_head = _run(lam_phase=0.0, heads=[Head.phase_shim("stage", 0.1)])
    np.testing.assert_allclose(
        via_arg,
        via_head,
        rtol=1e-6,
        atol=1e-7,
        err_msg="lam_phase is not byte-identical to its Head.phase_shim reconstruction",
    )


def test_phase_head_is_not_persisted() -> None:
    """The phase head is TRANSIENT: training with lam_phase>0 must leave nothing persisted on the saved model."""
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, lam_phase=0.1), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    assert len(loaded.aux_heads) == 0, "transient phase head leaked into the persisted checkpoint"


# --- (b) PERSISTED regression head (mse, reads=hidden): trains, saves, reloads, queryable --------------------
def test_persisted_hidden_regression_head_roundtrips_and_reads() -> None:
    """A persisted `Head(reads="hidden", loss="mse")` (the value head shape) trains, serializes with
    the checkpoint, reloads, and `head_output` returns a per-sequence scalar that TRACKS the target."""
    value = Head(name="value", reads="hidden", target="value", loss="mse", dim=1, transient=False)
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        # many small epochs so the Linear(h,1) readout fits the (few, distinct) per-row targets; last_epoch_selector
        # keeps the FINAL (fully-trained) epoch's head rather than the best-by-retr_mrr epoch (which can be ep0).
        Trainer(
            model,
            _args(td, epochs=60, eval_every=1, selector=last_epoch_selector, heads=[value]),
            _dataset(),
        ).train()
        loaded = LangSetModel.load(td, device="cpu")

    assert "value" in loaded.aux_heads, "persisted head missing from the reloaded model"
    spec = loaded.aux_head_specs["value"]
    assert spec["reads"] == "hidden" and spec["loss"] == "mse" and spec["out_dim"] == 1

    seeds = [r["input_text"] for r in _dataset()]
    targets = np.array([r["value"] for r in _dataset()], dtype=np.float64)
    pred = loaded.head_output("value", seeds).squeeze(-1).cpu().numpy().astype(np.float64)
    assert pred.shape == (len(seeds),)
    corr = float(np.corrcoef(pred, targets)[0, 1])
    assert corr > 0.9, (
        f"persisted value head does not track its target at inference (corr={corr:.3f})"
    )


# --- (c) TRANSIENT head shapes the geometry but is not persisted ---------------------------------------------
def test_transient_recon_head_shapes_geometry_without_persisting() -> None:
    """A transient `Head(reads="recon", loss="mse")` injects a real shaping gradient (its presence CHANGES the
    trained weights) yet is NOT written to the checkpoint — the phase-head lifecycle, now generic."""
    dense = Head(name="dense", reads="recon", target="dense", loss="mse", dim=1, transient=True)
    with_head = _run(heads=[dense])
    without_head = _run()  # same seed/data, no head
    max_delta = float(np.max(np.abs(with_head - without_head)))
    assert max_delta > 1e-5, (
        f"transient recon head did not shape the geometry (max param delta {max_delta:.2e})"
    )

    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, heads=[dense]), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    assert "dense" not in loaded.aux_heads, "transient head must not be persisted"
    assert len(loaded.aux_heads) == 0


# --- custom-loss plug (the survival/hazard seam) is wired end-to-end -----------------------------------------
def test_custom_callable_loss_runs() -> None:
    """The custom-callable loss path (needed later for a censored survival/hazard loss) trains without error — the
    callable receives (pred [N,dim], target [N] or [N,k]) and returns a scalar."""

    def huberish(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target if target.dim() == 2 else target.unsqueeze(1)
        keep = torch.isfinite(t).all(dim=-1)
        if not bool(keep.any()):
            return pred.new_zeros(())
        return torch.nn.functional.smooth_l1_loss(pred[keep], t[keep])

    head = Head(name="surv", reads="hidden", target="value", loss=huberish, dim=1, transient=False)
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, epochs=3, heads=[head]), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    out = loaded.head_output("surv", [r["input_text"] for r in _dataset()])
    assert tuple(out.shape) == (9, 1)


# --- custom loss with a MULTI-COLUMN per-item vector target: the time-to-event / survival seam ---------------
def _censored(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Toy censored exponential-hazard NLL. REQUIRES a 2-column target — indexing target[:, 1] IndexErrors unless
    the trainer actually feeds the per-item VECTOR target through. dt=target[:,0], obs=target[:,1] (0=censored)."""
    assert target.dim() == 2 and target.size(1) == 2, (
        f"expected [N, 2] target, got {tuple(target.shape)}"
    )
    dt, obs = target[:, 0], target[:, 1]
    haz = torch.nn.functional.softplus(pred[:, 0]) + 1e-4  # positive hazard rate
    return (
        haz * dt - obs * torch.log(haz)
    ).mean()  # -log-likelihood of an exponential w/ right-censoring


def test_custom_loss_vector_target_hidden() -> None:
    """A custom loss with a MULTI-COLUMN per-item target (the survival/hazard TIME head) trains end to end at the
    HIDDEN read site — proving the trainer feeds the (Δt, event) VECTOR through, not just a scalar — and the
    persisted head reads back at the right shape."""
    dt_head = Head(
        name="dt", reads="hidden", target="dt_pair", loss=_censored, dim=1, transient=False
    )
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, epochs=3, heads=[dt_head]), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    assert loaded.aux_head_specs["dt"]["loss"] == "custom"
    out = loaded.head_output("dt", [r["input_text"] for r in _dataset()])
    assert tuple(out.shape) == (9, 1)


def test_custom_loss_vector_target_recon() -> None:
    """Same survival seam at the RECON read site: a per-ITEM (Δt, event) pair per emitted latent flows through to
    the custom loss as an [N, 2] target, and reduce='none' gives the dense per-tick hazard readout."""
    dt_head = Head(
        name="dt", reads="recon", target="dt_seq", loss=_censored, dim=1, transient=False
    )
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, epochs=3, heads=[dt_head]), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    dense = loaded.head_output("dt", [r["input_text"] for r in _dataset()], reduce="none")
    assert isinstance(dense, list) and len(dense) == 9
    assert all(t.dim() == 2 and t.size(1) == 1 for t in dense)  # [Li, 1] per row


# --- bf16 dtype regression (0.13.1) -------------------------------------------------------------------------
def test_bf16_hidden_and_recon_heads_train() -> None:
    """REGRESSION: under bf16 the backbone hidden/recon is BFloat16 but the aux-head Linear is Float, so applying
    the head must cast the input (like head_output does) — otherwise the training forward raises
    `mat1 and mat2 must have the same dtype, but got BFloat16 and Float`. The fp32 CPU tests never exercised this;
    a bf16 model with a hidden AND a recon head must train without a dtype error."""
    heads = [
        Head(name="v", reads="hidden", target="value", loss="mse", dim=1, transient=False),
        Head(name="d", reads="recon", target="dense", loss="mse", dim=1, transient=False),
    ]
    model = LangSetModel.from_pretrained(
        TINY_MODEL, bf16=True, device="cpu", multi_latent=True, fsq_dim=32, fsq_levels=8
    )
    with tempfile.TemporaryDirectory() as td:
        Trainer(
            model, _args(td, epochs=2, heads=heads), _dataset()
        ).train()  # must not raise a dtype RuntimeError
        loaded = LangSetModel.load(td, device="cpu")
    assert "v" in loaded.aux_heads and "d" in loaded.aux_heads


# --- fail-fast validation (Copilot review) ------------------------------------------------------------------
def test_duplicate_head_names_raise() -> None:
    """Head.name keys the log entry, the checkpoint, and the head_output lookup — two heads with the same name
    would silently overwrite, so resolution must reject them."""
    dup = [
        Head(name="v", reads="hidden", target="value", loss="mse", dim=1, transient=False),
        Head(name="v", reads="recon", target="dense", loss="mse", dim=1, transient=False),
    ]
    model = _build_model()
    with (
        tempfile.TemporaryDirectory() as td,
        pytest.raises(ValueError, match="duplicate Head name"),
    ):
        Trainer(model, _args(td, heads=dup), _dataset()).train()


def test_ce_head_empty_classes_raises() -> None:
    """A CE head whose `target` labels are ALL missing yields a [N, 0] logit -> F.cross_entropy fails opaquely;
    resolution must raise a clear error instead."""
    rows = [
        {"input_text": f"row {i}", "target_texts": ["a", "b"], "empty": ["", "unknown"]}
        for i in range(6)
    ]
    head = Head(name="e", reads="recon", target="empty", loss="ce")
    model = _build_model()
    with tempfile.TemporaryDirectory() as td, pytest.raises(ValueError, match="no classes"):
        Trainer(model, _args(td, sup_field=None, heads=[head]), rows).train()


def test_ce_head_dim_mismatch_raises() -> None:
    """A CE head's width IS its class count — a user-set `dim` that disagrees must be rejected up front, not crash
    at loss time."""
    head = Head(name="p", reads="recon", target="stage", loss="ce", dim=99)
    model = _build_model()
    with tempfile.TemporaryDirectory() as td, pytest.raises(ValueError, match="conflicts with"):
        Trainer(model, _args(td, heads=[head]), _dataset()).train()


def test_head_output_bad_reduce_raises() -> None:
    """A typo'd `reduce` must fail loud, not silently take the mean path."""
    head = Head(name="value", reads="recon", target="dense", loss="mse", dim=1, transient=False)
    model = _build_model()
    with tempfile.TemporaryDirectory() as td:
        Trainer(model, _args(td, epochs=2, heads=[head]), _dataset()).train()
        loaded = LangSetModel.load(td, device="cpu")
    with pytest.raises(ValueError, match="reduce must be"):
        loaded.head_output("value", [r["input_text"] for r in _dataset()], reduce="meann")
