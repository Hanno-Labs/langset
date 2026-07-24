"""GOLDEN test (test-first): the hard-negative loss MUST reach the MOVE distribution (the reserved FSQ digit
logits) — but it does NOT on the current code. RED until the loss is fixed.

Symptom that surfaced this (chess neg-on / neg-w10 A/B): P-mass-on-blunders flat at ~34% across no-neg / 0.3 /
1.0 hard-neg weight, in-train (33.8%) matching held-out (33.7%) to the decimal. Tripling the dose moved the
mass by 0.8 pts -> inside the noise floor. The user's read: "if it's emitting blunders on TRAINED data it's a
training or loss problem." Root cause (traced in langset src):
  - `HardNegTerm` (strategies.py) pushes on `c.recon` = `EmissionOut.recon` = `head.encode(target_latents)`
    (strategies.py:473 returns rollout_train_codebook's `recon`; modeling.py:1029 `recon = head.encode(target)`
    where `target` is the STOP-GRAD EMA-twin target).
  - `head.encode` (modeling.py:132) = `up_proj(fsq(down_proj(t)))` -> grad reaches `down_proj`/`up_proj` ONLY.
  - The MOVE at decode time = argmax over the RESERVED cols of `dim_lg`, where `dim_lg = head.emit_logits(hid)
    = level_proj(hid)` (modeling.py:144). `level_proj` and `up_proj` are SEPARATE Linear layers with no shared
    graph edge. So a loss that is a function only of `recon` (=encode) has NO path to `dim_lg`/`level_proj`.
  - => "don't emit this blunder" trains the reconstruction matrix, not the move-prediction logits. No weight
    setting was ever going to reduce blunder-mass — the dose sweep was futile by construction.

This file asserts the CORRECT behavior and is therefore RED on the current code, going GREEN once the loss
routes the negative through the predicted move distribution (dim_lg). Two backward passes on the SAME forward
(authentic head tensors, correct grad flags):
  1. POSITIVE CONTROL (loss_dims, the base FSQ digit CE ON dim_lg): backward -> dim_lg.grad NON-zero.
     PASSES on current code. Proves the apparatus can put gradient on the move logits when the loss actually
     depends on them — so the RED result in test 2 is the loss being broken, not the rig being broken.
  2. THE FINDING (HardNegTerm, asserts the CORRECT contract): backward -> dim_lg.grad and level_proj.grad
     MUST be non-zero (the negative must steer the move). FAILS on current code (both are exactly zero) while
     up_proj/down_proj get grad 1105 / 257 — i.e. the term trains the reconstruction matrix, not the move.

Run:  .venv/bin/python tests/test_hardneg_reaches_move.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset.strategies import HardNegTerm, MultiStepCtx  # noqa: E402


class _FakeTargetSource:
    """Stand-in for the EMA-twin target source: .encode(texts) -> [n, d] stop-grad normalized. The real
    EMATwinTarget.encode returns stop-grad latents (the twin is frozen), so this is faithful for the
    hard-neg bank's grad flag (the only thing HardNegTerm reads off target_source)."""

    def __init__(self, d: int) -> None:
        self.d = d

    def encode(self, texts: list[str]) -> torch.Tensor:
        g = torch.Generator().manual_seed(1 + len(texts))
        return F.normalize(torch.randn(len(texts), self.d, generator=g), dim=-1)


class _FakeTrainer:
    """Only the fields HardNegTerm reads off the trainer: hard_neg_texts (list[list[str]] indexed by row id)."""

    def __init__(self, hard_neg_texts: list[list[str]]) -> None:
        self.hard_neg_texts = hard_neg_texts
        # LabelDimsTerm would read these, but we use loss_dims for the positive control, so leave them None.
        self.label_plan = None
        self.label_cols = None
        self.label_codewords = None


def _build_ctx(model, dev) -> tuple[MultiStepCtx, torch.Tensor, torch.Tensor]:
    """One authentic emission forward's worth of tensors with CORRECT grad flags:
    - recon = head.encode(stop-grad target)  -> depends on down_proj/up_proj (grad), NOT level_proj/dim_lg
    - dim_lg = head.emit_logits(hid)          -> depends on level_proj (grad); hid is the grad leaf
    - target_lat = the stop-grad target latents (same as encode's input)
    Mirrors FSQObjective.emit (strategies.py:433-478) without the backbone (the grad question is head-only)."""
    head = model.head
    B, L, d, h = 2, 2, model.latent_dim, model.h
    V = int(head.fsq_levels)

    # stop-grad EMA target latents (the thing recon tries to reconstruct)
    torch.manual_seed(0)
    target_lat = torch.randn(B, L, d, device=dev).detach()
    digits, recon = head.encode(
        target_lat
    )  # recon = up_proj(fsq(down_proj(target))) -> grad to down/up_proj

    # the model's predicted digit LOGITS (where the move lives). hid is the leaf so grad is well-defined.
    hid = torch.randn(B, L + 1, h, device=dev, requires_grad=True)
    dim_lg, _stop_lg = head.emit_logits(hid)  # dim_lg = level_proj(hid) -> grad to level_proj
    dim_lg.retain_grad()

    valid = torch.ones(B, L, dtype=torch.bool, device=dev)
    lens_l = [L, L]
    bidx = [0, 1]
    hard_neg_texts = [["blunder move Nf2 -420cp"], ["blunder move Qd3 -510cp"]]

    args = M._args("/tmp/_hn")
    args.lam_hard_neg = 1.0  # ensure the term is live
    c = MultiStepCtx(
        trainer=_FakeTrainer(hard_neg_texts),
        args=args,
        model=model,
        dev=dev,
        bidx=bidx,
        lens_l=lens_l,
        flat_texts=[],
        valid=valid,
        target_lat=target_lat,
        recon=recon,
        dim_lg=dim_lg,
        lmax=L,
        fsq_levels=V,
        lab_label=None,
        target_source=_FakeTargetSource(d),
        phase_head=None,
        phase_ids={},
    )
    return c, dim_lg, recon


def _zero_grads(model) -> None:
    for p in model.parameters():
        p.grad = None


def test_positive_control_loss_dims_reaches_dim_lg() -> None:
    """BASE FSQ digit CE on dim_lg (the same loss_dims FSQObjective computes) MUST put gradient on dim_lg /
    level_proj. If this fails, the test apparatus is broken and the hard-neg 'no gradient' result would be
    meaningless (dim_lg might just not be retaining grad). This is the guard that makes test 2 trustworthy."""
    torch.manual_seed(0)
    model = M._build_model()
    dev = torch.device("cpu")
    c, dim_lg, recon = _build_ctx(model, dev)
    fsq_levels = int(model.head.fsq_levels)
    # loss_dims = CE over the non-digit-0 dims at the valid L positions (ignore padding); target = the EMA
    # target's quantized digits (the `digits` from head.encode). Faithful to strategies.py:468-470.
    digits, _ = model.head.encode(c.target_lat.detach())
    lab_rest = torch.full((c.recon.size(0), c.lmax, model.head.fsq_dim - 1), -100, dtype=torch.long)
    for r, nl in enumerate(c.lens_l):
        lab_rest[r, :nl] = digits[r, :nl, 1:]
    loss_dims = F.cross_entropy(
        c.dim_lg[:, : c.lmax, 1:, :].reshape(-1, fsq_levels),
        lab_rest.reshape(-1),
        ignore_index=-100,
    )
    _zero_grads(model)
    loss_dims.backward()
    assert dim_lg.grad is not None, "positive control: dim_lg retained no grad (apparatus broken)"
    dg = float(dim_lg.grad.abs().sum())
    lg = (
        float(model.head.level_proj.weight.grad.abs().sum())
        if model.head.level_proj.weight.grad is not None
        else 0.0
    )
    assert dg > 1e-8, (
        f"positive control: dim_lg got ~0 grad ({dg:.2e}) -> apparatus can't put grad on move logits"
    )
    assert lg > 1e-8, (
        f"positive control: level_proj got ~0 grad ({lg:.2e}) -> the move-digit projection is frozen?"
    )
    print(f"positive_control PASS  dim_lg.grad|sum|={dg:.3e}  level_proj.grad|sum|={lg:.3e}")


def test_hard_neg_reaches_move_logits() -> None:
    """CORRECT contract: a 'don't emit this blunder' negative must put gradient on the MOVE logits (dim_lg /
    level_proj) — the move IS argmax over the reserved dim_lg cols, so a negative that can't reach them cannot
    suppress the blunder no matter its weight. RED on current code: hard-neg backward fills up_proj/down_proj
    (reconstruction) but leaves dim_lg / level_proj at exactly ZERO. Goes GREEN once the loss routes the
    negative through the predicted move distribution (e.g. a MoveNegTerm on dim_lg)."""
    torch.manual_seed(0)
    model = M._build_model()
    dev = torch.device("cpu")
    c, dim_lg, recon = _build_ctx(model, dev)

    contrib = HardNegTerm().contribute(c)
    assert contrib is not None, (
        "HardNegTerm returned None (lam_hard_neg<=0 or no hard_neg_texts) -> term inert"
    )
    _key, loss_hn, _w = contrib
    assert torch.isfinite(loss_hn), f"hard-neg loss not finite: {loss_hn}"

    _zero_grads(model)
    loss_hn.backward()

    # the move logits: MUST receive gradient from a 'don't emit this blunder' negative (RED on current code)
    dg = float(dim_lg.grad.abs().sum()) if dim_lg.grad is not None else 0.0
    lp = (
        float(model.head.level_proj.weight.grad.abs().sum())
        if model.head.level_proj.weight.grad is not None
        else 0.0
    )
    # the reconstruction matrix: MUST receive gradient too (sanity guard — the term trains *something*, so a
    # future 'fix' that silences the term entirely doesn't pass this test by making everything zero)
    up = (
        float(model.head.up_proj.weight.grad.abs().sum())
        if model.head.up_proj.weight.grad is not None
        else 0.0
    )
    dn = (
        float(model.head.down_proj.weight.grad.abs().sum())
        if model.head.down_proj.weight.grad is not None
        else 0.0
    )
    print(
        f"hard_neg: dim_lg.grad|sum|={dg:.3e}  level_proj.grad|sum|={lp:.3e}  "
        f"up_proj.grad|sum|={up:.3e}  down_proj.grad|sum|={dn:.3e}  loss_hn={float(loss_hn):.4f}"
    )
    assert dg > 1e-8, (
        f"RED: hard-neg put {dg:.2e} grad on dim_lg (the MOVE logits) — ZERO. The negative's gradient reaches "
        f"only up_proj/down_proj (reconstruction, grad {up:.2e}/{dn:.2e}), not level_proj/dim_lg (the move), so "
        f"'don't emit this blunder' cannot change which move the model commits to at ANY weight. This is the root "
        f"cause of the flat blunder-mass (no-neg 34.5% -> 0.3 34.1% -> 1.0 33.7%, in-train 33.8%). The loss must "
        f"route the negative through the predicted move distribution (dim_lg) for it to suppress the blunder."
    )
    assert lp > 1e-8, (
        f"RED: level_proj (move-digit projection) got {lp:.2e} grad — ZERO. Same root cause as dim_lg."
    )
    assert up > 1e-8, (
        f"hard-neg put ~0 grad on up_proj ({up:.2e}) -> the term is a TOTAL no-op (would mask the real bug; "
        f"expected it to train the reconstruction matrix at minimum on current code)"
    )
    print(
        "hard_neg_reaches_move PASS  -> negatives now steer the move logits (dim_lg / level_proj); the "
        "'don't emit this blunder' signal can suppress blunders in the move distribution."
    )


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in (
        "test_positive_control_loss_dims_reaches_dim_lg",
        "test_hard_neg_reaches_move_logits",
    ):
        try:
            globals()[name]()
            print(f"{name} PASS\n")
        except AssertionError as e:
            print(f"{name} FAIL -> {e}")
