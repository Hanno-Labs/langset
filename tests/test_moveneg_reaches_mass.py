"""GOLDEN test (test-first): a DISTRIBUTION-level negative (MoveNegTerm) must suppress the blunder's probability
MASS directly on the reserved-digit softmax — RED until the term exists, GREEN once it does.

Why this term: HardNegTerm (even fixed to route through dim_lg) is a latent-cosine contrastive — it tightens the
ARGMAX / committed move but leaves the MASS ~flat (chess A/B: P-mass-on-blunders 34.5% -> 33.3% across no-neg /
0.3 / 1.0 / fixed-loss). The reason: a latent nudge pulls the single committed move away from the blunder, but
does not directly pin the digit PROBABILITIES, so the broad distribution over moves still wants blunders ~1/3 of
the time. MoveNegTerm operates on the reserved-digit softmax (dim_lg) DIRECTLY: for each blunder's codeword
(looked up via label_codewords — the same map the positive label uses), P(blunder) = Π over reserved digits of
softmax(dim_lg[digit])[code]; minimizing Σ P(blunder) pushes the blunder's digit probabilities DOWN and (by
softmax normalization) the non-blunder levels UP. That is the mass-reshaping contract.

This test asserts the precise mechanism, not just "grad reaches dim_lg" (trivially true for any dim_lg loss):
  - the gradient of the loss w.r.t. the BLUNDER's digit logit is POSITIVE (gradient descent will push it DOWN);
  - the gradient on NON-blunder levels of the same digit is NEGATIVE (pushed UP, redistributing the mass);
  - non-reserved digits and the STOP position receive ZERO gradient (the term is scoped to the move digits).
Run:  .venv/bin/python tests/test_moveneg_reaches_mass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset.strategies import MoveNegTerm, MultiStepCtx  # noqa: E402


class _FakeTrainer:
    """Only the fields MoveNegTerm reads: the PRECOMPUTED padded codeword-index tensor + counts (mirrors what
    Trainer.__init__ builds) + label_plan. The step loop never builds these — __init__ does, once."""

    def __init__(self, move_neg_codes: list[list[tuple]], label_plan: list) -> None:
        self.label_plan = label_plan
        self.hard_neg_texts = None
        # precompute the padded [N_rows, max_neg, n_reserved] index tensor + counts, on CPU (test runs on CPU)
        nr = len(label_plan)
        maxk = max((len(rc) for rc in move_neg_codes), default=0)
        n_rows = len(move_neg_codes)
        pad = torch.full((n_rows, maxk, nr), -1, dtype=torch.long)
        counts = torch.zeros(n_rows, dtype=torch.long)
        for i, rc in enumerate(move_neg_codes):
            for j, cw in enumerate(rc):
                pad[i, j] = torch.tensor(cw, dtype=torch.long)
            counts[i] = len(rc)
        self.move_neg_idx = pad
        self.move_neg_n = counts
        self.legal_move_idx = None
        self.legal_move_n = None


def _build_ctx(model, dev, blunder_code: tuple) -> tuple[MultiStepCtx, torch.Tensor]:
    """One authentic emission forward's worth of tensors. dim_lg = head.emit_logits(hid) (grad leaf = hid),
    retained. Reserved layout mirrors chess label_dims={move:[1,2]} (2 reserved digits at rest cols 0,1)."""
    head = model.head
    B, L, h = 1, 1, model.h
    V = int(head.fsq_levels)

    torch.manual_seed(0)
    hid = torch.randn(B, L + 1, h, device=dev, requires_grad=True)
    dim_lg, _stop_lg = head.emit_logits(hid)  # [B, L+1, fsq_dim, V]
    dim_lg.retain_grad()

    # reserved layout: rest cols [0, 1] -> full dims [1, 2]; a single "move" facet spanning both digit positions
    label_plan = [(0, "move", 0), (1, "move", 1)]

    valid = torch.ones(B, L, dtype=torch.bool, device=dev)
    args = M._args("/tmp/_mn")
    args.lam_move_neg = 1.0
    # move_neg_codes is per-row (list of codeword tuples); one blunder on row 0, applied to each valid tick
    codes = [[blunder_code]]
    c = MultiStepCtx(
        trainer=_FakeTrainer(codes, label_plan),
        args=args,
        model=model,
        dev=dev,
        bidx=[0],
        lens_l=[L],
        flat_texts=[],
        valid=valid,
        target_lat=torch.zeros(B, L, model.latent_dim, device=dev),
        recon=torch.zeros(B, L, model.latent_dim, device=dev),
        dim_lg=dim_lg,
        lmax=L,
        fsq_levels=V,
        lab_label=None,
        target_source=None,
        phase_head=None,
        phase_ids={},
    )
    return c, dim_lg


def test_move_neg_suppresses_blunder_mass() -> None:
    """RED until MoveNegTerm exists and correctly minimizes P(blunder) on the reserved-digit softmax. Asserts the
    mass-reshaping mechanism: blunder digit-level grad > 0 (pushed DOWN), non-blunder levels < 0 (pushed UP),
    non-reserved digits + STOP position untouched."""
    torch.manual_seed(0)
    model = M._build_model()
    dev = torch.device("cpu")
    blunder = (3, 5)  # digit 0 -> level 3, digit 1 -> level 5 (full dims 1 and 2)
    c, dim_lg = _build_ctx(model, dev, blunder)

    contrib = MoveNegTerm().contribute(c)
    assert contrib is not None, (
        "MoveNegTerm returned None (lam_move_neg<=0 / no codes / no label_plan / no dim_lg)"
    )
    _k, loss_mn, _w = contrib
    assert torch.isfinite(loss_mn), f"move-neg loss not finite: {loss_mn}"

    for p in model.parameters():
        p.grad = None
    loss_mn.backward()

    g = dim_lg.grad
    assert g is not None, "dim_lg retained no grad (apparatus broken)"
    # the blunder occupies reserved digit 0 = full dim 1 (level 3) and reserved digit 1 = full dim 2 (level 5).
    # CE-shaped: grad on the blunder level is POSITIVE (descent pushes the logit DOWN -> P DOWN); grad on the
    # other levels of the same digit is NEGATIVE (softmax normalization pushes them UP -> mass redistributes).
    d0_blunder = float(g[0, 0, 1, blunder[0]])
    d1_blunder = float(g[0, 0, 2, blunder[1]])
    d0_others = [float(g[0, 0, 1, k]) for k in range(int(model.head.fsq_levels)) if k != blunder[0]]
    d1_others = [float(g[0, 0, 2, k]) for k in range(int(model.head.fsq_levels)) if k != blunder[1]]

    print(
        f"move_neg: loss={float(loss_mn):.5f}  "
        f"d0[blunder]={d0_blunder:+.3e} d0[others]min={min(d0_others):+.3e}  "
        f"d1[blunder]={d1_blunder:+.3e} d1[others]min={min(d1_others):+.3e}"
    )
    assert d0_blunder > 0, f"blunder digit 0 level grad not positive (push-down): {d0_blunder:.3e}"
    assert d1_blunder > 0, f"blunder digit 1 level grad not positive (push-down): {d1_blunder:.3e}"
    assert all(x < 0 for x in d0_others), (
        f"non-blunder levels of digit 0 not all negative (push-up): {d0_others}"
    )
    assert all(x < 0 for x in d1_others), (
        f"non-blunder levels of digit 1 not all negative (push-up): {d1_others}"
    )
    # non-reserved digits (e.g. full dim 3) and the STOP position (index 1) must be untouched
    assert float(g[0, 0, 3, :].abs().sum()) == 0.0, "move-neg leaked grad onto a non-reserved digit"
    assert float(g[0, 1, :, :].abs().sum()) == 0.0, "move-neg leaked grad onto the STOP position"
    print(
        "move_neg_suppresses_blunder_mass PASS  -> the blunder's digit probabilities are pushed DOWN and the "
        "non-blunder levels UP (mass redistributes off the blunder); non-reserved digits + STOP untouched."
    )


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("test_move_neg_suppresses_blunder_mass",):
        try:
            globals()[name]()
            print(f"{name} PASS\n")
        except (AssertionError, ImportError, AttributeError) as e:
            print(f"{name} FAIL -> {e}")
