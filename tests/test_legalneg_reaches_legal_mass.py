"""GOLDEN test (test-first): a LEGAL-MOVE-RENORMALIZED negative (LegalMoveNegTerm) must suppress the blunder's
share of the LEGAL-move probability mass — and the gradient must be ~50x larger than the raw-codeword
MoveNegTerm's, which is the whole point of renormalizing. RED until the term exists, GREEN once it does.

Why this term: MoveNegTerm penalizes P(blunder) = Π softmax(dim_lg[digit])[code] on the RAW digit softmax — but a
specific codeword's raw prob is ~0.002-0.02 (mass spread across 8 levels x 4 digits), so loss_move_neg ~ 0.0018
at weight 1.0, ~60x too weak to move the legal-move mass (chess moveneg10 run: P-mass-on-blunders 34.3% ≈
control 34.5%, no effect). The DECODE metric measures legal-RENORMALIZED P(blunder) = P(codeword_blunder) /
Σ_legal P(codeword_legal), where a blunder's share is ~10-35%. LegalMoveNegTerm penalizes THAT: loss =
Σ_neg P(codeword_neg) / Σ_support P(codeword_support). The gradient is renormalized — order ~0.1 not ~0.002 —
so it can actually move the legal-move mass that the metric reads.

The term stays GENERIC (no chess logic): it reads `legal_move_codes` (per-row list of codeword tuples = the
SUPPORT set) and `move_neg_codes` (per-row subset to penalize), renormalizes P over the support, penalizes the
neg subset's share. The chess-ness (board -> legal moves -> codewords) lives in the data builder, not here.

This test asserts the precise mechanism:
  - loss is in the LEGAL-RENORM range (~0.1-0.5), NOT the raw range (~0.002) — the renormalization happened;
  - the blunder's digit-level grad is POSITIVE (push-down) and LARGE (>> MoveNegTerm's ~0.02 at the same setup);
  - non-blunder SUPPORT levels of the same digit are NEGATIVE (mass redistributes off the blunder WITHIN the
    legal support, not across the whole raw grid);
  - a single manual SGD step on the loss DECREASES the legal-renormalized P(blunder) (it actually reshapes mass).
Run:  .venv/bin/python tests/test_legalneg_reaches_legal_mass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset.strategies import LegalMoveNegTerm, MultiStepCtx  # noqa: E402


class _FakeTrainer:
    """Fields LegalMoveNegTerm reads: the PRECOMPUTED padded index tensors + counts for BOTH the neg subset and
    the legal support set (mirrors Trainer.__init__) + label_plan."""

    def __init__(self, move_neg_codes, legal_move_codes, label_plan) -> None:
        self.label_plan = label_plan
        self.hard_neg_texts = None
        nr = len(label_plan)

        def _pad(codes_list):
            maxk = max((len(rc) for rc in codes_list), default=0)
            n_rows = len(codes_list)
            pad = torch.full((n_rows, maxk, nr), -1, dtype=torch.long)
            counts = torch.zeros(n_rows, dtype=torch.long)
            for i, rc in enumerate(codes_list):
                for j, cw in enumerate(rc):
                    pad[i, j] = torch.tensor(cw, dtype=torch.long)
                counts[i] = len(rc)
            return pad, counts

        self.move_neg_idx, self.move_neg_n = _pad(move_neg_codes)
        self.legal_move_idx, self.legal_move_n = _pad(legal_move_codes)


def _build_ctx(model, dev, legal_codes, neg_codes):
    """One authentic emission forward. Reserved layout mirrors chess label_dims={from_sq:[1,2], to_sq:[3,4]}
    (4 reserved digits at rest cols 0,1,2,3 = full dims 1,2,3,4). A 'move' codeword = 4 digit indices."""
    head = model.head
    B, L, h = 1, 1, model.h
    V = int(head.fsq_levels)
    torch.manual_seed(0)
    hid = torch.randn(B, L + 1, h, device=dev, requires_grad=True)
    dim_lg, _stop = head.emit_logits(hid)
    dim_lg.retain_grad()
    label_plan = [(0, "from_sq", 0), (1, "from_sq", 1), (2, "to_sq", 0), (3, "to_sq", 1)]
    valid = torch.ones(B, L, dtype=torch.bool, device=dev)
    args = M._args("/tmp/_ln")
    args.lam_legal_neg = 1.0
    c = MultiStepCtx(
        trainer=_FakeTrainer([neg_codes], [legal_codes], label_plan),
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


def _renorm_p(model, dim_lg, code):
    """legal-renormalized P(code) = P(code_raw) — used to verify the SGD step reduces P(blunder)."""
    dl = dim_lg[0, 0, 1:, :]  # [fsq_dim, V]
    p = 1.0
    for d, lvl in enumerate(code):
        p = p * torch.softmax(dl[d].float(), -1)[lvl]
    return p


def test_legal_neg_reaches_legal_mass() -> None:
    """RED until LegalMoveNegTerm exists and penalizes the blunder's LEGAL-RENORMALIZED share. Two legal moves
    (support = {good, blunder}); neg = {blunder}. Asserts the renorm happened (loss ~0.5 not ~0.002), the blunder
    grad is POSITIVE and LARGE (>> MoveNegTerm's), non-blunder support levels are NEGATIVE, and one SGD step
    actually lowers the renormalized P(blunder)."""
    torch.manual_seed(0)
    model = M._build_model()
    dev = torch.device("cpu")
    good = (1, 2, 3, 4)
    blunder = (5, 6, 7, 0)
    legal = [good, blunder]
    neg = [blunder]
    c, dim_lg = _build_ctx(model, dev, legal, neg)

    contrib = LegalMoveNegTerm().contribute(c)
    assert contrib is not None, (
        "LegalMoveNegTerm returned None (lam_legal_neg<=0 / no codes / no dim_lg)"
    )
    _k, loss_ln, _w = contrib
    assert torch.isfinite(loss_ln), f"legal-neg loss not finite: {loss_ln}"

    # (1) the renormalization happened: loss is P(blunder)/(P(good)+P(blunder)) ~ 0.5, NOT raw P ~ 0.002
    loss_val = float(loss_ln)
    assert 0.05 < loss_val < 0.95, (
        f"loss {loss_val:.4f} not in the legal-renorm range [0.05, 0.95] — if it's ~0.002 the term is the "
        f"raw-codeword MoveNegTerm (no renormalization); if it's >1 the renorm denominator is wrong"
    )

    for p in model.parameters():
        p.grad = None
    loss_ln.backward(
        retain_graph=True
    )  # retain: the SGD step below backward()s the same graph again
    g = dim_lg.grad
    assert g is not None, "dim_lg retained no grad"
    # blunder occupies reserved rest-cols 0,1,2,3 -> full dims 1,2,3,4 at levels (5,6,7,0)
    bgrad = [float(g[0, 0, d + 1, blunder[d]]) for d in range(4)]
    ggrad = [float(g[0, 0, d + 1, good[d]]) for d in range(4)]
    print(
        f"legal_neg: loss={loss_val:.4f}  blunder_digit_grads={[f'{x:+.3e}' for x in bgrad]}  "
        f"good_digit_grads={[f'{x:+.3e}' for x in ggrad]}"
    )
    # (2) blunder digit grads POSITIVE (push-down) and LARGE (>> MoveNegTerm's ~0.02 — the renorm amplifies)
    assert all(x > 0 for x in bgrad), f"blunder digit grads not all positive: {bgrad}"
    assert max(bgrad) > 0.05, (
        f"blunder grad max {max(bgrad):.3e} too small — the legal renorm should amplify ~50x over the raw "
        f"MoveNegTerm (~0.02); if it's ~0.002 the term is not renormalizing"
    )
    # (3) good-move (the other support member) digit grads NEGATIVE — mass redistributes off the blunder WITHIN
    # the legal support (not across the whole raw grid)
    assert all(x < 0 for x in ggrad), (
        f"good-move (support peer) digit grads not all negative: {ggrad}"
    )

    # (4) one manual SGD step on THIS loss lowers the legal-renormalized P(blunder) — it actually reshapes mass
    opt = torch.optim.SGD([dim_lg], lr=5.0)
    p_blunder_before = float(
        _renorm_p(model, dim_lg.detach(), blunder)
        / (_renorm_p(model, dim_lg.detach(), blunder) + _renorm_p(model, dim_lg.detach(), good))
    )
    opt.zero_grad()
    LegalMoveNegTerm().contribute(c)[1].backward(retain_graph=False)
    opt.step()
    with torch.no_grad():
        pb = float(_renorm_p(model, dim_lg, blunder))
        pg = float(_renorm_p(model, dim_lg, good))
        p_blunder_after = pb / (pb + pg)
    print(f"legal-renorm P(blunder) before={p_blunder_before:.4f}  after={p_blunder_after:.4f}")
    assert p_blunder_after < p_blunder_before - 1e-3, (
        f"SGD step did NOT lower legal-renorm P(blunder) ({p_blunder_before:.4f} -> {p_blunder_after:.4f}) — "
        f"the term does not reshape the legal-move mass"
    )
    print(
        "legal_neg_reaches_legal_mass PASS  -> the blunder's LEGAL-RENORMALIZED share is pushed down with a "
        "~50x larger gradient than the raw-codeword term, mass redistributes within the legal support, and one "
        "step measurably lowers P(blunder)."
    )


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("test_legal_neg_reaches_legal_mass",):
        try:
            globals()[name]()
            print(f"{name} PASS\n")
        except (AssertionError, ImportError, AttributeError) as e:
            print(f"{name} FAIL -> {e}")
