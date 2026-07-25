"""GOLDEN test (test-first): a LEGAL-MOVE-RENORMALIZED POSITIVE term (SupportPosTerm) must RAISE the best move's
share of the legal-move probability mass — the flip side of SupportNegTerm. RED until the term exists, GREEN
once it does. This is the missing positive loss: LabelDimsTerm trains per-digit MARGINALS only, but the decode
reads the JOINT-RENORMALIZED P(best)/Σ_legal P(legal) — which is never directly trained, so SF-best sits at
rank 17-37 with <1% mass on the trained models (the "barely considers best" failure, validated on 4 boards).

SupportPosTerm = cross-entropy toward best over the legal-move renormalized distribution:
  loss = -log( P(best) / Σ_legal P(legal) )
Same machinery as SupportNegTerm (legal-move support set + the precomputed codeword index tensor), opposite
sign. Grad flows into dim_lg / level_proj; mechanistically it concentrates on the WEAKER marginal (the
to-square, where TO-acc < FROM-acc) because the joint product is bottlenecked by the smaller factor.

This test asserts the precise mechanism:
  - one SGD step on the loss RAISES the legal-renormalized P(best) (it actually promotes best, not just
    suppresses alternatives);
  - the gradient reaches dim_lg (level_proj), specifically on the reserved digits (not non-reserved/STOP);
  - and the gradient concentrates on the weaker marginal: if P(from) is near-correct (solved) but P(to) is
    diffuse (unsolved), the to-digit gradient magnitude exceeds the from-digit gradient magnitude. This is the
    mechanistic reason the term targets the "right piece, wrong destination" failure.

Run:  .venv/bin/python tests/test_supportpos_raises_best.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_trainer_multi_characterization as M  # noqa: E402

from langset.strategies import MultiStepCtx, SupportPosTerm  # noqa: E402


class _FakeTrainer:
    """Fields SupportPosTerm reads: support_idx/support_n (the legal-move set, precomputed) + label_plan.
    The 'best' move is support_codes[0] in each row (by convention the positive label)."""

    def __init__(self, support_codes, label_plan) -> None:
        self.label_plan = label_plan
        self.hard_neg_texts = None
        self.label_neg_idx = None  # SupportPosTerm must not need the neg subset
        self.label_neg_n = None
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

        self.support_idx, self.support_n = _pad(support_codes)


def _build_ctx(model, dev, support_codes):
    """Reserved layout mirrors chess label_dims={from_sq:[1,2], to_sq:[3,4]} (4 reserved digits at rest cols
    0,1,2,3 = full dims 1,2,3,4). A 'move' codeword = 4 digit indices. support_codes[0] is the best move."""
    head = model.head
    B, L, h = 1, 1, model.h
    V = int(head.fsq_levels)
    torch.manual_seed(0)
    hid = torch.randn(B, L + 1, h, device=dev, requires_grad=True)
    dim_lg, _stop = head.emit_logits(hid)
    dim_lg.retain_grad()
    label_plan = [(0, "from_sq", 0), (1, "from_sq", 1), (2, "to_sq", 0), (3, "to_sq", 1)]
    valid = torch.ones(B, L, dtype=torch.bool, device=dev)
    args = M._args("/tmp/_sp")
    args.lam_support_pos = 1.0
    c = MultiStepCtx(
        trainer=_FakeTrainer([support_codes], label_plan),
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
    """raw P(code) = Π over reserved digits of softmax(dim_lg[digit])[level] — used to verify the SGD step
    raises best's LEGAL-RENORMALIZED share."""
    dl = dim_lg[0, 0, 1:, :]  # [fsq_dim, V]
    p = 1.0
    for d, lvl in enumerate(code):
        p = p * torch.softmax(dl[d].float(), -1)[lvl]
    return p


def test_support_pos_raises_best_share() -> None:
    """RED until SupportPosTerm exists and raises best's LEGAL-RENORMALIZED share. Two legal moves
    (support = {best, other}); loss = -log(P(best)/(P(best)+P(other))). Asserts: grad reaches dim_lg
    (level_proj), specifically the reserved digits; one SGD step raises legal-renorm P(best); and the gradient
    concentrates on the WEAKER marginal (to_sq) when from_sq is near-solved but to_sq is diffuse."""
    torch.manual_seed(0)
    model = M._build_model()
    dev = torch.device("cpu")
    best = (1, 2, 3, 4)
    other = (5, 6, 7, 0)
    support = [best, other]
    c, dim_lg = _build_ctx(model, dev, support)

    contrib = SupportPosTerm().contribute(c)
    assert contrib is not None, (
        "SupportPosTerm returned None (lam_support_pos<=0 / no support_idx / no dim_lg)"
    )
    _k, loss_sp, _w = contrib
    assert torch.isfinite(loss_sp), f"support-pos loss not finite: {loss_sp}"

    # (1) loss is the legal-renorm CE: -log(P(best)/(P(best)+P(other))) in [0, ~log2]; NOT a raw-codeword ~0.002
    loss_val = float(loss_sp)
    assert 0.0 < loss_val < 5.0, (
        f"loss {loss_val:.4f} not in the legal-renorm CE range [0, 5] — if it's ~0.002 the term is the raw "
        f"LabelNegTerm (no renormalization); if it's huge the renorm denominator is wrong"
    )

    for p in model.parameters():
        p.grad = None
    loss_sp.backward(
        retain_graph=True
    )  # retain: the SGD step below backward()s the same graph again
    g = dim_lg.grad
    assert g is not None, "dim_lg retained no grad (apparatus broken)"

    # (2) gradient reaches the reserved digits (the move), not non-reserved/STOP
    bgrad = [float(g[0, 0, d + 1, best[d]]) for d in range(4)]  # best's 4 reserved-digit levels
    assert all(x != 0 for x in bgrad), (
        f"best's reserved-digit grads all zero: {bgrad} — grad doesn't reach dim_lg"
    )
    assert float(g[0, 0, 5, :].abs().sum()) == 0.0, (
        "support-pos leaked grad onto a non-reserved digit"
    )
    assert float(g[0, 1, :, :].abs().sum()) == 0.0, "support-pos leaked grad onto the STOP position"

    # (3) one SGD step on THIS loss RAISES the legal-renormalized P(best) (it promotes best, not just suppresses)
    opt = torch.optim.SGD([dim_lg], lr=5.0)
    p_best_before = float(_renorm_p(model, dim_lg.detach(), best))
    p_other_before = float(_renorm_p(model, dim_lg.detach(), other))
    share_before = p_best_before / (p_best_before + p_other_before)
    opt.zero_grad()
    SupportPosTerm().contribute(c)[1].backward()
    opt.step()
    with torch.no_grad():
        pb = float(_renorm_p(model, dim_lg, best))
        po = float(_renorm_p(model, dim_lg, other))
        share_after = pb / (pb + po)
    assert share_after > share_before + 1e-3, (
        f"SGD step did NOT raise legal-renorm P(best) ({share_before:.4f} -> {share_after:.4f}) — the term "
        f"does not promote the best move's share"
    )

    # (4) gradient concentrates on the WEAKER marginal: construct a case where from is partially-solved (best's
    # from digit has MOST but not all of the mass) while to is diffuse. Then |grad on to digits| > |grad on from|.
    # (A near-saturated from-marginal zeroes BOTH gradients — the renorm CE vanishes when share->1 — so use a
    # moderate peak that leaves real gradient on both halves.)
    torch.manual_seed(1)
    model2 = M._build_model()
    c2, dim_lg2 = _build_ctx(model2, dev, support)
    with torch.no_grad():
        # from-digits (rest cols 0,1 = full dims 1,2): best's from levels get a strong boost (mostly solved)
        dim_lg2[0, 0, 1, :] = -2.0
        dim_lg2[0, 0, 1, best[0]] = 5.0
        dim_lg2[0, 0, 2, :] = -2.0
        dim_lg2[0, 0, 2, best[1]] = 5.0
        # to-digits (rest cols 2,3 = full dims 3,4): leave diffuse (uniform ~0)
    for p in model2.parameters():
        p.grad = None
    SupportPosTerm().contribute(c2)[1].backward()
    g2 = dim_lg2.grad
    from_grad = float(g2[0, 0, 1, best[0]].abs() + g2[0, 0, 2, best[1]].abs())
    to_grad = float(g2[0, 0, 3, best[2]].abs() + g2[0, 0, 4, best[3]].abs())
    print(
        f"support_pos: loss={loss_val:.4f}  share {share_before:.4f}->{share_after:.4f}  "
        f"from_grad={from_grad:.3e} to_grad={to_grad:.3e} (weaker-marginal concentration: to>from)"
    )
    assert to_grad > from_grad, (
        f"gradient did NOT concentrate on the weaker (to) marginal: from_grad={from_grad:.3e} "
        f">= to_grad={to_grad:.3e}. The mechanistic claim (joint loss bottlenecks on the unsolved half) failed — "
        f"either the from-marginal isn't actually near-solved in this construction, or the term doesn't couple them."
    )
    print(
        "support_pos_raises_best_share PASS  -> the best move's LEGAL-RENORMALIZED share is raised, grad reaches "
        "the reserved digits, and it concentrates on the weaker (to) marginal — the 'right piece, wrong "
        "destination' failure is the mechanistic target."
    )


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    for name in ("test_support_pos_raises_best_share",):
        try:
            globals()[name]()
            print(f"{name} PASS\n")
        except (AssertionError, ImportError, AttributeError) as e:
            print(f"{name} FAIL -> {e}")
