"""Codebook emission (`code_emit`) uses one softmax over a fixed alphabet.

It commits to the resulting MIXTURE of codes, which is what an emission that
is a SET (maze frontier: which of 256 cells) actually needs: mass has to be allocated among the members instead of
each member being scored independently.

These pin the three properties the maze arm depends on:
  * the rollout is shape-clean and the commit is the mixture, not a single winner;
  * KV-cache stays numerically identical on this path;
  * an orthonormal codebook round-trips: a membership target projects back to the uniform-over-members law, which
    is what CodeSoftmaxObjective trains toward, and the emitted mixture decodes to the right members.
"""

import os

import torch
import torch.nn.functional as F

from langset import LangSetModel

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
N_CODES = 12


def _code_model(seed: int = 0):
    m = LangSetModel.from_pretrained(
        ARCH,
        device="cpu",
        dropout=0.0,
        n_latents=1,
        multi_latent=True,
        code_emit=True,
        n_codes=N_CODES,
    )
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(m.latent_dim, N_CODES, generator=g)).Q[:, :N_CODES]
    m.head.set_code(q.t().contiguous())  # [n_codes, d] orthonormal rows
    return m


def _members(m, cells: list[list[int]]) -> torch.Tensor:
    """Membership targets: normalize(multi_hot @ code) — the same construction the maze arm's target source uses."""
    bits = torch.zeros(len(cells), N_CODES)
    for i, cs in enumerate(cells):
        bits[i, cs] = 1.0
    return F.normalize(bits @ m.head.code, dim=-1)


def test_code_emit_shapes_and_mixture_commit() -> None:
    """emit_logits returns one n_codes softmax; commit returns the softmax mixture of codes."""
    torch.manual_seed(0)
    m = _code_model()
    m.eval()
    hid = torch.randn(4, m.h)
    lg, stop = m.head.emit_logits(hid)
    assert lg.shape == (4, 1, N_CODES), f"expected [B, 1, n_codes], got {tuple(lg.shape)}"
    assert stop.shape == (4, 1)
    z = m.head.commit(lg)
    assert z.shape == (4, m.latent_dim)
    expect = F.normalize(lg.float().softmax(-1).squeeze(-2) @ m.head.code, dim=-1)
    assert torch.allclose(z, expect, atol=1e-6), "commit is not the softmax mixture of codes"
    # SUPERPOSITION: split the mass evenly over two codes and the commit must land BETWEEN them, not on either.
    # For an orthonormal codebook that midpoint sits at cos = 1/sqrt(2) from each, which is the whole property the
    # maze arm needs (a two-cell frontier is one latent holding both, not a winner).
    two = torch.full((1, 1, N_CODES), -1e9)
    two[0, 0, 2] = two[0, 0, 5] = 0.0  # softmax -> 0.5 / 0.5
    zt = m.head.commit(two)[0]
    sims = zt @ m.head.code.t()
    assert abs(sims[2].item() - 2**-0.5) < 1e-4 and abs(sims[5].item() - 2**-0.5) < 1e-4, (
        f"two-code mixture is not the midpoint: cos={sims[2].item():.4f}, {sims[5].item():.4f}"
    )
    assert sims.abs().max().item() < 0.999, "commit collapsed onto a single code"


def test_code_emit_target_law_round_trips() -> None:
    """An orthonormal codebook recovers membership: target @ code.T -> equal mass on members, ~0 elsewhere."""
    torch.manual_seed(0)
    m = _code_model()
    cells = [[0, 3, 7], [5], [1, 2, 4, 9, 11]]
    tgt = _members(m, cells)
    w = F.relu(tgt @ m.head.code.t())
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)  # the law CodeSoftmaxObjective builds
    for i, cs in enumerate(cells):
        assert torch.allclose(w[i, cs], torch.full((len(cs),), 1.0 / len(cs)), atol=1e-5), (
            f"members not uniform for row {i}: {w[i, cs]}"
        )
        off = [c for c in range(N_CODES) if c not in cs]
        assert w[i, off].abs().max().item() < 1e-5, f"non-members carry mass in row {i}"


def test_code_emit_kv_cache_matches_recompute() -> None:
    """The codebook path shares the rollout loops, so KV-cache must stay numerically identical here too."""
    torch.manual_seed(1)
    m = _code_model()
    m.eval()
    enc = m.tokenizer(
        ["seed row %d here now" % i for i in range(4)], padding=True, return_tensors="pt"
    )
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = (
        _members(m, [[0, 1], [3], [2, 5, 8], [7, 9]]).unsqueeze(1).expand(4, 5, -1).contiguous()
    )
    ss_mask = torch.ones(
        4, 5, dtype=torch.bool
    )  # self-feed: exercises commit() at each sequential hop
    with torch.no_grad():
        d0, s0, _, _ = m.rollout_train_state(
            ids, am, target, ss_prob=0.4, ss_mask=ss_mask, train_hops=3, kv_cache=False
        )
        d1, s1, _, _ = m.rollout_train_state(
            ids, am, target, ss_prob=0.4, ss_mask=ss_mask, train_hops=3, kv_cache=True
        )
    assert d0.shape == d1.shape == (4, 6, 1, N_CODES), f"unexpected logit shape {tuple(d0.shape)}"
    assert (d0 - d1).abs().max().item() < 1e-3, "codebook logits diverged under kv_cache"
    assert (s0 - s1).abs().max().item() < 1e-3, "stop logits diverged under kv_cache"


def test_code_emit_teacher_forcing_is_lossless() -> None:
    """encode() must hand back the target itself: there is no quantizer, so teacher forcing feeds it exactly."""
    torch.manual_seed(0)
    m = _code_model()
    tgt = _members(m, [[0, 3], [4, 6, 8]])
    idx, recon = m.head.encode(tgt)
    assert idx.shape == (2, 1) and recon.shape == tgt.shape
    assert torch.allclose(recon, tgt, atol=1e-6), (
        "codebook encode must be lossless (recon == target)"
    )


def test_code_emit_termination_is_independent_of_set_width() -> None:
    """Termination must NOT get easier as the emitted set widens.

    Folding STOP into the membership softmax couples the two through one normalizer: at a member position the
    target is 0 on STOP, so the gradient pushing the stop logit DOWN is proportional to P(STOP), and P(STOP)
    shrinks as the set widens (the denominator carries k members). Wider sets therefore suppress the stop logit
    more weakly for the same logits, letting it ratchet up. Factored out as its own sigmoid, the terminator is
    trained by its own signal and the decision is a function of the stop logit alone.

    (An earlier version of this test asserted the coupling shows up in the ARGMAX. It does not -- argmax
    compares logits directly, so it is k-independent. The coupling is in the gradient, via P(STOP).)"""
    torch.manual_seed(0)
    stop_logit = torch.full((3, 1), -2.5)
    p_stop_folded = []
    for k in (1, 2, 4, 8, 12):  # narrow -> wide sets, uniform over k members
        lg = torch.full((3, 1, N_CODES), -1e9)
        lg[:, 0, :k] = 0.0
        assert abs(lg.float().softmax(-1)[0, 0, 0].item() - 1 / k) < 1e-5, (
            "setup: members not uniform"
        )
        # FACTORED: the terminator sees only its own logit — same decision at every width
        assert bool((torch.sigmoid(stop_logit) < 0.5).all()), (
            f"terminated at k={k} on a keep-going logit"
        )
        # FOLDED counterfactual: P(STOP) under one shared normalizer, which is the gradient's magnitude
        folded = torch.cat([lg[:, 0, :], stop_logit], -1)
        p_stop_folded.append(folded.float().softmax(-1)[0, -1].item())
    assert p_stop_folded == sorted(p_stop_folded, reverse=True), (
        f"P(STOP) should fall monotonically as the set widens: {p_stop_folded}"
    )
    assert p_stop_folded[0] > 3 * p_stop_folded[-1], (
        f"widening 1->12 members should weaken STOP's suppression gradient substantially: {p_stop_folded}"
    )


def test_code_emit_recon_carries_emission_gradient() -> None:
    """`recon` must be the EMISSION with gradient, not the target.

    Every auxiliary loss term builds its query from `c.recon` (MultiNCETerm: "emitted, gradient flows here";
    the auxiliary contrastive terms). The codebook path is lossless, so the rollout's own recon IS the
    stop-grad target -- handing that back makes in-batch InfoNCE compare each target against itself: near-zero
    loss, zero gradient, silently inert. Caught in a live run as a `loss - loss_code` gap pinned at 0.145 for
    five straight epochs while loss_code fell 5.17 -> 3.19."""
    from langset import TrainingArguments
    from langset.strategies import CodeSoftmaxObjective

    torch.manual_seed(0)
    m = _code_model()
    cells = [[0, 1, 2], [4, 5]]
    b, lmax = len(cells), 3
    target_lat = _members(m, cells).unsqueeze(1).expand(b, lmax, -1).contiguous()
    enc = m.tokenizer(["row %d seed" % i for i in range(b)], padding=True, return_tensors="pt")
    se = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    valid = torch.ones(b, lmax, dtype=torch.bool)

    obj = CodeSoftmaxObjective(
        m,
        TrainingArguments(emission=CodeSoftmaxObjective, ss_prob=0.0),
        torch.device("cpu"),
        trainer=None,  # type: ignore[arg-type]
    )
    em = obj.emit(se, target_lat, valid, [lmax] * b, list(range(b)), b, lmax, ep=0)
    assert "loss_stop" in em.logs, (
        "termination must be its own reported term, not folded into the members"
    )
    assert em.recon.requires_grad, (
        "recon has no gradient -- every aux loss term would be silently inert"
    )
    assert em.recon.shape == target_lat.shape
    assert not torch.allclose(em.recon, target_lat, atol=1e-4), (
        "recon is the TARGET, not the emission"
    )

    # and the gradient must actually reach the query projection (the only learned map on this path)
    m.zero_grad()
    em.recon.sum().backward()
    qp = m.head.query_proj
    assert qp is not None and qp.weight.grad is not None and qp.weight.grad.abs().sum() > 0, (
        "gradient from recon does not reach query_proj"
    )


def test_code_softmax_objective_learns_the_member_law() -> None:
    """End-to-end: optimizing CodeSoftmaxObjective drives the emitted mixture onto the target's member law.

    Four rows, each a fixed member set, repeated for every tick — a memorization task the tiny backbone can fit.
    What is being pinned is that the loss is wired to something learnable: gradient reaches the query projection
    through the softmax and the mixture moves toward the target, so a real run's failure would be capability, not
    plumbing."""
    from langset import TrainingArguments
    from langset.strategies import CodeSoftmaxObjective

    torch.manual_seed(0)
    m = _code_model()
    cells = [[0, 1, 2], [4, 5], [7, 8, 9], [3, 6]]
    b, lmax = len(cells), 3
    target_lat = _members(m, cells).unsqueeze(1).expand(b, lmax, -1).contiguous()
    enc = m.tokenizer(["row %d seed text" % i for i in range(b)], padding=True, return_tensors="pt")
    se = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    valid = torch.ones(b, lmax, dtype=torch.bool)

    args = TrainingArguments(emission=CodeSoftmaxObjective, ss_prob=0.0)
    obj = CodeSoftmaxObjective(m, args, torch.device("cpu"), trainer=None)  # type: ignore[arg-type]
    assert obj.n_codes == N_CODES

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    first = last = None
    for _ in range(40):
        em = obj.emit(se, target_lat, valid, [lmax] * b, list(range(b)), b, lmax, ep=0)
        opt.zero_grad()
        em.base_loss.backward()
        opt.step()
        cos = float(em.logs["emit_cos"])
        first = (float(em.base_loss), cos) if first is None else first
        last = (float(em.base_loss), cos)
    assert last[0] < first[0] - 0.05, f"loss did not fall: {first[0]:.4f} -> {last[0]:.4f}"
    assert last[1] > first[1], (
        f"emitted mixture moved away from target: {first[1]:.4f} -> {last[1]:.4f}"
    )
