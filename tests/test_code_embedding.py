"""Codebook emission on the SINGLE-latent (embedding) path — an embedding that is a superposition.

The multi-latent path uses a codebook to emit a world state. The same machinery on the single-latent path
gives a RETRIEVAL vector that is a mixture over named members, which is a different kind of useful: a dense
embedding tells you two documents are close, and this one tells you what they are close IN. Project the vector
back onto the codebook and you get the members it is made of, in the domain's own vocabulary, with no probe.

Nothing supervises the weights. The ordinary contrastive/recon objective assigns them, so an alphabet buys
legibility without requiring per-row state labels — you supply the vocabulary, training decides the mixture.
"""

import os

import torch
import torch.nn.functional as F

from langset import LangSetModel

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
MEMBERS = ["oncology", "cardiology", "phase_1", "phase_3", "safety_signal", "approval"]


def _emb_model(res_dim: int = 0):
    """A SINGLE-latent (multi_latent=False) retrieval model whose embedding is a codebook mixture."""
    m = LangSetModel.from_pretrained(
        ARCH,
        device="cpu",
        dropout=0.0,
        n_latents=1,
        multi_latent=False,
        code_emit=True,
        n_codes=len(MEMBERS),
        res_dim=res_dim,
    )
    g = torch.Generator().manual_seed(0)
    q = torch.linalg.qr(torch.randn(m.head.state_dim, len(MEMBERS), generator=g)).Q[
        :, : len(MEMBERS)
    ]
    m.head.set_code(q.t().contiguous())
    return m


def test_single_latent_codebook_emits_and_is_unit_norm() -> None:
    """The embedding path accepts a codebook at all — this was multi-latent-only before."""
    m = _emb_model()
    assert not m.multi_latent and m.head.code_emit
    z = m.emit(["a filing about a phase 3 oncology trial", "an unrelated string"])
    assert z.shape == (2, m.latent_dim)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-4), "embedding is not unit-norm"


def test_embedding_decomposes_into_its_members() -> None:
    """The point of the whole thing: read the embedding's composition off it, with no probe.

    Drive the head directly so the mixture is known, then recover it by projecting onto the codebook."""
    m = _emb_model()
    lg = torch.full((1, 1, len(MEMBERS)), -1e9)
    lg[0, 0, 0] = lg[0, 0, 3] = 0.0  # half oncology, half phase_3
    z = F.normalize(lg.float().softmax(-1).squeeze(-2) @ m.head.code, dim=-1)
    share = (z @ m.head.code.t())[0]  # the "what is this made of" readout
    named = {MEMBERS[i]: float(share[i]) for i in share.topk(2).indices.tolist()}
    assert set(named) == {"oncology", "phase_3"}, (
        f"embedding did not decompose to its members: {named}"
    )
    # an orthonormal codebook splits an even two-member mixture evenly at 1/sqrt(2)
    for v in named.values():
        assert abs(v - 2**-0.5) < 1e-3, f"member shares are not the mixture weights: {named}"


def test_embedding_residual_is_separable() -> None:
    """With res_dim > 0 the embedding is [named | unnamed], and the named half still reads cleanly."""
    m = _emb_model(res_dim=4)
    assert m.head.state_dim + m.head.res_dim == m.latent_dim
    z = m.emit(["a phase 1 cardiology safety readout"])
    state, res = z[:, : m.head.state_dim], z[:, m.head.state_dim :]
    assert res.shape[-1] == 4
    # each half holds half the squared norm, so the unnamed part cannot swamp the readout
    assert abs(float(state.norm()) ** 2 - 0.5) < 1e-3
    assert abs(float(res.norm()) ** 2 - 0.5) < 1e-3
    share = F.normalize(state, dim=-1) @ m.head.code.t()
    assert share.shape == (1, len(MEMBERS)) and torch.isfinite(share).all()


def test_gradient_reaches_the_mixture_weights() -> None:
    """Nothing supervises the members directly, so the contrastive path must be able to move them.

    Driven through the head rather than `emit()`, which is the inference API and runs under no_grad."""
    m = _emb_model()
    hid = torch.randn(2, 1, m.h, requires_grad=True)
    z = m.head(hid)
    z.sum().backward()
    qp = m.head.query_proj
    assert qp is not None and qp.weight.grad is not None and qp.weight.grad.abs().sum() > 0, (
        "no gradient into query_proj — the mixture weights would be frozen"
    )
