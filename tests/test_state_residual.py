"""STATE + RESIDUAL emission — the named half and the unnamed half.

This mode CONSTITUTES the latent: its leading dims
are a mixture over a named alphabet, its trailing `res_dim` dims are a residual nothing names.

The split is what makes adopting a named alphabet safe. An alphabet is a ceiling — whatever it cannot name, the
emission cannot carry — so the residual is where unnamed information can go.
"""

import os

import torch
import torch.nn.functional as F

from langset import LangSetModel

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
N_CODES, RES = 12, 4


def _model(res_dim: int = RES):
    m = LangSetModel.from_pretrained(
        ARCH,
        device="cpu",
        dropout=0.0,
        n_latents=1,
        multi_latent=True,
        code_emit=True,
        n_codes=N_CODES,
        res_dim=res_dim,
    )
    g = torch.Generator().manual_seed(0)
    q = torch.linalg.qr(torch.randn(m.head.state_dim, N_CODES, generator=g)).Q[:, :N_CODES]
    m.head.set_code(q.t().contiguous())
    return m


def test_split_widths_and_unit_norm() -> None:
    """The emission is exactly [state | residual] and stays unit-norm, so neither half wins on scale alone."""
    m = _model()
    assert m.head.state_dim + m.head.res_dim == m.latent_dim
    assert m.head.code.shape == (N_CODES, m.head.state_dim)
    hid = torch.randn(5, m.h)
    lg, _ = m.head.emit_logits(hid)
    z = m.head.commit(lg, hid=hid)
    assert z.shape == (5, m.latent_dim)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-4), "emission is not unit-norm"
    # each half carries half the squared norm — the residual cannot drown the state or vice versa
    assert torch.allclose(
        z[:, : m.head.state_dim].norm(dim=-1) ** 2, torch.full((5,), 0.5), atol=1e-4
    )
    assert torch.allclose(
        z[:, m.head.state_dim :].norm(dim=-1) ** 2, torch.full((5,), 0.5), atol=1e-4
    )


def test_state_half_is_readable_without_a_probe() -> None:
    """The named half decodes by plain matmul: put mass on members 2 and 5, read 2 and 5 back out.

    This is the property a decode head cannot supply. An aux head reads whatever the latent happens to contain;
    here the members ARE what the latent is made of, so the readout is exact by construction."""
    m = _model()
    lg = torch.full((1, 1, N_CODES), -1e9)
    lg[0, 0, 2] = lg[0, 0, 5] = 0.0  # half the mass on each of two members
    z = m.head.commit(lg, hid=torch.zeros(1, m.h))
    scores = z[:, : m.head.state_dim] @ m.head.code.t()  # no probe, no learned decoder
    top2 = set(scores[0].topk(2).indices.tolist())
    assert top2 == {2, 5}, f"named half did not decode to its own members: {top2}"


def test_res_dim_zero_is_pure_state() -> None:
    """res_dim=0 leaves the emission byte-identical to the plain codebook path (no residual head built)."""
    m = _model(res_dim=0)
    assert m.head.res_proj is None and m.head.state_dim == m.latent_dim
    hid = torch.randn(3, m.h)
    lg, _ = m.head.emit_logits(hid)
    z = m.head.commit(lg, hid=hid)
    expect = F.normalize(lg.float().softmax(-1).squeeze(-2) @ m.head.code, dim=-1)
    assert torch.allclose(z, expect, atol=1e-6)


def test_rollout_returns_emit_hidden_only_when_asked() -> None:
    """The extra return is opt-in, so every existing caller that unpacks four values is untouched."""
    m = _model()
    m.eval()
    enc = m.tokenizer(["row %d text" % i for i in range(3)], padding=True, return_tensors="pt")
    target = F.normalize(torch.randn(3, 4, m.latent_dim), dim=-1)
    with torch.no_grad():
        four = m.rollout_train_state(enc["input_ids"], enc["attention_mask"], target)
        five = m.rollout_train_state(
            enc["input_ids"], enc["attention_mask"], target, return_emit_hidden=True
        )
    assert len(four) == 4 and len(five) == 5
    assert five[4].shape == (3, 5, m.h), (
        f"emit hidden should be [B, L+1, h], got {tuple(five[4].shape)}"
    )
    for a, b in zip(four, five[:4]):
        assert torch.allclose(a, b), "asking for the hidden changed the rollout's other outputs"


def test_residual_carries_what_the_alphabet_cannot() -> None:
    """Two ticks whose NAMED content is identical must still be distinguishable through the residual.

    This is the ceiling argument made concrete: if the alphabet is the whole latent, two states that share
    members are indistinguishable no matter what else differs about them."""
    m = _model()
    lg = torch.full((2, 1, N_CODES), -1e9)
    lg[:, 0, 3] = 0.0  # identical named content on both rows
    hid = torch.randn(2, m.h) * 5.0  # but different context
    z = m.head.commit(lg, hid=hid)
    state_a, state_b = z[0, : m.head.state_dim], z[1, : m.head.state_dim]
    assert torch.allclose(state_a, state_b, atol=1e-5), "named halves should match — same members"
    res_a, res_b = z[0, m.head.state_dim :], z[1, m.head.state_dim :]
    assert not torch.allclose(res_a, res_b, atol=1e-3), (
        "residual collapsed; the unnamed half carries nothing"
    )
