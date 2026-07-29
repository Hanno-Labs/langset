"""CONCEPTS — the text-in format for a named, superposed state.

A row says what is true of it, in words:

    {"input_text": "PUP — Morbid Stuff (2019, Rise, Rock)",
     "target_text": "<the review prose>",
     "concepts": {"vocals": ["yell-singing", "throat-shredding"], "tempo": {"7": 0.1, "8": 0.9}}}

No index is ever authored. langset scans the column, discovers each facet's alphabet, and turns the names into
vectors via `code_source`. A list means "all true, equally" — the superposition. A dict gives explicit weights,
which is also how a continuous value rides on the same machinery: 7.9 is 0.1 of "7" plus 0.9 of "8".
"""

import os

import torch

from langset import LangSetModel
from langset.strategies import discover_concept_alphabet, parse_concepts

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")

PUP = {
    "vocals": ["yell-singing", "throat-shredding", "gang-vocals"],
    "mood": ["angry-but-vulnerable", "self-aware"],
    "tempo": {"7": 0.1, "8": 0.9},
}
OTHER = {
    "vocals": ["clean-sung"],
    "mood": ["wistful"],
    "tempo": {"3": 1.0},
}


def test_list_means_equal_mass_dict_means_weights() -> None:
    """A list is the superposition (mass split evenly); a dict states the weights, so a continuous value can be
    carried as a mixture over adjacent concepts instead of a regressed scalar."""
    law = parse_concepts(PUP)
    assert law["vocals"] == {"yell-singing": 1 / 3, "throat-shredding": 1 / 3, "gang-vocals": 1 / 3}
    assert law["mood"] == {"angry-but-vulnerable": 0.5, "self-aware": 0.5}
    assert abs(law["tempo"]["8"] - 0.9) < 1e-9 and abs(law["tempo"]["7"] - 0.1) < 1e-9
    for facet in law.values():  # every facet is its own distribution
        assert abs(sum(facet.values()) - 1.0) < 1e-9


def test_alphabet_is_discovered_not_configured() -> None:
    """The vocabulary comes from the corpus, sorted for determinism — the way a tokenizer's does."""
    alpha = discover_concept_alphabet([PUP, OTHER])
    assert alpha["vocals"] == ["clean-sung", "gang-vocals", "throat-shredding", "yell-singing"]
    assert alpha["mood"] == ["angry-but-vulnerable", "self-aware", "wistful"]
    assert alpha["tempo"] == ["3", "7", "8"]
    assert list(alpha) == ["mood", "tempo", "vocals"]  # facets sorted too, so layout is stable


def test_multi_latent_rows_carry_one_dict_per_tick() -> None:
    """A world-model row passes a LIST of concept dicts aligned with target_texts."""
    ticks = [{"stage": ["ph2"], "result": ["ongoing"]}, {"stage": ["ph3"], "result": ["missed"]}]
    alpha = discover_concept_alphabet([ticks])
    assert alpha == {"result": ["missed", "ongoing"], "stage": ["ph2", "ph3"]}


def test_facets_get_separate_slices_and_separate_softmaxes() -> None:
    """Each facet owns its own dims and normalizes within itself, so a wide `vocals` mixture does not make
    `tempo` look uncertain. That separation is the whole reason facets exist rather than one flat bag."""
    alpha = discover_concept_alphabet([PUP, OTHER])
    total = sum(len(v) for v in alpha.values())
    m = LangSetModel.from_pretrained(
        ARCH,
        device="cpu",
        dropout=0.0,
        n_latents=1,
        multi_latent=True,
        code_emit=True,
        n_codes=total,
        res_dim=4,
    )
    share = m.head.state_dim // len(alpha)
    facets = []
    for seed, f in enumerate(alpha):
        g = torch.Generator().manual_seed(seed)
        facets.append((f, torch.randn(len(alpha[f]), share, generator=g), share))
    m.head.set_concepts(facets)
    assert m.head.concept_names == list(alpha)
    assert len(m.head.concept_spans) == 3

    lg = torch.randn(2, 1, total)
    p = m.head.concept_probs(lg)
    for m_lo, m_hi, _, _ in m.head.concept_spans:  # each facet is its own distribution
        assert torch.allclose(p[..., m_lo:m_hi].sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(p.sum(-1), torch.full((2,), 3.0), atol=1e-4), (
        "facets should not share one softmax"
    )

    # a facet's codes live ONLY in its own dims — that is what keeps the readouts independent
    for fi, (m_lo, m_hi, d_lo, d_hi) in enumerate(m.head.concept_spans):
        block = m.head.code[m_lo:m_hi]
        assert block[:, d_lo:d_hi].abs().sum() > 0, "facet has no code in its own dims"
        outside = block.abs().sum() - block[:, d_lo:d_hi].abs().sum()
        assert float(outside) < 1e-6, "facet's codes leak into another facet's dims"


def test_silence_about_a_facet_is_not_evidence() -> None:
    """A row that names only `vocals` should train only `vocals`. Saying nothing about `mood` is not the same
    as saying the mood is uniform, and treating it that way would teach the model to hedge."""
    law = parse_concepts({"vocals": ["yell-singing"]})
    assert set(law) == {"vocals"}, "an unstated facet must not appear in the target law"
