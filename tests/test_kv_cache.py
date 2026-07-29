"""KV-cache rollout exactness (multi-latent).

The KV-cache path (`rollout_train_codebook(..., kv_cache=True)`) forwards the prompt ONCE and then each latent
token alone against the cached prefix K/V, instead of re-running the full growing sequence every tick. Its whole
justification is that it is NUMERICALLY IDENTICAL to the recompute path — same emit logits, same feeds — while
holding ~1 prompt forward + n single tokens of activations instead of n full-prefix forwards (killing the
O(ticks) memory blowup that forces grad_ckpt). These tests pin that identity.

Given the SAME shared ss_mask, cached and non-cached must produce the same dim/stop logits:
  * teacher-forced feeds (ss_mask all-False): tight, no argmax cascade — pure forward-equivalence.
  * self-feed (ss_mask all-True): the model consumes its own argmax'd emission; still identical as long as the
    ~1e-5 hidden match doesn't flip a discrete argmax (it doesn't on well-separated logits).
"""

import os

import torch

from langset import LangSetModel

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


def _multi_model():
    return LangSetModel.from_pretrained(
        ARCH, device="cpu", dropout=0.0, n_latents=1, multi_latent=True, fsq_dim=8, fsq_levels=4
    )


def _run(m, ids, am, target, ss_mask, kv_cache, train_hops=None):
    return m.rollout_train_codebook(
        ids, am, target, ss_prob=0.4, ss_mask=ss_mask, kv_cache=kv_cache, train_hops=train_hops
    )


def test_kv_cache_matches_teacher_forced() -> None:
    """ss_mask all-False -> every feed is the TRUE latent (no dependence on the model's own argmax), so cached
    vs recompute differ only by forward numerics. Emit logits must match to forward precision."""
    torch.manual_seed(0)
    m = _multi_model()
    m.eval()
    tok = m.tokenizer
    B, L, d = 5, 6, m.latent_dim
    enc = tok(["seed row %d here now" % i for i in range(B)], padding=True, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = torch.randn(B, L, d)
    ss_mask = torch.zeros(B, L, dtype=torch.bool)  # all teacher-forced

    with torch.no_grad():
        d0, s0, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=False)
        d1, s1, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=True)
    dd = (d0 - d1).abs().max().item()
    ds = (s0 - s1).abs().max().item()
    assert d0.shape == d1.shape and s0.shape == s1.shape, "cached rollout changed output shape"
    assert dd < 1e-3, f"dim logits diverged: {dd:.3e}"
    assert ds < 1e-3, f"stop logits diverged: {ds:.3e}"


def test_kv_cache_matches_selffeed() -> None:
    """ss_mask all-True -> the model feeds its OWN emitted latent each tick. Still identical to the recompute
    path as long as the tiny hidden match doesn't flip an argmax."""
    torch.manual_seed(1)
    m = _multi_model()
    m.eval()
    tok = m.tokenizer
    B, L, d = 5, 6, m.latent_dim
    enc = tok(["another seed %d text" % i for i in range(B)], padding=True, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = torch.randn(B, L, d)
    ss_mask = torch.ones(B, L, dtype=torch.bool)  # all self-feed

    with torch.no_grad():
        d0, s0, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=False)
        d1, s1, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=True)
    dd = (d0 - d1).abs().max().item()
    assert dd < 1e-2, f"self-feed dim logits diverged: {dd:.3e} (argmax cascade?)"


def test_kv_cache_matches_train_hops() -> None:
    """Mixed regime: train_hops splits self-feed (t<H) from teacher-forced (t>=H). Cached must match with a
    shared mask across BOTH regions."""
    torch.manual_seed(2)
    m = _multi_model()
    m.eval()
    tok = m.tokenizer
    B, L, d = 4, 8, m.latent_dim
    enc = tok(["hop seed %d row" % i for i in range(B)], padding=True, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = torch.randn(B, L, d)
    ss_mask = torch.rand(B, L) < 0.5

    with torch.no_grad():
        d0, s0, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=False, train_hops=3)
        d1, s1, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=True, train_hops=3)
    assert (d0 - d1).abs().max().item() < 1e-2
    assert (s0 - s1).abs().max().item() < 1e-2


def test_kv_cache_grad_flows() -> None:
    """The cache is not detached: a loss on the cached rollout's logits must backprop to the LoRA params."""
    torch.manual_seed(3)
    m = _multi_model()
    m.train()
    tok = m.tokenizer
    B, L, d = 3, 5, m.latent_dim
    enc = tok(["grad seed %d" % i for i in range(B)], padding=True, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = torch.randn(B, L, d)
    ss_mask = torch.rand(B, L) < 0.5
    d1, s1, _, _ = _run(m, ids, am, target, ss_mask, kv_cache=True)
    d1.pow(2).mean().backward()
    trainable = [p for p in m.parameters() if p.requires_grad]
    got = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable)
    assert got, "no gradient reached the trainable params through the KV cache"
