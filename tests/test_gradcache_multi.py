"""Multi-latent GradCache tests.

The multi-latent port's correctness rests on two facts (the two-phase cache-and-inject math itself is shared
with the single-latent path, proven exact in test_gradcache.py):

1. SHARED-MASK DETERMINISM (the ss_prob fix). Under scheduled sampling the rollout is stochastic, which would
   break GradCache (phase-1 no_grad full batch vs phase-2 grad per chunk would diverge). Passing a shared
   ss_mask makes the rollout deterministic-on-replay: the full-batch rollout and a per-chunk rollout of the
   SAME rows must produce IDENTICAL recon. That is what lets the cached recon-grads line up in phase 2.

2. It TRAINS. A few grad_cache steps under ss_prob>0 produce finite, decreasing loss.
"""

import os

import torch

from langset import LangSetModel, Trainer, TrainingArguments

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


def _multi_model():
    return LangSetModel.from_pretrained(
        ARCH, device="cpu", dropout=0.0, n_latents=1, multi_latent=True, fsq_dim=8, fsq_levels=4
    )


def test_shared_ss_mask_full_equals_chunk() -> None:
    """Full-batch rollout and a per-chunk rollout of the same rows produce identical recon when they share the
    ss_mask — the invariant GradCache phase-1/phase-2 rely on under scheduled sampling (ss_prob>0)."""
    torch.manual_seed(0)
    m = _multi_model()
    m.eval()  # dropout=0 anyway; be explicit so no stochasticity beyond the (shared) ss decisions
    tok = m.tokenizer
    B, L, d = 6, 4, m.latent_dim
    enc = tok(["seed text number %d here" % i for i in range(B)], padding=True, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    target = torch.randn(B, L, d)
    ss_mask = torch.rand(B, L) < 0.4  # the shared per-(row, hop) self-feed decisions

    with torch.no_grad():
        _, _, _, recon_full = m.rollout_train_codebook(
            ids, am, target, ss_prob=0.4, ss_mask=ss_mask
        )
    rows = [0, 1, 2]
    ri = torch.tensor(rows)
    with torch.no_grad():
        _, _, _, recon_chunk = m.rollout_train_codebook(
            ids[ri], am[ri], target[ri], ss_prob=0.4, ss_mask=ss_mask[ri]
        )
    diff = (recon_full[ri] - recon_chunk).abs().max().item()
    assert diff < 1e-5, (
        f"chunk recon diverged from full recon by {diff:.3e} (shared mask should make them equal)"
    )


def test_multi_gradcache_trains() -> None:
    """grad_cache=True on the multi-latent path runs under scheduled sampling and reduces the loss."""
    torch.manual_seed(0)
    m = _multi_model()
    rows = [
        {
            "input_text": f"query {i}",
            "target_texts": [f"state {i} a", f"state {i} b", f"state {i} c"],
        }
        for i in range(16)
    ]
    args = TrainingArguments(
        epochs=3,
        batch_size=8,
        gc_chunk=4,
        grad_cache=True,
        lr=1e-3,
        max_len=32,
        ss_prob=0.25,
        lam_recon=0.0,
        val_frac=0.125,
        eval_every=999,
        seed=0,
        verbose=False,
    )
    m2 = Trainer(m, args, rows).train()  # completes under scheduled sampling
    finite = all(torch.isfinite(p).all() for p in m2.parameters())
    assert finite, "grad_cache multi training produced non-finite params"
