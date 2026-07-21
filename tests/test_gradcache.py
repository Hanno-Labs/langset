"""GradCache exactness (single-latent): one step with grad_cache=True (gc_chunk < batch_size) must produce the
SAME parameter update as the direct full-batch step. That EXACTNESS is GradCache's whole guarantee (Gao et al.
2021) — it only reduces peak activation memory, never changes the gradient.

Two identically-initialized tiny models (seed the RNG before each build so LoRA init + data order match) run
exactly one step — one direct, one GradCache with gc_chunk=2 on batch_size=4 — then every trainable parameter
is compared. lam_recon=0 (required by grad_cache) and dropout=0 (required) make the two paths mathematically
identical, so the params must agree to fp tolerance.
"""

import os

import torch

from langset import LangSetModel, Trainer, TrainingArguments

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")

_ROWS = [
    {
        "input_text": f"query about topic {i}",
        "target_text": f"document describing topic {i} in detail",
    }
    for i in range(8)
]


def _one_step(grad_cache: bool, gc_chunk: int, stop_grad_target: bool = False) -> dict:
    torch.manual_seed(0)  # identical LoRA init + data order across the two builds
    m = LangSetModel.from_pretrained(ARCH, device="cpu", dropout=0.0)
    args = TrainingArguments(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_len=32,
        val_frac=0.125,
        lam_recon=0.0,
        lam_uniform=0.1,
        max_steps_per_epoch=1,
        eval_every=999,
        grad_cache=grad_cache,
        gc_chunk=gc_chunk,
        stop_grad_target=stop_grad_target,
        seed=0,
        verbose=False,
    )
    Trainer(m, args, _ROWS).train()
    return {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}


def test_gradcache_matches_direct_update() -> None:
    direct = _one_step(grad_cache=False, gc_chunk=0)
    gc = _one_step(grad_cache=True, gc_chunk=2)
    assert set(direct) == set(gc)
    max_abs = max((direct[n] - gc[n]).abs().max().item() for n in direct)
    assert max_abs < 1e-4, (
        f"GradCache update diverged from direct by {max_abs:.3e} (should be fp noise)"
    )


def test_gradcache_matches_direct_update_stop_grad_target() -> None:
    """stop_grad_target: the target is a frozen key (gradient reaches only `pred`). GradCache must cache/inject
    only the pred grad and skip the no-grad target — otherwise autograd.backward errors on a graph-less target."""
    direct = _one_step(grad_cache=False, gc_chunk=0, stop_grad_target=True)
    gc = _one_step(grad_cache=True, gc_chunk=2, stop_grad_target=True)
    assert set(direct) == set(gc)
    max_abs = max((direct[n] - gc[n]).abs().max().item() for n in direct)
    assert max_abs < 1e-4, (
        f"GradCache (stop_grad_target) diverged from direct by {max_abs:.3e} (should be fp noise)"
    )
