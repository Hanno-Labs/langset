"""Modal runner for the trophic-cascade world -- the pretrain-WIN ablation.

Unlike maze (mechanical, in-distribution), this world's transitions are gated on real-entity knowledge
(each species' trophic role), and the eval corpus is a DISJOINT held-out species pool. So predicting a
held-out species' survival needs pretraining. Hypothesis: pretrained BEATS random-init here (the opposite
of maze). Both arms train on the same fixed-seed train pool, eval on the same held-out pool; wandb project
`langset-eco` (eco-pretrained / eco-random).

    modal run --detach examples/ecosystem-cascade/train_modal.py                 # both arms
    modal run --detach examples/ecosystem-cascade/train_modal.py --only random
"""
from __future__ import annotations

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .pip_install("torch>=2.7.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.50", "hf_xet", "datasets", "wandb>=0.18", "numpy>=1.26",
                 "peft>=0.13.2", "sentence-transformers>=3.0")
    .add_local_dir(str(_ROOT / "src"), "/pkg/src")
    .add_local_dir(str(_ROOT / "examples" / "ecosystem-cascade"), "/pkg/example")
)

app = modal.App("langset-eco")
hf_cache = modal.Volume.from_name("langset-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")])
def train_eco(random_init: bool = False, epochs: int = 30, n_train: int = 8000, n_eval: int = 1000,
              backbone: str = "HuggingFaceTB/SmolLM2-135M", tokenizer: str = "", arch_overrides: str = "",
              bs: int = 64, lr: float = 2e-4, max_len: int = 256, eval_seed: int = 999) -> None:
    import os
    import subprocess
    import sys

    ex = "/pkg/example"
    arm = "random" if random_init else "pretrained"
    out = f"/cache/eco-{arm}"
    train_npz, eval_npz = "/tmp/eco_train.npz", "/tmp/eco_eval.npz"
    env = {**os.environ, "PYTHONPATH": "/pkg/src", "HF_HOME": "/cache/hf",
           "WANDB_NAME": f"eco-{arm}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def sh(cmd: list[str]) -> None:
        print(f"[eco/{arm}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ex, env=env, check=True)

    # train pool (seed 0) and DISJOINT held-out pool (seed 999) -> generalization test, not memorization
    sh([sys.executable, "gen_eco.py", "build", str(n_train), train_npz, "0", "train"])
    sh([sys.executable, "gen_eco.py", "build", str(n_eval), eval_npz, str(eval_seed), "heldout"])

    train_cmd = [sys.executable, "train.py", "--data", train_npz, "--out", out,
                 "--backbone", backbone, "--epochs", str(epochs), "--bs", str(bs), "--lr", str(lr),
                 "--max-len", str(max_len), "--device", "cuda", "--wandb", "--wandb-project", "langset-eco"]
    if random_init:
        train_cmd.append("--random-init")
        if tokenizer:
            train_cmd += ["--tokenizer", tokenizer]
        if arch_overrides:
            train_cmd += ["--arch-overrides", arch_overrides]
    sh(train_cmd)
    hf_cache.commit()

    print(f"[eco/{arm}] ===== HELD-OUT EVAL =====", flush=True)
    sh([sys.executable, "eco_eval.py", "--data", eval_npz, "--ckpt", out, "--device", "cuda"])
    hf_cache.commit()


@app.local_entrypoint()
def main(only: str = "both", epochs: int = 30, n_train: int = 8000, n_eval: int = 1000,
         bs: int = 64, lr: float = 2e-4, arch_overrides: str = "") -> None:
    kw = dict(epochs=epochs, n_train=n_train, n_eval=n_eval, bs=bs, lr=lr)
    handles = []
    if only in ("both", "pretrained"):
        handles.append(("pretrained", train_eco.spawn(random_init=False, **kw)))
    if only in ("both", "random"):
        handles.append(("random", train_eco.spawn(random_init=True, arch_overrides=arch_overrides, **kw)))
    for name, h in handles:
        print(f"spawned eco-{name}: {h.object_id}")
    print("watch: wandb project `langset-eco` (runs eco-pretrained / eco-random)")
