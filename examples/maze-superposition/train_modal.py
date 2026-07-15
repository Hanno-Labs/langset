"""Modal runner for the maze-superposition demo — the pretrained-vs-random-init ablation.

The question: does the pretrained LLM matter for THIS (mechanical) world? Maze BFS-frontier rollout is governed
by rules that live IN the training data, so the hypothesis is that a random-init transformer matches (or beats)
the pretrained backbone here — the "mechanical worlds don't need pretraining" arm of the 2x2. Both arms train on
the SAME corpus (fixed seeds) and log to wandb project `langset-maze` as `maze-pretrained` / `maze-random`.

    modal run --detach examples/maze-superposition/train_modal.py                  # both arms, defaults
    modal run --detach examples/maze-superposition/train_modal.py --only random    # just the random-init arm
    modal run --detach examples/maze-superposition/train_modal.py --epochs 40 --n-train 6000

Pretrained arm = SmolLM2-135M + LoRA (the example default). Random arm = same architecture, random weights,
full-parameter training (LoRA on random weights is meaningless), decoupled tokenizer. Pass --arch-overrides to
shrink the random net (a fairer small baseline) once we see whether same-capacity random underfits.
"""
from __future__ import annotations

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent.parent          # langset repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .pip_install("torch>=2.7.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.50", "hf_xet", "datasets", "wandb>=0.18", "numpy>=1.26",
                 "peft>=0.13.2", "sentence-transformers>=3.0")
    .add_local_dir(str(_ROOT / "src"), "/pkg/src")
    .add_local_dir(str(_ROOT / "examples" / "maze-superposition"), "/pkg/example")
)

app = modal.App("langset-maze")
hf_cache = modal.Volume.from_name("langset-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")])
def train_maze(random_init: bool = False, epochs: int = 30, n_train: int = 4000, n_eval: int = 800,
               backbone: str = "HuggingFaceTB/SmolLM2-135M", tokenizer: str = "", arch_overrides: str = "",
               sigreg: bool = False, bs: int = 16, lr: float = 2e-4, eval_seed: int = 999) -> None:
    """One arm: generate a fixed corpus, train (wandb), then run the calibration/solvability eval."""
    import os
    import subprocess
    import sys

    ex = "/pkg/example"
    arm = "random" if random_init else "pretrained"
    out = f"/cache/maze-{arm}"
    train_npz, eval_npz = "/tmp/maze.npz", "/tmp/maze_eval.npz"
    env = {**os.environ, "PYTHONPATH": "/pkg/src", "HF_HOME": "/cache/hf",
           "WANDB_NAME": f"maze-{arm}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def sh(cmd: list[str]) -> None:
        print(f"[maze/{arm}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ex, env=env, check=True)

    # Fixed seeds -> both arms train/eval on byte-identical corpora (fair comparison).
    sh([sys.executable, "gen_maze.py", "build", str(n_train), train_npz, "0"])
    sh([sys.executable, "gen_maze.py", "build", str(n_eval), eval_npz, str(eval_seed)])

    train_cmd = [sys.executable, "train.py", "--data", train_npz, "--out", out,
                 "--backbone", backbone, "--epochs", str(epochs), "--bs", str(bs), "--lr", str(lr),
                 "--device", "cuda", "--wandb", "--wandb-project", "langset-maze"]
    if random_init:
        train_cmd.append("--random-init")
        if tokenizer:
            train_cmd += ["--tokenizer", tokenizer]
        if arch_overrides:
            train_cmd += ["--arch-overrides", arch_overrides]
    if sigreg:
        train_cmd.append("--sigreg")
    sh(train_cmd)
    hf_cache.commit()                                            # persist the checkpoint before eval

    print(f"[maze/{arm}] ===== EVAL =====", flush=True)
    sh([sys.executable, "eval.py", "--data", eval_npz, "--ckpt", out])
    hf_cache.commit()


@app.local_entrypoint()
def main(only: str = "both", epochs: int = 30, n_train: int = 4000, n_eval: int = 800,
         bs: int = 64, lr: float = 2e-4, arch_overrides: str = "", sigreg: bool = False) -> None:
    """Launch the ablation. `only` = both | pretrained | random. Spawns arms in parallel (detached).
    bs default 64 (A10G VRAM was ~14% at bs16; a bigger batch fills it AND gives the NCE more negatives)."""
    kw = dict(epochs=epochs, n_train=n_train, n_eval=n_eval, bs=bs, lr=lr, sigreg=sigreg)
    handles = []
    if only in ("both", "pretrained"):
        handles.append(("pretrained", train_maze.spawn(random_init=False, **kw)))
    if only in ("both", "random"):
        handles.append(("random", train_maze.spawn(random_init=True, arch_overrides=arch_overrides, **kw)))
    for name, h in handles:
        print(f"spawned maze-{name}: {h.object_id}")
    print("watch: wandb project `langset-maze` (runs maze-pretrained / maze-random)")
