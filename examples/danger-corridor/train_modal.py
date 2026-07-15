"""Modal runner for the danger-corridor world -- the cleanest pretrain-WIN ablation.

The walk is mechanical (a random-init transformer learns "advance one cell per tick" from a handful of
examples, exactly as maze showed). The ONLY knowledge-gated moment is the crux tick, where you stand on the
animal and die-vs-pass is decided by the animal's danger -- a fact never in the text, only pretraining
supplies it for a HELD-OUT animal. So the entire pretrained-vs-random gap here IS the animal knowledge,
with maze skill held equal. Hypothesis: pretrained BEATS random on held-out die-vs-survive accuracy.

Both arms train on the same fixed-seed train pool (seed 0) and eval on a DISJOINT held-out animal pool
(seed 999); wandb project `langset-danger` (danger-pretrained / danger-random).

    modal run --detach examples/danger-corridor/train_modal.py                  # both arms
    modal run --detach examples/danger-corridor/train_modal.py --only random
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
    .add_local_dir(str(_ROOT / "examples" / "danger-corridor"), "/pkg/example")
)

app = modal.App("langset-danger")
hf_cache = modal.Volume.from_name("langset-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")])
def train_danger(random_init: bool = False, epochs: int = 30, n_train: int = 6000, n_eval: int = 1000,
                 backbone: str = "HuggingFaceTB/SmolLM2-135M", tokenizer: str = "", arch_overrides: str = "",
                 bs: int = 64, lr: float = 2e-4, max_len: int = 256, eval_seed: int = 999) -> None:
    import os
    import subprocess
    import sys

    ex = "/pkg/example"
    arm = "random" if random_init else "pretrained"
    out = f"/cache/danger-{arm}"
    train_npz, eval_npz = "/tmp/danger_train.npz", "/tmp/danger_eval.npz"
    env = {**os.environ, "PYTHONPATH": "/pkg/src", "HF_HOME": "/cache/hf",
           "WANDB_NAME": f"danger-{arm}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def sh(cmd: list[str]) -> None:
        print(f"[danger/{arm}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ex, env=env, check=True)

    # train pool (seed 0) and DISJOINT held-out animal pool (seed 999) -> generalization test, not memorization
    sh([sys.executable, "gen_danger.py", "build", str(n_train), train_npz, "0", "train"])
    sh([sys.executable, "gen_danger.py", "build", str(n_eval), eval_npz, str(eval_seed), "heldout"])

    train_cmd = [sys.executable, "train.py", "--data", train_npz, "--out", out,
                 "--backbone", backbone, "--epochs", str(epochs), "--bs", str(bs), "--lr", str(lr),
                 "--max-len", str(max_len), "--device", "cuda", "--wandb", "--wandb-project", "langset-danger"]
    if random_init:
        train_cmd.append("--random-init")
        if tokenizer:
            train_cmd += ["--tokenizer", tokenizer]
        if arch_overrides:
            train_cmd += ["--arch-overrides", arch_overrides]
    sh(train_cmd)
    hf_cache.commit()

    print(f"[danger/{arm}] ===== HELD-OUT EVAL =====", flush=True)
    sh([sys.executable, "danger_eval.py", "--data", eval_npz, "--ckpt", out, "--device", "cuda"])
    hf_cache.commit()


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def eval_danger(arms: str = "pretrained,random", n_eval: int = 1000, eval_seed: int = 999) -> None:
    """Re-run the held-out eval (incl. the cat-in-the-box FSQ-entropy readout) against persisted checkpoints,
    so eval-code changes don't require retraining. Uses the SAME seed-999 held-out animal pool."""
    import os
    import subprocess
    import sys

    ex = "/pkg/example"
    env = {**os.environ, "PYTHONPATH": "/pkg/src", "HF_HOME": "/cache/hf"}
    eval_npz = "/tmp/danger_eval.npz"
    subprocess.run([sys.executable, "gen_danger.py", "build", str(n_eval), eval_npz, str(eval_seed), "heldout"],
                   cwd=ex, env=env, check=True)
    for arm in [x.strip() for x in arms.split(",") if x.strip()]:
        print(f"########## DANGER HELD-OUT EVAL: danger-{arm} ##########", flush=True)
        subprocess.run([sys.executable, "danger_eval.py", "--data", eval_npz,
                        "--ckpt", f"/cache/danger-{arm}", "--device", "cuda"], cwd=ex, env=env, check=True)


@app.local_entrypoint()
def main(only: str = "both", epochs: int = 30, n_train: int = 6000, n_eval: int = 1000,
         bs: int = 64, lr: float = 2e-4, arch_overrides: str = "") -> None:
    kw = dict(epochs=epochs, n_train=n_train, n_eval=n_eval, bs=bs, lr=lr)
    handles = []
    if only in ("both", "pretrained"):
        handles.append(("pretrained", train_danger.spawn(random_init=False, **kw)))
    if only in ("both", "random"):
        handles.append(("random", train_danger.spawn(random_init=True, arch_overrides=arch_overrides, **kw)))
    for name, h in handles:
        print(f"spawned danger-{name}: {h.object_id}")
    print("watch: wandb project `langset-danger` (runs danger-pretrained / danger-random)")
