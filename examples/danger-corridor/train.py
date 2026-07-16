"""Train a langset multi-latent world model on the maze BFS-frontier task — the SUPERPOSITION demo.

Each row is one maze; its `target_texts` are the per-tick frontiers of a parallel breadth-first flood from S
(see gen_maze.py). The model emits ONE latent per tick, and each tick's target describes the whole SET of
wavefront cells active that tick — so a single emitted latent must represent a *superposition* of next states,
not one. Reading out `rollout(..., return_soft=True)`, eval.py then checks the headline property: the emitted
latent's FSQ entropy tracks the frontier SIZE (calibrated uncertainty), and the frontier is recoverable across
branch counts. That's a world model that holds the distribution of where the search could be, not a single guess.

Two langset pieces make this work:
  * multi_latent=True            — variable-length latent set, one latent per tick, STOP-terminated
  * selector=last_epoch_selector — retrieval MRR rewards a COLLAPSED one-future-per-tick geometry, exactly the
                                   wrong signal here (it's meant to fall as the latent spreads over the set), so
                                   we keep the last epoch instead of early-stopping on it
Anti-collapse is the default stop-grad EMA twin; pass --sigreg for the EMA-free LeJEPA alternative.

  python gen_maze.py build 4000 maze.npz                 # generate the training corpus
  python train.py --data maze.npz --out maze_model       # train (add --wandb to log)
  python gen_maze.py build 800 maze_eval.npz 999         # disjoint eval corpus (seed 999)
  python eval.py --data maze_eval.npz --ckpt maze_model  # calibration + solvability property eval
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from langset import LangSetModel, Trainer, TrainingArguments
from langset.strategies import SIGRegTarget, last_epoch_selector


def build_rows(z, max_fut: int) -> list[dict]:
    """npz -> langset rows: input_text = maze ASCII, target_texts = per-tick frontier descriptions."""
    seeds = [str(s) for s in list(z["seed"])]
    fut_text = [[str(t) for t in list(ft)] for ft in list(z["fut_text"])]
    rows = []
    for s, fts in zip(seeds, fut_text):
        keep = [t for t in fts if t.strip()][:max_fut]
        if keep and s.strip():
            rows.append({"input_text": s, "target_texts": keep})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="corpus npz from gen_eco.py")
    p.add_argument("--out", required=True, help="checkpoint output dir")
    p.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-fut", type=int, default=32, help="cap on emitted latents (ticks) per maze")
    p.add_argument("--fsq-dim", type=int, default=256)
    p.add_argument("--fsq-levels", type=int, default=8)
    p.add_argument("--sigreg", action="store_true", help="EMA-free anti-collapse (LeJEPA) instead of the EMA twin")
    p.add_argument("--sigreg-lambda", type=float, default=0.3)
    p.add_argument("--random-init", action="store_true",
                   help="CONTROL ARM: random-init backbone (no pretraining), full-param train, decoupled tokenizer. "
                        "Tests whether the pretrained LLM matters for THIS world (mechanical maze -> expect it does not).")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id for --random-init (default: same as --backbone)")
    p.add_argument("--arch-overrides", default=None,
                   help="JSON dict of config shrinks for --random-init, e.g. "
                        "'{\"num_hidden_layers\": 6, \"hidden_size\": 384, \"num_attention_heads\": 6, "
                        "\"num_key_value_heads\": 2, \"intermediate_size\": 1024}'")
    p.add_argument("--train-base", action="store_true",
                   help="DISENTANGLER ARM: pretrained backbone, FULL-PARAM train (unfreeze base, not LoRA-only). "
                        "Removes the LoRA-vs-full-FT regime confound so pretrained-vs-random differs ONLY in init. "
                        "Use a gentler lr (~2e-5) so full-FT does not overwrite the pretrained knowledge.")
    p.add_argument("--device", default="cuda", help="cuda (real runs) or cpu (a tiny smoke)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (recommended for real runs)")
    p.add_argument("--wandb-project", default="langset-danger")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    rows = build_rows(z, a.max_fut)
    print(f"[train] {len(rows)} maze rows | backbone={a.backbone} | device={a.device}", flush=True)

    if a.random_init:                        # CONTROL ARM: no pretrained knowledge, full-param train, chosen tokenizer
        import json
        overrides = json.loads(a.arch_overrides) if a.arch_overrides else None
        print(f"[train] RANDOM-INIT arch={a.backbone} tokenizer={a.tokenizer or a.backbone} overrides={overrides}",
              flush=True)
        model = LangSetModel.from_scratch(
            a.backbone, tokenizer_id=a.tokenizer, latent_dim=None, n_latents=1, multi_latent=True,
            fsq_dim=a.fsq_dim, fsq_levels=a.fsq_levels, max_len=a.max_len,
            bf16=(a.device == "cuda"), device=a.device, arch_overrides=overrides)
    else:
        if a.train_base:
            print("[train] PRETRAINED FULL-FT (train_base=True): base unfrozen, disentangler arm", flush=True)
        model = LangSetModel.from_pretrained(
            a.backbone, latent_dim=None, n_latents=1, multi_latent=True,
            fsq_dim=a.fsq_dim, fsq_levels=a.fsq_levels, max_len=a.max_len,
            train_base=a.train_base, bf16=(a.device == "cuda"), device=a.device)

    opts: dict = dict(
        epochs=a.epochs, batch_size=a.bs, lr=a.lr, max_len=a.max_len,
        max_target_items=a.max_fut, val_frac=0.1,
        selector=last_epoch_selector,        # superposition: keep the last epoch (retr_mrr is meant to fall)
        output_dir=a.out,
        report_to="wandb" if a.wandb else None, wandb_project=a.wandb_project)
    if a.sigreg:                             # optional: EMA-free isotropic-Gaussian anti-collapse (see langset/sigreg.py)
        opts.update(target_source=SIGRegTarget, sigreg_lambda=a.sigreg_lambda)
    if a.wandb:
        os.environ.setdefault("WANDB_NAME", a.out)

    Trainer(model, TrainingArguments(**opts), rows).train()
    print(f"[train] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
