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
import re

import numpy as np

from langset import LangSetModel, Trainer, TrainingArguments
from langset.strategies import (
    CODE_SOURCES,
    ConceptObjective,
    SIGRegTarget,
    StateResidualObjective,
    last_epoch_selector,
)

_CELL_RE = re.compile(r"r(\d+)c(\d+)")


def _active_cells(line: str, grid: int) -> list[int]:
    """A frontier line -> its ACTIVE cell indices (r*grid+c). `dead:` cells dropped; non-frontier -> []. Mirrors
    eval_width.parse_active so the aux-head labels match exactly what the offline decode probe scores."""
    if "tick" not in line or " at " not in line:
        return []
    body = line.split(" at ", 1)[1].split("| dead:", 1)[0]
    out = []
    for r, c in _CELL_RE.findall(body):
        r, c = int(r), int(c)
        if 0 <= r < grid and 0 <= c < grid:
            out.append(r * grid + c)
    return out


def build_rows(z: np.lib.npyio.NpzFile, max_fut: int, grid: int = 0) -> list[dict]:
    """npz -> langset rows: input_text = maze ASCII, target_texts = per-tick frontier descriptions. When `grid`>0
    also attaches a `frontier` column of per-tick sparse active-cell-index lists,
    built from the SAME kept lines so it aligns 1:1 with target_texts."""
    seeds = [str(s) for s in list(z["seed"])]
    fut_text = [[str(t) for t in list(ft)] for ft in list(z["fut_text"])]
    rows = []
    for s, fts in zip(seeds, fut_text):
        keep = [t for t in fts if t.strip()][:max_fut]
        if keep and s.strip():
            row = {"input_text": s, "target_texts": keep}
            if grid > 0:
                cells = [_active_cells(t, grid) for t in keep]
                row["frontier"] = cells
                # the same frontier as TEXT: one concept dict per tick, cells named the way the target text
                # names them. No index is authored — langset discovers the alphabet from this column.
                row["concepts"] = [
                    {"frontier": [f"r{c // grid}c{c % grid}" for c in tick]} for tick in cells
                ]
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="corpus npz from gen_maze.py")
    p.add_argument("--out", required=True, help="checkpoint output dir")
    p.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument(
        "--max-fut", type=int, default=32, help="cap on emitted latents (ticks) per maze"
    )
    p.add_argument("--fsq-dim", type=int, default=256)
    p.add_argument("--fsq-levels", type=int, default=8)
    p.add_argument(
        "--kv-cache",
        action="store_true",
        help="KV-CACHE the multi-latent rollout: forward the prompt ONCE, then feed each latent token "
        "alone against the cached prefix K/V instead of re-running the whole growing sequence every "
        "tick. Numerically identical (~1e-7), but activation memory is ~1 prompt forward + n single "
        "tokens instead of n full-prefix forwards, removing the O(ticks) blowup. Trains long-rollout "
        "corpora without gradient checkpointing.",
    )
    p.add_argument(
        "--sigreg",
        action="store_true",
        help="EMA-free anti-collapse (LeJEPA) instead of the EMA twin",
    )
    p.add_argument("--sigreg-lambda", type=float, default=0.3)
    p.add_argument(
        "--random-init",
        action="store_true",
        help="CONTROL ARM: random-init backbone (no pretraining), full-param train, decoupled tokenizer. "
        "Tests whether the pretrained LLM matters for THIS world (mechanical maze -> expect it does not).",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer id for --random-init (default: same as --backbone)",
    )
    p.add_argument(
        "--arch-overrides",
        default=None,
        help="JSON dict of config shrinks for --random-init, e.g. "
        '\'{"num_hidden_layers": 6, "hidden_size": 384, "num_attention_heads": 6, '
        '"num_key_value_heads": 2, "intermediate_size": 1024}\'',
    )
    p.add_argument(
        "--train-base",
        action="store_true",
        help="DISENTANGLER ARM: pretrained backbone, FULL-PARAM train (unfreeze base, not LoRA-only). "
        "Removes the LoRA-vs-full-FT regime confound so pretrained-vs-random differs ONLY in init. "
        "Use a gentler lr (~2e-5) so full-FT does not overwrite the pretrained knowledge.",
    )
    p.add_argument(
        "--state-residual",
        action="store_true",
        help="STATE + RESIDUAL emission (the third way to shape the latent, next to --fsq-dim and "
        "emb_slots). The latent splits: its first dims are a MIXTURE over the grid*grid cells — "
        "one softmax, soft 1/k target, so the cells COMPETE for a fixed budget of mass and the "
        "emission is constituted from the frontier rather than having it decoded back out — and "
        "the last --res-dim dims are a residual the cell alphabet cannot name, shaped by the "
        "twin/recon terms. Needs --grid; reuses the same per-tick cell column as --concepts.",
    )
    p.add_argument(
        "--res-dim",
        type=int,
        default=64,
        help="Width of the residual half under --state-residual (0 = pure state). The residual is what "
        "keeps a named alphabet from being a ceiling: watch `res_share` in the logs, which is how "
        "much of the emission the alphabet could NOT account for.",
    )
    p.add_argument(
        "--concepts",
        action="store_true",
        help="CONCEPTS emission (the text-in form of --state-residual, and the one to prefer). Each "
        "tick's frontier is written as cell NAMES (r3c4) in a `concepts` column; langset "
        "discovers the alphabet and builds the codebook via --code-source. With one facet, "
        "--code-source random and --res-dim 0 this is exactly the arm that beat the text "
        "baseline at wide frontiers — a soft 1/k target over the cells, so they compete for a "
        "fixed budget of mass, plus an independent stop head.",
    )
    p.add_argument(
        "--code-source",
        default="random",
        choices=sorted(CODE_SOURCES),
        help="Where each cell's vector comes from under --state-residual. 'random' = seeded "
        "orthonormal, arbitrary but LOSSLESSLY decodable (the benchmark default: no probe, so "
        "recall measures the emission). 'model' = the base model's embedding of the cell's name, "
        "so adjacent cells are related — semantic, but a mixture no longer inverts exactly. "
        "'orthogonal' = embed then orthogonalize (both, with distortion). 'twin' = encode the "
        "name through the emit path, putting the codebook in the SAME space as the targets.",
    )
    p.add_argument(
        "--grid",
        type=int,
        default=16,
        help="maze grid size (frontier label space = grid*grid cells)",
    )
    p.add_argument("--device", default="cuda", help="cuda (real runs) or cpu (a tiny smoke)")
    p.add_argument(
        "--wandb", action="store_true", help="log to Weights & Biases (recommended for real runs)"
    )
    p.add_argument("--wandb-project", default="langset-maze")
    a = p.parse_args()
    if a.state_residual and a.concepts:
        p.error("--state-residual and --concepts are alternative state-data interfaces; choose one")

    z = np.load(a.data, allow_pickle=True)
    rows = build_rows(z, a.max_fut, grid=a.grid if (a.state_residual or a.concepts) else 0)
    print(f"[train] {len(rows)} maze rows | backbone={a.backbone} | device={a.device}", flush=True)

    if a.random_init:  # CONTROL ARM: no pretrained knowledge, full-param train, chosen tokenizer
        import json

        overrides = json.loads(a.arch_overrides) if a.arch_overrides else None
        print(
            f"[train] RANDOM-INIT arch={a.backbone} tokenizer={a.tokenizer or a.backbone} overrides={overrides}",
            flush=True,
        )
        model = LangSetModel.from_scratch(
            a.backbone,
            tokenizer_id=a.tokenizer,
            latent_dim=None,
            n_latents=1,
            multi_latent=True,
            fsq_dim=a.fsq_dim,
            fsq_levels=a.fsq_levels,
            max_len=a.max_len,
            code_emit=(a.state_residual or a.concepts),
            n_codes=a.grid * a.grid if (a.state_residual or a.concepts) else 0,
            res_dim=a.res_dim if (a.state_residual or a.concepts) else 0,
            bf16=(a.device == "cuda"),
            device=a.device,
            arch_overrides=overrides,
        )
    else:
        if a.train_base:
            print(
                "[train] PRETRAINED FULL-FT (train_base=True): base unfrozen, disentangler arm",
                flush=True,
            )
        model = LangSetModel.from_pretrained(
            a.backbone,
            latent_dim=None,
            n_latents=1,
            multi_latent=True,
            fsq_dim=1 if (a.state_residual or a.concepts) else a.fsq_dim,
            fsq_levels=a.fsq_levels,
            max_len=a.max_len,
            code_emit=(a.state_residual or a.concepts),
            n_codes=a.grid * a.grid if (a.state_residual or a.concepts) else 0,
            res_dim=a.res_dim if (a.state_residual or a.concepts) else 0,
            train_base=a.train_base,
            bf16=(a.device == "cuda"),
            device=a.device,
        )

    opts: dict = dict(
        epochs=a.epochs,
        batch_size=a.bs,
        lr=a.lr,
        max_len=a.max_len,
        max_target_items=a.max_fut,
        val_frac=0.1,
        selector=last_epoch_selector,  # superposition: keep the last epoch (retr_mrr is meant to fall)
        output_dir=a.out,
        kv_cache=a.kv_cache,
        report_to="wandb" if a.wandb else None,
        wandb_project=a.wandb_project,
    )
    if a.concepts:  # text-in concepts: cells as named members, one facet
        opts.update(emission=ConceptObjective, concept_field="concepts", code_source=a.code_source)
    if a.state_residual:  # named half = the cells; unnamed half = the residual
        # The alphabet is given as NAMES ("r3c4"), and `code_source` decides what vector each one gets — the
        # trade being measured here: "random" is arbitrary but decodes losslessly, so a recall number reflects
        # the emission and not a clever readout; "model"/"twin" put neighbouring cells near each other and give
        # that exactness up. Same alphabet either way, so the two are directly comparable on this task.
        cell_names = [f"r{i // a.grid}c{i % a.grid}" for i in range(a.grid * a.grid)]
        opts.update(
            emission=StateResidualObjective,
            state_field="frontier",
            state_classes=a.grid * a.grid,
            code_source=a.code_source,
            code_names=cell_names,
        )
    if a.sigreg:  # optional: EMA-free isotropic-Gaussian anti-collapse (see langset/sigreg.py)
        opts.update(target_source=SIGRegTarget, sigreg_lambda=a.sigreg_lambda)
    if a.wandb:
        os.environ.setdefault("WANDB_NAME", a.out)

    Trainer(model, TrainingArguments(**opts), rows).train()
    print(f"[train] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
