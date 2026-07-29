"""Property-eval for the maze BFS-frontier world model — does it hold a CALIBRATED superposition?

The two world-model properties are measured with langset's own probes (`langset.probes`, graduated out of this
example): this file is now just maze glue — roll the model out, name the two ground-truth labels (frontier SIZE
and solvability), and hand them to the library. Everything reads off `rollout(..., return_soft=True)`:

  * SUPERPOSITION CALIBRATION (headline) — `calibration_corr(entropy, nbranch)`: does the emitted latent's native
    FSQ entropy track the true frontier size? Positive = a *calibrated* set of next states, not one guess.
  * SOLVABILITY — `linear_decodability(terminal/mean latent -> solvable)`: can the emitted trajectory separate a
    SOLVABLE maze from an UNSOLVABLE one?

Both use a maze-disjoint split (no leakage: the group id is the maze, so no maze straddles train/test).

  python eval.py --data maze_eval.npz --ckpt maze_model
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from langset import LangSetModel, calibration_corr, linear_decodability

_CELL_RE = re.compile(r"r(\d+)c(\d+)")


def true_cells(line: str, grid: int) -> list[int]:
    """A frontier line -> its ACTIVE cell indices. `dead:` cells have left the wavefront."""
    if "tick" not in line or " at " not in line:
        return []
    body = line.split(" at ", 1)[1].split("| dead:", 1)[0]
    return [
        int(r) * grid + int(c)
        for r, c in _CELL_RE.findall(body)
        if 0 <= int(r) < grid and 0 <= int(c) < grid
    ]


def recall_at_true_k(scores: np.ndarray, cells: list[int]) -> float:
    """Top-k of the per-cell scores against the true set, k = the true count — count-neutral, so it measures
    WHICH cells the emission holds rather than rewarding an over-inclusive readout."""
    k = len(cells)
    return len(set(np.argsort(scores)[-k:].tolist()) & set(cells)) / k if k else 1.0


def concept_rows_by_cell(grid: int) -> dict[int, int]:
    """Map numeric maze-cell ids to the concept codebook rows discovered during training.

    Concept alphabets are sorted by name for determinism, so row order is lexicographic
    (``r0c0, r0c1, r0c10, ...``), not numeric cell order.
    """
    names = sorted(f"r{r}c{c}" for r in range(grid) for c in range(grid))
    return {
        int(row) * grid + int(col): code_row
        for code_row, name in enumerate(names)
        for row, col in [_CELL_RE.fullmatch(name).groups()]
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="disjoint eval corpus npz from gen_maze.py")
    p.add_argument("--ckpt", required=True, help="trained checkpoint dir from train.py")
    p.add_argument("--max-rows", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument(
        "--grid",
        type=int,
        default=16,
        help="maze grid size — enables the IDENTITY metric (recall@true-k) when the checkpoint "
        "carries a concept codebook, i.e. which cells the emission holds, not just how many",
    )
    p.add_argument("--out", help="optional path for the structured JSON report")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    seeds = [str(s) for s in list(z["seed"])][: a.max_rows]
    nbr = [[int(x) for x in list(v)] for v in list(z["label_nbranch"])][
        : a.max_rows
    ]  # per-tick wavefront count (verdict tick=0)
    solv = [1 if str(v[0]) == "yes" else 0 for v in list(z["label_solvable"])][
        : a.max_rows
    ]  # constant per maze

    m = LangSetModel.load(a.ckpt, device=a.device)
    B = 64
    # per-tick arrays (calibration + count) and per-maze terminal/mean latents (solvability)
    tick_lat, tick_ent, tick_k, tick_gid = [], [], [], []
    maze_term, maze_mean, maze_y, maze_gid = [], [], [], []
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0 : s0 + B]
        L, lengths, sL, ent = m.rollout(
            chunk, max_steps=a.max_steps, return_lengths=True, return_soft=True
        )
        L = L.float().cpu().numpy()
        ent = ent.float().cpu().numpy()
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            ln = int(lengths[j])
            if ln < 1:
                continue
            ks = nbr[gi]
            # solvability: terminal emitted latent + mean over the whole emitted trajectory (one row per maze)
            maze_term.append(L[j, ln - 1])
            maze_mean.append(L[j, :ln].mean(0))
            maze_y.append(solv[gi])
            maze_gid.append(gi)
            # per-tick: align emitted tick t with the true wavefront count at t (skip the verdict tick, k=0)
            for t in range(min(ln, len(ks))):
                if ks[t] <= 0:  # verdict tick (no wavefront) -> not a superposition target
                    continue
                tick_lat.append(L[j, t])
                tick_ent.append(float(ent[j, t]))
                tick_k.append(ks[t])
                tick_gid.append(gi)

    # ---- (C) IDENTITY: recall@true-k, read straight off the concept codebook (no probe) ---------------------
    # Only meaningful for a codebook checkpoint: the emission's state half IS a mixture over the cells, so
    # projecting it back on the codebook gives per-cell scores directly. Split by solvability because ~37% of
    # these corpora are dead-end floods whose frontiers are much narrower — blending them makes the number move
    # when the corpus MIX changes rather than when the model does.
    identity = None
    if getattr(m.head, "code_emit", False) and bool(m.head.code.abs().sum()):
        code = m.head.code.detach().float().cpu()
        cell_to_code_row = concept_rows_by_cell(a.grid)
        if code.size(0) != len(cell_to_code_row):
            raise ValueError(
                f"maze identity eval needs {len(cell_to_code_row)} cell concepts, "
                f"but checkpoint carries {code.size(0)} code rows"
            )
        sd = m.head.state_dim
        futs = [[str(t) for t in list(ft)] for ft in list(z["fut_text"])][: a.max_rows]
        recs = []
        for s0 in range(0, len(seeds), B):
            chunk = seeds[s0 : s0 + B]
            L2, lengths2, _s, _e = m.rollout(
                chunk, max_steps=a.max_steps, return_lengths=True, return_soft=True
            )
            L2 = L2.float().cpu()
            for j, gi in enumerate(range(s0, s0 + len(chunk))):
                ln = int(lengths2[j])
                t_scored = 0
                for line in futs[gi]:
                    cells = true_cells(line, a.grid)
                    if not cells:
                        continue
                    if t_scored >= ln:
                        break
                    sc = (L2[j, t_scored, :sd] @ code.t()).numpy()
                    code_rows = [cell_to_code_row[cell] for cell in cells]
                    recs.append(
                        {
                            "k": len(cells),
                            "rec": recall_at_true_k(sc, code_rows),
                            "solv": solv[gi],
                        }
                    )
                    t_scored += 1
        if recs:

            def agg(rs: list[dict[str, float | int | bool]]) -> dict[str, object]:
                by_k = {
                    int(k): round(float(np.mean([r["rec"] for r in rs if r["k"] == k])), 3)
                    for k in sorted({r["k"] for r in rs})
                    if len([r for r in rs if r["k"] == k]) >= 8
                }
                return {
                    "n_ticks": len(rs),
                    "recall_at_true_k": round(float(np.mean([r["rec"] for r in rs])), 3),
                    "mean_k": round(float(np.mean([r["k"] for r in rs])), 2),
                    "by_k": by_k,
                }

            identity = {
                "all": agg(recs),
                "solvable": agg([r for r in recs if r["solv"]]),
                "unsolvable": agg([r for r in recs if not r["solv"]]),
            }
            print(
                f"[eval] identity recall@true-k: all={identity['all']['recall_at_true_k']} "
                f"solvable={identity['solvable']['recall_at_true_k']} "
                f"unsolvable={identity['unsolvable']['recall_at_true_k']}",
                flush=True,
            )

    n_maze = len(maze_y)
    print(
        f"[eval] {a.ckpt}: {n_maze} mazes | {len(tick_k)} wavefront ticks "
        f"| solvable={sum(maze_y)}/{n_maze}",
        flush=True,
    )
    if n_maze < 40 or len(tick_k) < 50:
        print(
            json.dumps(
                {
                    "ckpt": a.ckpt,
                    "error": "too few samples",
                    "n_maze": n_maze,
                    "n_tick": len(tick_k),
                }
            )
        )
        return

    # ---- (A) SOLVABILITY: langset.probes linear decodability, split BY MAZE (each maze is one row here) --------
    Xt = np.stack(maze_term)
    Xm = np.stack(maze_mean)
    gm = np.array(maze_gid)
    solv_terminal = linear_decodability(Xt, maze_y, gm, test_frac=0.4)
    solv_mean = linear_decodability(Xm, maze_y, gm, test_frac=0.4)

    # ---- (B) CALIBRATION: corr(entropy, count) + count decodability, on ONE shared maze-disjoint cut ----------
    # derive the split from the TICK groups (the exact universe both per-tick probes see) and hand the SAME
    # held-out set to each, so corr_entropy_nbranch and count_decodability are always comparable.
    Xk = np.stack(tick_lat)
    ka = np.array(tick_k)
    gk = np.array(tick_gid)
    order = np.unique(gk).copy()
    np.random.default_rng(0).shuffle(order)
    test_g = set(order[: max(1, int(round(len(order) * 0.4)))].tolist())
    corr = calibration_corr(tick_ent, ka, groups=gk, test_groups=test_g)
    count = linear_decodability(Xk, ka, gk, test_groups=test_g)

    out = {
        "ckpt": a.ckpt,
        "n_maze": n_maze,
        "n_tick": len(tick_k),
        "solvable_frac": round(float(np.mean(maze_y)), 3),
        "A_solvability": {"terminal_latent": solv_terminal, "mean_latent": solv_mean},
        "B_calibration": {
            "corr_entropy_nbranch": corr,
            "count_decodability": count,
            "k_hist": {int(k): int((ka == k).sum()) for k in sorted(set(ka.tolist()))},
        },
    }
    if identity is not None:  # codebook checkpoints also report WHICH cells, not just how many
        out["C_identity"] = identity
    print("=== MAZE PROPERTY EVAL ===")
    print(json.dumps(out))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote {a.out}", flush=True)
    # ---- verdict: do the properties HOLD? ----
    solv_ok = (solv_mean["bal_acc"] or 0) >= 0.65 or (solv_terminal["bal_acc"] or 0) >= 0.65
    calib_ok = (corr or 0) > 0
    print(
        json.dumps(
            {
                "PROPERTIES_HOLD": bool(solv_ok and calib_ok),
                "solvability_ok": bool(solv_ok),
                "calibration_ok": bool(calib_ok),
            }
        )
    )


if __name__ == "__main__":
    main()
