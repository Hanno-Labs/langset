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

import numpy as np
from langset import LangSetModel, calibration_corr, linear_decodability


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="disjoint eval corpus npz from gen_maze.py")
    p.add_argument("--ckpt", required=True, help="trained checkpoint dir from train.py")
    p.add_argument("--max-rows", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    seeds = [str(s) for s in list(z["seed"])][:a.max_rows]
    nbr = [[int(x) for x in list(v)] for v in list(z["label_nbranch"])][:a.max_rows]      # per-tick wavefront count (verdict tick=0)
    solv = [1 if str(v[0]) == "yes" else 0 for v in list(z["label_solvable"])][:a.max_rows]  # constant per maze

    m = LangSetModel.load(a.ckpt, device=a.device)
    B = 64
    # per-tick arrays (calibration + count) and per-maze terminal/mean latents (solvability)
    tick_lat, tick_ent, tick_k, tick_gid = [], [], [], []
    maze_term, maze_mean, maze_y, maze_gid = [], [], [], []
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0:s0 + B]
        L, lengths, sL, ent = m.rollout(chunk, max_steps=a.max_steps, return_lengths=True, return_soft=True)
        L = L.float().cpu().numpy(); ent = ent.float().cpu().numpy()
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            ln = int(lengths[j])
            if ln < 1:
                continue
            ks = nbr[gi]
            # solvability: terminal emitted latent + mean over the whole emitted trajectory (one row per maze)
            maze_term.append(L[j, ln - 1]); maze_mean.append(L[j, :ln].mean(0)); maze_y.append(solv[gi]); maze_gid.append(gi)
            # per-tick: align emitted tick t with the true wavefront count at t (skip the verdict tick, k=0)
            for t in range(min(ln, len(ks))):
                if ks[t] <= 0:            # verdict tick (no wavefront) -> not a superposition target
                    continue
                tick_lat.append(L[j, t]); tick_ent.append(float(ent[j, t])); tick_k.append(ks[t]); tick_gid.append(gi)

    n_maze = len(maze_y)
    print(f"[eval] {a.ckpt}: {n_maze} mazes | {len(tick_k)} wavefront ticks "
          f"| solvable={sum(maze_y)}/{n_maze}", flush=True)
    if n_maze < 40 or len(tick_k) < 50:
        print(json.dumps({"ckpt": a.ckpt, "error": "too few samples", "n_maze": n_maze, "n_tick": len(tick_k)}))
        return

    # ---- (A) SOLVABILITY: langset.probes linear decodability, split BY MAZE (each maze is one row here) --------
    Xt = np.stack(maze_term); Xm = np.stack(maze_mean); gm = np.array(maze_gid)
    solv_terminal = linear_decodability(Xt, maze_y, gm, test_frac=0.4)
    solv_mean = linear_decodability(Xm, maze_y, gm, test_frac=0.4)

    # ---- (B) CALIBRATION: corr(entropy, count) + count decodability, on ONE shared maze-disjoint cut ----------
    # derive the split from the TICK groups (the exact universe both per-tick probes see) and hand the SAME
    # held-out set to each, so corr_entropy_nbranch and count_decodability are always comparable.
    Xk = np.stack(tick_lat); ka = np.array(tick_k); gk = np.array(tick_gid)
    order = np.unique(gk).copy(); np.random.default_rng(0).shuffle(order)
    test_g = set(order[:max(1, int(round(len(order) * 0.4)))].tolist())
    corr = calibration_corr(tick_ent, ka, groups=gk, test_groups=test_g)
    count = linear_decodability(Xk, ka, gk, test_groups=test_g)

    out = {
        "ckpt": a.ckpt, "n_maze": n_maze, "n_tick": len(tick_k),
        "solvable_frac": round(float(np.mean(maze_y)), 3),
        "A_solvability": {"terminal_latent": solv_terminal, "mean_latent": solv_mean},
        "B_calibration": {"corr_entropy_nbranch": corr, "count_decodability": count,
                          "k_hist": {int(k): int((ka == k).sum()) for k in sorted(set(ka.tolist()))}},
    }
    print("=== MAZE PROPERTY EVAL ===")
    print(json.dumps(out))
    # ---- verdict: do the properties HOLD? ----
    solv_ok = (solv_mean["bal_acc"] or 0) >= 0.65 or (solv_terminal["bal_acc"] or 0) >= 0.65
    calib_ok = (corr or 0) > 0
    print(json.dumps({"PROPERTIES_HOLD": bool(solv_ok and calib_ok),
                      "solvability_ok": bool(solv_ok), "calibration_ok": bool(calib_ok)}))


if __name__ == "__main__":
    main()
