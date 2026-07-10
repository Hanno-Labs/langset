"""Property-eval for the maze BFS-frontier world model — does it hold a CALIBRATED superposition?

Probes the trained checkpoint on a DISJOINT eval corpus (build it with a different seed, e.g.
`python gen_maze.py build 800 maze_eval.npz 999`). Two properties, both read from the emitted latents (reported
below as `B_calibration` and `A_solvability`):

  * SUPERPOSITION CALIBRATION (the headline) — does the per-tick emitted latent encode the frontier SIZE, and
    does its FSQ entropy track that size? Reported as branch-count probe acc/MAE + corr(entropy, nbranch). A
    positive correlation means the single latent carries a *calibrated* set of next states, not one guess.
  * SOLVABILITY — can the emitted trajectory tell a SOLVABLE maze from an UNSOLVABLE one? Probe:
    terminal / mean emitted latent -> solvable yes/no.

Everything is a linear probe on a maze-disjoint 60/40 split (no leakage: split by maze id, not by tick).

  python eval.py --data maze_eval.npz --ckpt maze_model
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from langset import LangSetModel
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression


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
    # per-tick rows (for calibration) and per-maze terminal/mean latents (for solvability)
    tick_lat, tick_ent, tick_k, tick_mi = [], [], [], []
    maze_term, maze_mean, maze_y = [], [], []
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0:s0 + B]
        L, lengths, sL, ent = m.rollout(chunk, max_steps=a.max_steps, return_lengths=True, return_soft=True)
        L = L.float().cpu().numpy(); ent = ent.float().cpu().numpy()
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            ln = int(lengths[j])
            if ln < 1:
                continue
            ks = nbr[gi]
            # (A) solvability features: terminal emitted latent + mean over the whole emitted trajectory
            maze_term.append(L[j, ln - 1]); maze_mean.append(L[j, :ln].mean(0)); maze_y.append(solv[gi])
            # (B) per-tick: align emitted tick t with the true wavefront count at t (skip the verdict tick, k=0)
            for t in range(min(ln, len(ks))):
                if ks[t] <= 0:            # verdict tick (no wavefront) -> not a superposition target
                    continue
                tick_lat.append(L[j, t]); tick_ent.append(float(ent[j, t])); tick_k.append(ks[t]); tick_mi.append(gi)

    n_maze = len(maze_y)
    print(f"[eval] {a.ckpt}: {n_maze} mazes | {len(tick_k)} wavefront ticks "
          f"| solvable={sum(maze_y)}/{n_maze}", flush=True)
    if n_maze < 40 or len(tick_k) < 50:
        print(json.dumps({"ckpt": a.ckpt, "error": "too few samples", "n_maze": n_maze, "n_tick": len(tick_k)}))
        return

    # ---- (A) SOLVABILITY: probe emitted latent -> solvable, split by MAZE (each maze is one row here) ----
    Xt = np.stack(maze_term); Xm = np.stack(maze_mean); ym = np.array(maze_y)
    order = np.arange(n_maze); rng = np.random.default_rng(0); rng.shuffle(order)
    cut = int(n_maze * 0.6); tr_m = order[:cut]; te_m = order[cut:]
    ybase = float(max(ym[te_m].mean(), 1 - ym[te_m].mean()))   # majority-class baseline

    def solv_probe(X):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr_m], ym[tr_m])
        pr = clf.predict(X[te_m]); yt = ym[te_m]
        acc = float((pr == yt).mean())
        # balanced acc = mean of per-class recall (robust to the 55/45 skew)
        rec_s = float((pr[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0
        rec_u = float((pr[yt == 0] == 0).mean()) if (yt == 0).any() else 0.0
        return {"acc": round(acc, 3), "bal_acc": round((rec_s + rec_u) / 2, 3),
                "recall_solvable": round(rec_s, 3), "recall_unsolvable": round(rec_u, 3)}

    # ---- (B) CALIBRATION: per-tick latent -> wavefront count, + corr(entropy, count). Split by maze id ----
    Xk = np.stack(tick_lat); ent_a = np.array(tick_ent); ka = np.array(tick_k); mia = np.array(tick_mi)
    te_set = set(te_m.tolist())
    te_t = np.array([mm in te_set for mm in mia]); tr_t = ~te_t

    def count_probe():
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xk[tr_t], ka[tr_t])
        pr = clf.predict(Xk[te_t])
        return round(float((pr == ka[te_t]).mean()), 3), round(float(np.abs(pr - ka[te_t]).mean()), 3)

    def ent_corr():
        if np.std(ent_a[te_t]) == 0 or np.std(ka[te_t]) == 0:
            return None
        return round(float(pearsonr(ent_a[te_t], ka[te_t])[0]), 3)

    c_acc, c_mae = count_probe()
    out = {
        "ckpt": a.ckpt, "n_maze": n_maze, "n_tick": len(tick_k),
        "solvable_frac": round(float(ym.mean()), 3),
        "A_solvability": {"baseline_majority": round(ybase, 3),
                          "terminal_latent": solv_probe(Xt), "mean_latent": solv_probe(Xm)},
        "B_calibration": {"count_acc": c_acc, "count_mae": c_mae,
                          "corr_entropy_nbranch": ent_corr(),
                          "k_hist": {int(k): int((ka == k).sum()) for k in sorted(set(ka.tolist()))}},
    }
    print("=== MAZE PROPERTY EVAL ===")
    print(json.dumps(out))
    # ---- verdict: do the properties HOLD? ----
    solv_ok = out["A_solvability"]["mean_latent"]["bal_acc"] >= 0.65 or \
        out["A_solvability"]["terminal_latent"]["bal_acc"] >= 0.65
    calib_ok = (out["B_calibration"]["corr_entropy_nbranch"] or 0) > 0
    print(json.dumps({"PROPERTIES_HOLD": bool(solv_ok and calib_ok),
                      "solvability_ok": bool(solv_ok), "calibration_ok": bool(calib_ok)}))


if __name__ == "__main__":
    main()
