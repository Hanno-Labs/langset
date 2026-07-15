"""Geometry probe for a trained maze world model: HOW does it arrange the emitted-latent space?

Rolls a disjoint seed-999 corpus through the model, collects per-tick emitted latents aligned with (grid cell
positions on the wavefront, wavefront size, solvable), and measures:

  (1) EFFECTIVE DIMENSIONALITY — participation ratio of the emitted-latent PCA (full 256-dim use vs low-rank).
  (2) SPATIAL MAP — each grid cell (r,c) gets a linear readout direction (logistic-probe weight). Correlate
      cosine(dir_a, dir_b) against the cells' 2D Manhattan distance. Strong NEGATIVE = the latent geometry
      mirrors the maze grid (adjacent cells -> similar directions): a cognitive map.
  (3) AXES — where the solvable direction and the wavefront-COUNT direction sit vs the top principal components.

  python geom_probe.py --data maze_eval.npz --ckpt maze_v2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from langset import LangSetModel
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

GRID = 9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max-rows", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--dump", default=None, help="path to save an .npz of raw arrays for visualization")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    seeds = [str(s) for s in list(z["seed"])][:a.max_rows]
    nbr = [[int(x) for x in list(v)] for v in list(z["label_nbranch"])][:a.max_rows]
    solv = [1 if str(v[0]) == "yes" else 0 for v in list(z["label_solvable"])][:a.max_rows]
    import re
    # parse the per-tick wavefront cell coords from fut_text (same format eval_maze/maze_viz parse)
    futs = [[str(t) for t in list(ft)] for ft in list(z["fut_text"])][:a.max_rows]
    fronts = []
    for ft in futs:
        rows_ = []
        for e in ft:
            if e.startswith("verdict"):
                continue
            head = e.split("| dead:")[0]
            rows_.append([(int(r), int(c)) for r, c in re.findall(r"r(\d+)c(\d+)", head)])
        fronts.append(rows_)

    m = LangSetModel.load(a.ckpt, device="cuda")
    B = 64
    Lat, Cellhot, Ks, Solv = [], [], [], []
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0:s0 + B]
        L, lengths, sL, ent = m.rollout(chunk, max_steps=a.max_steps, return_lengths=True, return_soft=True)
        sL = sL.float().cpu().numpy()
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            fr = fronts[gi]; ks = nbr[gi]
            for t in range(min(int(lengths[j]), len(fr))):
                if not fr[t]:
                    continue
                Lat.append(sL[j, t])
                hot = np.zeros(GRID * GRID, dtype=int)
                for (r, c) in fr[t]:
                    if r < GRID and c < GRID:
                        hot[r * GRID + c] = 1
                Cellhot.append(hot); Ks.append(ks[t] if t < len(ks) else 0); Solv.append(solv[gi])
    X = np.stack(Lat); Y = np.stack(Cellhot); Ks = np.array(Ks); Solv = np.array(Solv)
    print(f"[geom] {a.ckpt}: {len(X)} tick-latents, dim={X.shape[1]}", flush=True)

    # (1) effective dimensionality: participation ratio of the covariance eigenspectrum
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum())      # ~n dims that carry the variance
    cum = np.cumsum(ev)
    d90 = int(np.searchsorted(cum, 0.90) + 1)

    # (2) spatial-map test: per-cell readout direction (probe weight), cosine vs 2D grid distance
    dirs, cells = {}, []
    for cell in range(GRID * GRID):
        pos = int(Y[:, cell].sum())
        if 8 <= pos <= len(Y) - 8:                             # cell seen enough to fit a stable direction
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, Y[:, cell])
            w = clf.coef_[0]; dirs[cell] = w / (np.linalg.norm(w) + 1e-9); cells.append(cell)
    cos_list, gdist_list = [], []
    for i in range(len(cells)):
        for jx in range(i + 1, len(cells)):
            ca, cb = cells[i], cells[jx]
            cos_list.append(float(dirs[ca] @ dirs[cb]))
            gdist_list.append(abs(ca // GRID - cb // GRID) + abs(ca % GRID - cb % GRID))
    spatial_rho = float(spearmanr(gdist_list, cos_list)[0]) if cos_list else None

    # (3) where solvable / count axes sit vs the top PCs
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:5]                                               # top-5 principal directions

    def axis_dir(target):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, target)
        w = clf.coef_[0]; return w / (np.linalg.norm(w) + 1e-9)
    solv_dir = axis_dir(Solv)
    kcut = (Ks >= 3).astype(int)                               # "many branches" vs "few"
    count_dir = axis_dir(kcut)
    solv_pc = [round(float(abs(solv_dir @ v)), 3) for v in pcs]
    count_pc = [round(float(abs(count_dir @ v)), 3) for v in pcs]

    out = {
        "ckpt": a.ckpt, "n_latents": len(X), "dim": int(X.shape[1]),
        "effective_dim_participation_ratio": round(part_ratio, 2),
        "dims_for_90pct_variance": d90,
        "top5_variance_frac": [round(float(x), 3) for x in ev[:5]],
        "spatial_map_spearman_gdist_vs_cos": round(spatial_rho, 3) if spatial_rho is not None else None,
        "n_cells_mapped": len(cells),
        "solvable_axis_|cos|_with_top5_PCs": solv_pc,
        "count_axis_|cos|_with_top5_PCs": count_pc,
    }
    print("=== MAZE GEOMETRY ===")
    print(json.dumps(out))

    # dump raw arrays so we can build any visualization locally (PCA/MDS/UMAP scatter,
    # recovered-map from per-cell directions, scree, distance-distance).
    if a.dump:
        rr = np.arange(GRID * GRID) // GRID
        cc = np.arange(GRID * GRID) % GRID
        cnt = Y.sum(1).clip(min=1)
        front_centroid_rc = np.stack([(Y * rr).sum(1) / cnt, (Y * cc).sum(1) / cnt], 1)
        pca_proj = Xc @ Vt[:10].T                              # top-10 PC scores per latent
        cell_dirs = np.stack([dirs[c] for c in cells]) if cells else np.zeros((0, X.shape[1]))
        cell_rc = np.array([[c // GRID, c % GRID] for c in cells], dtype=int)
        np.savez_compressed(
            a.dump,
            X=X.astype(np.float32),                            # raw tick-latents (for UMAP/t-SNE)
            pca_proj=pca_proj.astype(np.float32),
            pca_components=Vt[:10].astype(np.float32),
            eigenspectrum=ev.astype(np.float32),               # normalized PC variance fractions
            front_centroid_rc=front_centroid_rc.astype(np.float32),
            cellhot=Y.astype(np.int8),                         # full wavefront one-hot per latent
            branch_count=Ks.astype(np.int16),
            solvable=Solv.astype(np.int8),
            cell_ids=np.array(cells, dtype=int),               # which of the 81 cells were mapped
            cell_rc=cell_rc,
            cell_dirs=cell_dirs.astype(np.float32),            # per-cell readout directions (MDS map)
            solv_dir=solv_dir.astype(np.float32),
            count_dir=count_dir.astype(np.float32),
            summary_json=json.dumps(out),
        )
        print(f"[geom] dumped arrays -> {a.dump}", flush=True)


if __name__ == "__main__":
    main()
