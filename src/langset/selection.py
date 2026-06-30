"""Validation / early-stop metrics. The hard-won rules:
  - NEVER select on training loss (InfoNCE+EMA can minimize it by COLLAPSING the geometry).
  - NEVER score retrieval against the frozen bootstrap targets (the model specializes AWAY from them).
  - Score on held-out, in the CURRENT geometry: retrieval mrr + a collapse guard; and, when the rows carry
    geometry labels, held-out kNN purity (+ beats-bootstrap), which is the preferred, intent-aligned selector.
Geometry labels are EVAL-ONLY probes — they never enter training (training on them would collapse the space).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def retrieval_mrr(pred: np.ndarray, bank: np.ndarray) -> dict[str, float]:
    """Each pred[i] should retrieve target bank[i] among the val set (rank-based, scale-free, collapse-sensitive)."""
    sims = pred @ bank.T
    truth = np.arange(len(pred))
    order = np.argsort(-sims, axis=1)
    rank = np.array([int(np.where(order[i] == truth[i])[0][0]) for i in range(len(truth))])
    return {"r@1": float((rank == 0).mean()), "mrr": float((1.0 / (rank + 1)).mean())}


def collapse_score(X: np.ndarray) -> float:
    """Mean off-diagonal cosine of (normalized) emissions. ->1 means collapsed; healthy spaces sit low."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    s = Xn @ Xn.T
    np.fill_diagonal(s, np.nan)
    return float(np.nanmean(s))


def knn_purity(X: np.ndarray, labels: list[str], k: int = 5) -> float:
    """Leave-one-out majority-vote accuracy by label (does the geometry organize by this attribute?)."""
    keep = [i for i, l in enumerate(labels) if str(l) not in ("", "unknown", "none", "nan")]
    if len(keep) < k + 2:
        return float("nan")
    Xk = X[keep] / (np.linalg.norm(X[keep], axis=1, keepdims=True) + 1e-9)
    lk = [labels[i] for i in keep]
    sims = Xk @ Xk.T
    np.fill_diagonal(sims, -1e9)
    nn = np.argsort(-sims, axis=1)[:, :k]
    correct = sum(max(set(v := [lk[j] for j in row]), key=v.count) == lk[i] for i, row in enumerate(nn))
    return correct / len(keep)


def evaluate(emit_val: np.ndarray, target_val: np.ndarray, label_cols: dict[str, list[str]],
             bootstrap_val: Optional[np.ndarray], select: str) -> dict[str, Any]:
    """Compute all val signals + the scalar `score` used for selection/early-stop."""
    out: dict[str, Any] = {}
    out.update(retrieval_mrr(emit_val, target_val))
    out["collapse"] = collapse_score(emit_val)
    have_labels = len(label_cols) > 0
    if have_labels:
        beaten, purities = 0, []
        for name, labels in label_cols.items():
            pe = knn_purity(emit_val, labels)
            out[f"purity/{name}"] = pe
            if pe == pe:
                purities.append(pe)
                if bootstrap_val is not None:
                    pb = knn_purity(bootstrap_val, labels)
                    out[f"purity_bootstrap/{name}"] = pb
                    beaten += int(pe > pb)
        out["purity_mean"] = float(np.mean(purities)) if purities else float("nan")
        out["beats_bootstrap"] = beaten

    mode = select
    if mode == "auto":
        mode = "purity" if have_labels else "retrieval"
    if mode == "purity" and have_labels:
        score = out.get("beats_bootstrap", 0) + (out.get("purity_mean", 0.0) or 0.0)
    elif mode == "loss":
        score = -out["collapse"]                       # discouraged; at least penalize collapse
    else:
        score = out["mrr"]
    # collapse guard: a fully collapsed geometry is never "best"
    if out["collapse"] > 0.95:
        score = -1.0
    out["score"] = float(score)
    out["select_mode"] = mode
    return out
