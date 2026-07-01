"""Validation metrics for early-stop. The hard-won rule: NEVER select on training loss — a contrastive objective
can minimize it by COLLAPSING the geometry. Selection scores held-out input-view <-> target-view retrieval and
held-out reconstruction, with a hard collapse penalty (see Trainer)."""
from __future__ import annotations

import numpy as np


def retrieval_mrr(pred: np.ndarray, bank: np.ndarray) -> dict[str, float]:
    """Each pred[i] should retrieve target bank[i] among the val set (rank-based, scale-free, collapse-sensitive)."""
    sims = pred @ bank.T
    order = np.argsort(-sims, axis=1)
    truth = np.arange(len(pred))
    rank = np.array([int(np.where(order[i] == truth[i])[0][0]) for i in range(len(truth))])
    return {"r@1": float((rank == 0).mean()), "mrr": float((1.0 / (rank + 1)).mean())}


def collapse_score(X: np.ndarray) -> float:
    """Mean off-diagonal cosine of (normalized) emissions. ->1 means collapsed; healthy spaces sit low."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    s = Xn @ Xn.T
    np.fill_diagonal(s, np.nan)
    return float(np.nanmean(s))
