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


def knn_purity(X: np.ndarray, labels: list[str], k: int = 5) -> float:
    """Leave-one-out kNN majority-vote accuracy by label: does the EMITTED geometry organize into separate regions by
    this attribute? The selection signal for label-shaping (SupCon) runs — retrieval MRR measures event IDENTITY and is
    suppressed when same-label items are pulled together, so it can't select a stage-separated checkpoint; this can.
    Labels ''/'unknown'/'none'/'nan' are dropped. Returns the fraction of held-out items whose kNN majority matches."""
    keep = [i for i, l in enumerate(labels) if str(l).strip().lower() not in ("", "unknown", "none", "nan")]
    if len(keep) < k + 1:
        return 0.0
    Xk = X[keep]
    Xn = Xk / (np.linalg.norm(Xk, axis=1, keepdims=True) + 1e-9)
    lab = [labels[i] for i in keep]
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1e9)                                # leave-one-out: never vote for yourself
    nbr = np.argsort(-sims, axis=1)[:, :k]                      # k nearest by cosine
    correct = 0
    for i in range(len(lab)):
        votes: dict[str, int] = {}
        for j in nbr[i]:
            votes[lab[j]] = votes.get(lab[j], 0) + 1
        if max(votes, key=lambda kk: votes[kk]) == lab[i]:
            correct += 1
    return float(correct / len(lab))
