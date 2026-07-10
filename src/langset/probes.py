"""World-model property probes — is a trained multi-latent langset model a *calibrated* JEPA world model?

A JEPA world model doesn't just emit *a* latent per step; it emits a **superposition** — a distribution over
next states — and the question that matters is whether that distribution is *honest*: does the model's own
uncertainty grow when the future is genuinely more open, and are the predicted states actually recoverable?

These two probes read straight off `rollout(..., return_soft=True)` and answer exactly that, for ANY multi-latent
model — not just the maze example this graduated from:

  * `calibration_corr`   — CALIBRATION. corr(native emission entropy, ground-truth set-cardinality). Positive =
                           the single emitted latent widens when the true future set is wider, so it carries a
                           *calibrated* superposition rather than one guess. This is the headline world-model signal.
  * `linear_decodability` — DECODABILITY. Can a linear probe recover a per-emission label (e.g. a count, a class)
                           from the emitted latent, on a GROUP-DISJOINT split (no leakage across the train/test
                           cut)? Measures whether the predicted state is linearly present in the latent at all.

Both are deliberately model-agnostic: you pass in the arrays that `rollout(return_soft=True)` already gives you
(`ent`, the emitted/soft latents) plus your task's ground-truth labels and a per-emission group id (so the split
never leaks one trajectory across both sides). scikit-learn and scipy are imported lazily, so importing langset
never requires them — `pip install "langset[probes]"` (or bring your own sklearn/scipy) only when you probe.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _require(mod: str, pkg: str):
    try:
        return __import__(mod, fromlist=["_"])
    except ImportError as e:  # pragma: no cover - trivial guard
        raise ImportError(
            f"langset.probes needs {pkg} — `pip install \"langset[probes]\"` (or `pip install {pkg}`)."
        ) from e


def calibration_corr(entropy: Sequence[float], cardinality: Sequence[int],
                     groups: Optional[Sequence] = None, test_groups: Optional[set] = None) -> Optional[float]:
    """Superposition CALIBRATION: Pearson corr between the emitted latent's native entropy and the ground-truth
    number of possible next states at that emission.

    entropy      per-emission native FSQ entropy — the `ent` array from `rollout(..., return_soft=True)`.
    cardinality  per-emission ground-truth set size (how many next states were actually possible there).
    groups       optional per-emission group id (e.g. trajectory/maze id). If given with `test_groups`, the
                 correlation is computed on the held-out group split only, so it matches a decodability probe's
                 test set and never rewards memorised trajectories.
    test_groups  the subset of group ids to evaluate on (the rest are ignored). Ignored if `groups` is None.

    Returns the correlation in [-1, 1] (positive = calibrated), or None if either side has zero variance on the
    evaluated subset (correlation undefined — e.g. every emission had the same cardinality)."""
    pearsonr = _require("scipy.stats", "scipy").pearsonr
    ent = np.asarray(entropy, dtype=float)
    k = np.asarray(cardinality, dtype=float)
    if groups is not None and test_groups is not None:
        keep = np.array([g in test_groups for g in groups], dtype=bool)
        ent, k = ent[keep], k[keep]
    if ent.size < 2 or np.std(ent) == 0 or np.std(k) == 0:
        return None
    return float(pearsonr(ent, k)[0])


def linear_decodability(latents: np.ndarray, labels: Sequence, groups: Sequence,
                        test_frac: float = 0.4, seed: int = 0, balanced: bool = True) -> dict:
    """Linear DECODABILITY of a per-emission label from the emitted latent, on a GROUP-DISJOINT split.

    latents   [N, d] emitted latents (the `lat`/`soft_lat` rows from `rollout`, one per emission).
    labels    length-N per-emission targets (int class ids — a count, a solvable/unsolvable flag, a stage, ...).
    groups    length-N group id per emission (trajectory/maze id). The train/test cut is made BY GROUP, so no
              trajectory appears on both sides — the number a probe reports is genuine generalisation, not leakage.
    test_frac fraction of GROUPS held out for test. seed fixes the group shuffle.
    balanced  fit the logistic probe with class_weight='balanced' (robust to skewed label frequencies).

    Returns {'acc', 'bal_acc', 'baseline_majority', 'n_train', 'n_test', 'n_classes'}. bal_acc is the mean
    per-class recall — compare it to baseline_majority to see if the latent carries the label above chance."""
    skl = _require("sklearn.linear_model", "scikit-learn")
    metrics = _require("sklearn.metrics", "scikit-learn")
    X = np.asarray(latents, dtype=float)
    y = np.asarray(labels)
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    order = uniq.copy()
    rng.shuffle(order)
    n_test = max(1, int(round(len(order) * test_frac)))
    test_g = set(order[:n_test].tolist())
    te = np.array([gi in test_g for gi in g], dtype=bool)
    tr = ~te
    if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
        return {"acc": None, "bal_acc": None, "baseline_majority": None,
                "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_classes": int(len(np.unique(y)))}
    yt = y[te]
    vals, counts = np.unique(yt, return_counts=True)
    baseline = float(counts.max() / counts.sum())          # majority-class accuracy on the test set
    clf = skl.LogisticRegression(max_iter=2000,
                                 class_weight="balanced" if balanced else None).fit(X[tr], y[tr])
    pr = clf.predict(X[te])
    return {"acc": round(float((pr == yt).mean()), 4),
            "bal_acc": round(float(metrics.balanced_accuracy_score(yt, pr)), 4),
            "baseline_majority": round(baseline, 4),
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_classes": int(len(vals))}
