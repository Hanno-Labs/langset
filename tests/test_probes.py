"""Unit tests for langset.probes — the graduated world-model property probes."""

import numpy as np
import pytest

from langset import calibration_corr, linear_decodability


def test_calibration_corr_positive_when_entropy_tracks_cardinality():
    # entropy rises with cardinality -> strong positive correlation
    ent = [0.1, 0.15, 0.2, 0.9, 0.95, 1.0]
    card = [1, 1, 1, 3, 3, 3]
    c = calibration_corr(ent, card)
    assert c is not None and c > 0.9


def test_calibration_corr_negative_when_anticorrelated():
    ent = [1.0, 0.95, 0.9, 0.2, 0.15, 0.1]
    card = [1, 1, 1, 3, 3, 3]
    assert calibration_corr(ent, card) < 0


def test_calibration_corr_none_on_zero_variance():
    # every cardinality identical -> correlation undefined -> None (not a crash)
    assert calibration_corr([0.1, 0.5, 0.9], [2, 2, 2]) is None
    # fewer than 2 points -> None
    assert calibration_corr([0.3], [1]) is None


def test_calibration_corr_group_restriction_matches_manual():
    ent = [0.1, 0.2, 0.9, 1.0, 0.15, 0.95]
    card = [1, 1, 3, 3, 1, 3]
    groups = [0, 0, 1, 1, 2, 2]
    got = calibration_corr(ent, card, groups=groups, test_groups={1, 2})
    # manual: keep indices where group in {1,2} -> entries 2,3,4,5
    from scipy.stats import pearsonr

    keep = [2, 3, 4, 5]
    want = pearsonr([ent[i] for i in keep], [float(card[i]) for i in keep])[0]
    assert got == pytest.approx(want)


def test_linear_decodability_separable_labels():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0, 0, 0], 0.1, (10, 4)), rng.normal([4, 4, 4, 4], 0.1, (10, 4))])
    y = np.array([0] * 10 + [1] * 10)
    groups = np.arange(20)  # one emission per group -> split is by row
    out = linear_decodability(X, y, groups, test_frac=0.4, seed=0)
    assert out["acc"] == 1.0 and out["bal_acc"] == 1.0
    assert out["baseline_majority"] == pytest.approx(0.5, abs=0.2)
    assert out["n_test"] > 0 and out["n_classes"] == 2


def test_linear_decodability_group_disjoint_split():
    # two groups, each all-one-label: held-out group's label is UNSEEN in train -> probe can't cheat via leakage
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (10, 4)), rng.normal(0, 1, (10, 4))])
    y = np.array([0] * 10 + [1] * 10)
    groups = np.array([0] * 10 + [1] * 10)  # label perfectly confounded with group
    out = linear_decodability(X, y, groups, test_frac=0.5, seed=0)
    # only one label present in train -> undefined probe -> graceful None, no crash
    assert out["acc"] is None
    assert out["n_train"] > 0 and out["n_test"] > 0


def test_n_classes_is_from_full_labels_not_test_set():
    # Copilot #1: a class absent from the held-out groups must NOT shrink n_classes, and both the normal and
    # early-return paths must report the same count (from the full label vector).
    rng = np.random.default_rng(3)
    # 3 groups: two carry classes {0,1}, one (group 2) is the ONLY carrier of class 2. Hold out only group 0
    # so class 2 stays in TRAIN and is absent from the test set -> test-set unique classes = {0,1} (2), full = 3.
    X = rng.normal(0, 1, (30, 4))
    y = np.array([0, 1] * 5 + [0, 1] * 5 + [2] * 10)
    groups = np.array([0] * 10 + [1] * 10 + [2] * 10)
    out = linear_decodability(X, y, groups, test_groups={0})
    assert out["acc"] is not None  # normal path
    assert out["n_classes"] == 3  # full label set, not the 2 present in the test split


def test_explicit_test_groups_gives_one_shared_cut():
    # Copilot #2: passing an explicit test_groups must reproduce EXACTLY the internal-shuffle split for the
    # same seed (so a caller can compute one cut and share it), and honor an arbitrary caller-chosen cut.
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal([0, 0, 0, 0], 0.1, (15, 4)), rng.normal([4, 4, 4, 4], 0.1, (15, 4))])
    y = np.array([0, 1, 0, 1, 0] * 3 + [1, 0, 1, 0, 1] * 3)
    groups = np.repeat(
        np.arange(6), 5
    )  # 6 groups, mixed labels within each -> train always sees both
    # reconstruct the split the default path would pick, then feed it back explicitly
    order = np.unique(groups).copy()
    np.random.default_rng(0).shuffle(order)
    picked = set(order[: max(1, int(round(len(order) * 0.4)))].tolist())
    internal = linear_decodability(X, y, groups, test_frac=0.4, seed=0)
    explicit = linear_decodability(X, y, groups, test_groups=picked)
    assert internal["n_test"] == explicit["n_test"] and internal["n_train"] == explicit["n_train"]
    assert internal["acc"] == explicit["acc"]
    # an arbitrary explicit cut is honored verbatim (group 5 held out -> its 5 rows are the test set)
    one = linear_decodability(X, y, groups, test_groups={5})
    assert one["n_test"] == 5
