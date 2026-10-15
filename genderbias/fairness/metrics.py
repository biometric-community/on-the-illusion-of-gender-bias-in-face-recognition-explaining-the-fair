"""Attribute decorrelation + gender-gap / vanishing-bias search (lab twin)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    """X: (N, D) binary/float attributes → DxD absolute Pearson corr."""
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = X / std
    c = (X.T @ X) / max(X.shape[0] - 1, 1)
    return np.clip(c, -1.0, 1.0)


def cluster_correlated(names: Sequence[str], corr: np.ndarray, thresh: float = 0.5) -> List[List[str]]:
    """Greedy merge of attribute pairs with |corr| >= thresh (paper-style clustering)."""
    d = len(names)
    parent = list(range(d))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((abs(corr[i, j]), i, j))
    pairs.sort(reverse=True)
    for a, i, j in pairs:
        if a < thresh:
            break
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups: Dict[int, List[str]] = {}
    for i, n in enumerate(names):
        groups.setdefault(find(i), []).append(n)
    return list(groups.values())


def verification_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    pred = (scores >= threshold).astype(np.int32)
    return float((pred == labels).mean()) if len(labels) else float("nan")


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_t, best_acc = 0.0, -1.0
    for t in thresholds:
        acc = verification_accuracy(scores, labels, t)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def gender_gap(
    scores: np.ndarray,
    labels: np.ndarray,
    gender_pair: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    gender_pair: 0=female-female, 1=male-male, 2=mixed (ignored for gap).
    Gap = Acc_MM - Acc_FF (positive ⇒ male-male easier).
    """
    ff = gender_pair == 0
    mm = gender_pair == 1
    acc_ff = verification_accuracy(scores[ff], labels[ff], threshold) if ff.any() else float("nan")
    acc_mm = verification_accuracy(scores[mm], labels[mm], threshold) if mm.any() else float("nan")
    acc_all = verification_accuracy(scores, labels, threshold)
    return {
        "acc_all": acc_all,
        "acc_ff": acc_ff,
        "acc_mm": acc_mm,
        "gap_mm_minus_ff": float(acc_mm - acc_ff) if np.isfinite(acc_ff) and np.isfinite(acc_mm) else float("nan"),
        "n_ff": int(ff.sum()),
        "n_mm": int(mm.sum()),
    }


def absolute_fairness_impact(gap_base: float, gap_filtered: float) -> float:
    """How much the filter reduces |gap| (paper absolute fairness proxy)."""
    return float(abs(gap_base) - abs(gap_filtered))


def search_vanishing_filters(
    scores: np.ndarray,
    labels: np.ndarray,
    gender_pair: np.ndarray,
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    nd_names: Sequence[str],
    threshold: float,
    min_pairs: int = 40,
    top_k: int = 10,
) -> List[Dict]:
    """
    Unsupervised search: require both images share attr i==1 and attr j==1 (or both 0).
    Rank by reduction in |gender gap|.
    attr_*: (N_pairs, D_nd) binary attributes of image A / B in each pair.
    """
    gap0 = gender_gap(scores, labels, gender_pair, threshold)
    base = gap0["gap_mm_minus_ff"]
    d = len(nd_names)
    results = []
    # single-attr and pairwise share-positive filters
    for i in range(d):
        for mode, name in (("pos", f"{nd_names[i]}=1"), ("neg", f"{nd_names[i]}=0")):
            if mode == "pos":
                mask = (attr_a[:, i] == 1) & (attr_b[:, i] == 1)
            else:
                mask = (attr_a[:, i] == 0) & (attr_b[:, i] == 0)
            if int(mask.sum()) < min_pairs:
                continue
            g = gender_gap(scores[mask], labels[mask], gender_pair[mask], threshold)
            if not np.isfinite(g["gap_mm_minus_ff"]):
                continue
            results.append(
                {
                    "filter": name,
                    "n_pairs": int(mask.sum()),
                    "gap": g["gap_mm_minus_ff"],
                    "acc_ff": g["acc_ff"],
                    "acc_mm": g["acc_mm"],
                    "afi": absolute_fairness_impact(base, g["gap_mm_minus_ff"]),
                }
            )
    # top pairwise combinations (share both attrs = 1)
    for i in range(d):
        for j in range(i + 1, d):
            mask = (
                (attr_a[:, i] == 1)
                & (attr_b[:, i] == 1)
                & (attr_a[:, j] == 1)
                & (attr_b[:, j] == 1)
            )
            if int(mask.sum()) < min_pairs:
                continue
            g = gender_gap(scores[mask], labels[mask], gender_pair[mask], threshold)
            if not np.isfinite(g["gap_mm_minus_ff"]):
                continue
            results.append(
                {
                    "filter": f"{nd_names[i]}=1 & {nd_names[j]}=1",
                    "n_pairs": int(mask.sum()),
                    "gap": g["gap_mm_minus_ff"],
                    "acc_ff": g["acc_ff"],
                    "acc_mm": g["acc_mm"],
                    "afi": absolute_fairness_impact(base, g["gap_mm_minus_ff"]),
                }
            )
    results.sort(key=lambda r: (-r["afi"], abs(r["gap"])))
    return [{"baseline_gap": base, **gap0}] + results[:top_k]
