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
    """Algorithm 3: repeatedly merge the two clusters with the highest mean absolute Pearson
    correlation between their members, until no pair of clusters reaches `thresh`."""
    absr = np.abs(np.asarray(corr, dtype=np.float64))
    clusters: List[List[int]] = [[i] for i in range(len(names))]
    while len(clusters) > 1:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                link = float(absr[np.ix_(clusters[a], clusters[b])].mean())
                if link > best:
                    best, pair = link, (a, b)
        if best < thresh:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return [[names[i] for i in c] for c in clusters]


def cluster_attributes(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    names: Sequence[str],
    clusters: Sequence[Sequence[str]],
    corr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Sec. III-C2: harmonise each cluster into one binary attribute. Members that are
    negatively correlated with the cluster's first attribute are flipped, then the cluster
    is set when most of its members are."""
    index = {n: i for i, n in enumerate(names)}
    cols_a, cols_b, cluster_names = [], [], []
    for cluster in clusters:
        idx = [index[n] for n in cluster]
        flip = np.array([corr[idx[0], j] < 0 for j in idx])
        for attrs, cols in ((attr_a, cols_a), (attr_b, cols_b)):
            vals = attrs[:, idx].astype(np.float64)
            vals[:, flip] = 1.0 - vals[:, flip]
            cols.append((vals.mean(axis=1) >= 0.5).astype(np.int8))
        cluster_names.append("+".join(cluster))
    return np.stack(cols_a, axis=1), np.stack(cols_b, axis=1), cluster_names


def gini(values: Sequence[float]) -> float:
    """Eq. 2: Gini coefficient adjusted for the number of groups."""
    x = np.asarray(values, dtype=np.float64)
    n, mean = len(x), x.mean()
    if n < 2 or mean == 0:
        return 0.0
    return float(n / (n - 1) * np.abs(x[:, None] - x[None, :]).sum() / (2 * n**2 * mean))


def igarbe(fmr_by_group: Sequence[float], fnmr_by_group: Sequence[float], alpha: float = 0.5) -> float:
    """Eqs. 1 and 3: iGARBE = 1 - (alpha * G(FMR) + (1 - alpha) * G(FNMR)); 1 is perfectly fair."""
    return float(1.0 - (alpha * gini(fmr_by_group) + (1.0 - alpha) * gini(fnmr_by_group)))


def threshold_at_fmr(scores: np.ndarray, labels: np.ndarray, fmr: float = 1e-3) -> float:
    """Smallest threshold whose false match rate does not exceed `fmr`."""
    neg = np.sort(scores[labels == 0])[::-1]
    k = int(np.floor(fmr * len(neg)))
    if k >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))


def _error_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    fnmr = float((genuine < threshold).mean()) if len(genuine) else float("nan")
    fmr = float((impostor >= threshold).mean()) if len(impostor) else float("nan")
    return fmr, fnmr


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
    fmr_ff, fnmr_ff = _error_rates(scores[ff], labels[ff], threshold)
    fmr_mm, fnmr_mm = _error_rates(scores[mm], labels[mm], threshold)
    return {
        "fmr_ff": fmr_ff,
        "fnmr_ff": fnmr_ff,
        "fmr_mm": fmr_mm,
        "fnmr_mm": fnmr_mm,
        "igarbe": igarbe([fmr_ff, fmr_mm], [fnmr_ff, fnmr_mm]),
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
    Sec. III-B: rank by iGARBE at the fixed-FMR threshold (1 is perfectly fair), then by
    |gender gap|. Each filter also reports both genders' FNMR, so a gap closed by raising
    the male error rate is visible.
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
                    "fnmr_ff": g["fnmr_ff"],
                    "fnmr_mm": g["fnmr_mm"],
                    "igarbe": g["igarbe"],
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
                    "fnmr_ff": g["fnmr_ff"],
                    "fnmr_mm": g["fnmr_mm"],
                    "igarbe": g["igarbe"],
                    "afi": absolute_fairness_impact(base, g["gap_mm_minus_ff"]),
                }
            )
    results.sort(key=lambda r: (-(r["igarbe"] if np.isfinite(r["igarbe"]) else -np.inf), abs(r["gap"])))
    return [{"baseline_gap": base, **gap0}] + results[:top_k]
