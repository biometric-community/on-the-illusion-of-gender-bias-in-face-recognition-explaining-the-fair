"""Fairness metric, attribute decorrelation and the gender gap, on synthetic inputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import genderbias.fairness.metrics as metrics
from genderbias.eval import best_acc_eer
from genderbias.fairness.metrics import cluster_correlated, gender_gap, pearson_corr_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_correlations_are_pearson_coefficients():
    rng = np.random.default_rng(0)
    x = (rng.random((500, 4)) > 0.5).astype(np.int8)
    x[:, 1] = x[:, 0]
    assert np.allclose(pearson_corr_matrix(x), np.corrcoef(x.T), atol=1e-2)


def test_clusters_merge_by_mean_correlation_between_clusters():
    """Sec. III-C1, Algorithm 3: once a and b form a cluster, c joins only if its mean absolute
    correlation with the cluster's members is high enough, so one strong link is not enough."""
    corr = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.6], [0.1, 0.6, 1.0]])
    clusters = sorted(sorted(c) for c in cluster_correlated(["a", "b", "c"], corr, thresh=0.5))
    assert clusters == [["a", "b"], ["c"]]


def test_negative_correlations_also_cluster():
    """Sec. III-C1: absolute correlations are used, so strongly anti-correlated attributes merge."""
    corr = np.array([[1.0, -0.9], [-0.9, 1.0]])
    assert len(cluster_correlated(["a", "b"], corr, thresh=0.5)) == 1


def test_gender_gap_compares_male_and_female_pairs():
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 0, 1, 1, 0, 0])
    groups = np.array([1, 1, 0, 0, 0, 0])
    scores[3] = 0.1  # one female genuine pair is missed
    gap = gender_gap(scores, labels, groups, threshold=0.5)
    assert gap["acc_mm"] == pytest.approx(1.0)
    assert gap["acc_ff"] == pytest.approx(0.75)
    assert gap["gap_mm_minus_ff"] == pytest.approx(0.25)


def test_igarbe_is_implemented_and_rewards_equal_error_rates():
    """Eqs. (1)-(3): iGARBE is 1 for equal FMR and FNMR across groups and drops below 1 otherwise."""
    igarbe = getattr(metrics, "igarbe", None)
    assert igarbe is not None, "iGARBE (Eqs. 1-3) is not implemented"
    assert igarbe([0.001, 0.001], [0.05, 0.05]) == pytest.approx(1.0)
    assert igarbe([0.001, 0.001], [0.02, 0.08]) < 1.0


def test_gap_is_measured_at_a_fixed_false_match_rate():
    """Sec. IV-C: FNMRs are reported at the threshold where FMR = 1e-3, not at the most
    accurate threshold."""
    source = (ROOT / "genderbias" / "eval.py").read_text(encoding="utf-8")
    assert "best_threshold(" not in source


def test_filter_search_uses_the_decorrelated_clusters():
    """Sec. III-C: the clusters are used as standalone attributes in all further analyses."""
    source = (ROOT / "genderbias" / "eval.py").read_text(encoding="utf-8")
    call = re.search(r"search_vanishing_filters\((.*?)\)\n", source, re.S)
    assert call and "clusters" in call.group(1)


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
