"""Metrics. Every number in the results table comes through here."""

from __future__ import annotations

import numpy as np
import pytest

from fbp_benchmark.metrics import (
    correlation_ci,
    distribution_metrics,
    evaluate,
    paired_bootstrap_difference,
    point_metrics,
)


def test_perfect_prediction_scores_perfectly():
    y = np.array([1.0, 4.0, 7.0, 9.0])
    m = point_metrics(y, y).as_dict()
    assert m["PC"] == pytest.approx(1.0)
    assert m["SROCC"] == pytest.approx(1.0)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)


def test_a_constant_prediction_has_zero_correlation():
    # The mean baseline relies on this being exactly 0, not merely small.
    y = np.array([2.0, 5.0, 8.0, 3.0])
    m = point_metrics(np.full_like(y, y.mean()), y).as_dict()
    assert m["PC"] == pytest.approx(0.0, abs=1e-9)


def test_spearman_ignores_monotone_rescaling_but_pearson_does_not():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    squashed = np.exp(y)
    m = point_metrics(squashed, y).as_dict()
    assert m["SROCC"] == pytest.approx(1.0)
    assert m["PC"] < 1.0


def test_mae_and_rmse_differ_when_one_error_is_large():
    # RMSE must punish the outlier harder; if they match, one is miscomputed.
    y = np.zeros(4)
    predicted = np.array([0.0, 0.0, 0.0, 4.0])
    m = point_metrics(predicted, y).as_dict()
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(2.0)


def test_identical_distributions_score_as_agreement():
    p = np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]])
    m = distribution_metrics(p, p).as_dict()
    assert m["KL"] == pytest.approx(0.0, abs=1e-9)
    assert m["Chebyshev"] == pytest.approx(0.0, abs=1e-9)
    assert m["Cosine"] == pytest.approx(1.0, abs=1e-6)
    assert m["Intersection"] == pytest.approx(1.0, abs=1e-6)


def test_unnormalised_distributions_are_renormalised_not_rejected():
    # A head that emits logits summing to 3 is a model bug, not a reason to
    # score it against a differently-scaled target.
    p = np.array([[1.0, 1.0, 1.0]])
    q = np.array([[3.0, 3.0, 3.0]])
    assert distribution_metrics(p, q).as_dict()["KL"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_omits_distribution_metrics_for_point_methods():
    y = np.array([3.0, 5.0, 7.0])
    scores = evaluate(y, y)
    assert "PC" in scores
    assert "KL" not in scores


def test_evaluate_includes_distribution_metrics_when_both_sides_exist():
    y = np.array([2.0, 3.0])
    d = np.array([[0.5, 0.5], [0.2, 0.8]])
    scores = evaluate(y, y, predicted_distributions=d, true_distributions=d)
    assert "KL" in scores
    # Both metric sets define `n`; exactly one must survive.
    assert isinstance(scores["n"], int)


def test_correlation_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    predicted = y + rng.normal(scale=0.5, size=200)
    low, high = correlation_ci(predicted, y, iterations=500, seed=0)
    observed = point_metrics(predicted, y).pearson
    assert low <= observed <= high


def test_paired_bootstrap_finds_no_difference_between_a_method_and_itself():
    rng = np.random.default_rng(1)
    y = rng.normal(size=150)
    p = y + rng.normal(scale=0.4, size=150)
    result = paired_bootstrap_difference(p, p, y, iterations=500, seed=0)
    assert result["difference"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] > 0.5


def test_paired_bootstrap_detects_a_clearly_better_method():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    good = y + rng.normal(scale=0.2, size=400)
    bad = y + rng.normal(scale=2.0, size=400)
    result = paired_bootstrap_difference(good, bad, y, iterations=1000, seed=0)
    assert result["difference"] > 0
    assert result["p_value"] < 0.05
    assert result["ci_low"] > 0
