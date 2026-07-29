import numpy as np
import pytest

from mebeauty_benchmark.benchmark.metrics import (
    correlation_ci,
    distribution_metrics,
    paired_bootstrap_difference,
    point_metrics,
)


def test_perfect_prediction_scores_perfectly():
    actual = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    m = point_metrics(actual.copy(), actual)

    assert m.pearson == pytest.approx(1.0)
    assert m.spearman == pytest.approx(1.0)
    assert m.mae == pytest.approx(0.0)
    assert m.rmse == pytest.approx(0.0)


def test_a_constant_prediction_scores_zero_not_nan():
    # A model that outputs the same number for everything has no correlation.
    # Returning NaN would let it vanish from a results table instead of
    # appearing as the failure it is.
    actual = np.array([1.0, 4.0, 6.0, 9.0])

    m = point_metrics(np.full(4, 5.0), actual)

    assert m.pearson == 0.0
    assert m.spearman == 0.0
    assert np.isfinite(m.mae)


def test_spearman_ignores_monotone_rescaling_but_pearson_does_not():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    squashed = actual**3  # same order, very different spacing

    m = point_metrics(squashed, actual)

    assert m.spearman == pytest.approx(1.0)
    assert m.pearson < 0.99


def test_spearman_handles_ties_by_averaging_ranks():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    tied = np.array([5.0, 5.0, 9.0, 9.0])

    m = point_metrics(tied, actual)

    assert np.isfinite(m.spearman)
    assert m.spearman == pytest.approx(0.894, abs=0.01)


def test_mae_and_rmse_differ_when_one_error_is_large():
    # RMSE punishes the single big miss; MAE does not. Reporting only one
    # hides which kind of mistake a model makes.
    actual = np.zeros(10)
    predicted = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0])

    m = point_metrics(predicted, actual)

    assert m.mae == pytest.approx(1.0)
    assert m.rmse == pytest.approx(np.sqrt(10))


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="does not match"):
        point_metrics(np.zeros(3), np.zeros(4))


def test_non_finite_predictions_are_rejected():
    # A NaN prediction must fail loudly, not silently poison the mean.
    with pytest.raises(ValueError, match="NaN or infinity"):
        point_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        point_metrics(np.array([]), np.array([]))


def test_identical_distributions_score_at_the_optimum():
    actual = np.array([[0.1, 0.2, 0.4, 0.3], [0.25, 0.25, 0.25, 0.25]])

    m = distribution_metrics(actual.copy(), actual)

    assert m.chebyshev == pytest.approx(0.0, abs=1e-9)
    assert m.kl == pytest.approx(0.0, abs=1e-9)
    assert m.cosine == pytest.approx(1.0, abs=1e-9)
    assert m.intersection == pytest.approx(1.0, abs=1e-9)


def test_distributions_are_renormalised_before_scoring():
    # Counts and probabilities of the same shape must score identically.
    counts = np.array([[2.0, 4.0, 8.0, 6.0]])
    probabilities = counts / counts.sum()

    from_counts = distribution_metrics(counts, probabilities)

    assert from_counts.kl == pytest.approx(0.0, abs=1e-9)
    assert from_counts.intersection == pytest.approx(1.0, abs=1e-9)


def test_a_zero_mass_distribution_is_rejected():
    with pytest.raises(ValueError, match="positive total mass"):
        distribution_metrics(np.array([[0.0, 0.0]]), np.array([[0.5, 0.5]]))


def test_disagreeing_distributions_score_worse_than_agreeing_ones():
    actual = np.array([[0.7, 0.2, 0.1]])
    close = np.array([[0.6, 0.3, 0.1]])
    far = np.array([[0.1, 0.2, 0.7]])

    good = distribution_metrics(close, actual)
    bad = distribution_metrics(far, actual)

    assert good.chebyshev < bad.chebyshev
    assert good.kl < bad.kl
    assert good.cosine > bad.cosine


def test_correlation_interval_contains_the_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=400)
    predicted = actual + rng.normal(scale=0.5, size=400)
    point = point_metrics(predicted, actual).pearson

    low, high = correlation_ci(predicted, actual, iterations=400, seed=1)
    low_small, high_small = correlation_ci(
        predicted[:40], actual[:40], iterations=400, seed=1
    )

    assert low <= point <= high
    assert (high - low) < (high_small - low_small)


def test_paired_comparison_detects_a_real_difference():
    rng = np.random.default_rng(2)
    actual = rng.normal(size=300)
    good = actual + rng.normal(scale=0.3, size=300)
    poor = actual + rng.normal(scale=2.0, size=300)

    result = paired_bootstrap_difference(good, poor, actual, iterations=400, seed=3)

    assert result["difference"] > 0
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.05


def test_paired_comparison_reports_no_difference_between_equal_methods():
    rng = np.random.default_rng(4)
    actual = rng.normal(size=300)
    a = actual + rng.normal(scale=0.5, size=300)
    b = actual + rng.normal(scale=0.5, size=300)

    result = paired_bootstrap_difference(a, b, actual, iterations=400, seed=5)

    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.05
