"""Evaluation metrics for facial beauty prediction.

Every number a benchmark reports depends on these, so each one is defined
explicitly rather than deferred to whatever a library happens to do.

**Point-prediction metrics.** The facial-beauty literature reports Pearson
correlation (PC), MAE and RMSE almost universally; Spearman (SROCC) appears
in ranking-oriented work. All four ship, because they answer different
questions: PC and SROCC measure whether a model orders faces correctly,
MAE and RMSE whether it lands on the right value. A model can be excellent
at one and poor at the other.

**Distribution metrics.** MEBeauty ships per-image rating distributions, so
label-distribution methods (Ren & Geng 2017 and descendants) can be scored on
what they actually predict instead of being collapsed to a mean first. The six
measures here are the standard set from the label-distribution-learning
literature: Chebyshev, Clark, Canberra and KL are distances (lower is better);
cosine and intersection are similarities (higher is better).

**On correlation with n=523.** A test-set PC is a sample statistic, not a
constant. Two methods differing by 0.01 are not distinguishable at this size.
`correlation_ci` gives a bootstrap interval so comparisons can be made
honestly, and `paired_bootstrap_difference` tests two methods on the *same*
images, which is far more sensitive than comparing two independent intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Guard against division by zero in the distribution metrics without
#: perturbing any real value.
EPSILON = 1e-12


@dataclass(frozen=True)
class PointMetrics:
    """Scores for a model that predicts one number per image."""

    pearson: float
    spearman: float
    mae: float
    rmse: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "PC": round(self.pearson, 4),
            "SROCC": round(self.spearman, 4),
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
            "n": self.n,
        }


@dataclass(frozen=True)
class DistributionMetrics:
    """Scores for a model that predicts a rating distribution per image."""

    chebyshev: float
    clark: float
    canberra: float
    kl: float
    cosine: float
    intersection: float
    n: int
    lower_is_better: tuple[str, ...] = field(
        default=("Chebyshev", "Clark", "Canberra", "KL")
    )

    def as_dict(self) -> dict[str, float]:
        return {
            "Chebyshev": round(self.chebyshev, 4),
            "Clark": round(self.clark, 4),
            "Canberra": round(self.canberra, 4),
            "KL": round(self.kl, 4),
            "Cosine": round(self.cosine, 4),
            "Intersection": round(self.intersection, 4),
            "n": self.n,
        }


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort Spearman."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_centred, b_centred = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a_centred**2).sum() * (b_centred**2).sum())
    # A constant prediction has no correlation with anything; report 0 rather
    # than NaN so a degenerate model scores badly instead of disappearing from
    # the results table.
    return float((a_centred * b_centred).sum() / denominator) if denominator else 0.0


def point_metrics(predicted: np.ndarray, actual: np.ndarray) -> PointMetrics:
    """Score point predictions against ground-truth labels."""
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if predicted.shape != actual.shape:
        raise ValueError(
            f"predicted {predicted.shape} does not match actual {actual.shape}"
        )
    if predicted.size == 0:
        raise ValueError("Cannot score an empty prediction")
    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain NaN or infinity")

    difference = predicted - actual
    return PointMetrics(
        pearson=_pearson(predicted, actual),
        spearman=_pearson(_rank(predicted), _rank(actual)),
        mae=float(np.abs(difference).mean()),
        rmse=float(np.sqrt((difference**2).mean())),
        n=int(predicted.size),
    )


def _normalise(distributions: np.ndarray) -> np.ndarray:
    totals = distributions.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every distribution must have positive total mass")
    return distributions / totals


def distribution_metrics(
    predicted: np.ndarray, actual: np.ndarray
) -> DistributionMetrics:
    """Score predicted rating distributions.

    The six standard label-distribution measures. Inputs are renormalised
    first: a predicted distribution that does not sum to 1 is a model bug, not
    a reason to score it against a differently-scaled target.
    """
    predicted = _normalise(np.asarray(predicted, dtype=float))
    actual = _normalise(np.asarray(actual, dtype=float))
    if predicted.shape != actual.shape:
        raise ValueError(
            f"predicted {predicted.shape} does not match actual {actual.shape}"
        )
    if not np.isfinite(predicted).all():
        raise ValueError("Predicted distributions contain NaN or infinity")

    difference = np.abs(predicted - actual)
    total = predicted + actual

    chebyshev = difference.max(axis=1).mean()
    clark = np.sqrt(((difference / (total + EPSILON)) ** 2).sum(axis=1)).mean()
    canberra = (difference / (total + EPSILON)).sum(axis=1).mean()
    kl = (
        (actual * np.log((actual + EPSILON) / (predicted + EPSILON))).sum(axis=1).mean()
    )
    cosine = (
        (predicted * actual).sum(axis=1)
        / (np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1) + EPSILON)
    ).mean()
    intersection = np.minimum(predicted, actual).sum(axis=1).mean()

    return DistributionMetrics(
        chebyshev=float(chebyshev),
        clark=float(clark),
        canberra=float(canberra),
        kl=float(kl),
        cosine=float(cosine),
        intersection=float(intersection),
        n=int(predicted.shape[0]),
    )


def correlation_ci(
    predicted: np.ndarray,
    actual: np.ndarray,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% interval for Pearson correlation.

    Reported alongside PC because a test set of a few hundred images gives an
    interval wide enough that small differences between methods are noise. A
    table of bare correlations invites conclusions the sample cannot support.
    """
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(predicted)
    samples = np.empty(iterations)
    for step in range(iterations):
        index = rng.integers(0, n, n)
        samples[step] = _pearson(predicted[index], actual[index])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_bootstrap_difference(
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    actual: np.ndarray,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Is method A's correlation really higher than method B's?

    Resamples images and recomputes *both* correlations on the same resample,
    so the shared difficulty of the images cancels out. Comparing two
    independent confidence intervals instead would be far more conservative
    and would miss real differences.
    """
    predicted_a = np.asarray(predicted_a, dtype=float).ravel()
    predicted_b = np.asarray(predicted_b, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(actual)

    differences = np.empty(iterations)
    for step in range(iterations):
        index = rng.integers(0, n, n)
        differences[step] = _pearson(predicted_a[index], actual[index]) - _pearson(
            predicted_b[index], actual[index]
        )

    observed = _pearson(predicted_a, actual) - _pearson(predicted_b, actual)
    # Two-sided p: how often the resampled difference lands on the other side
    # of zero from the observed one.
    p = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
    return {
        "difference": round(float(observed), 4),
        "ci_low": round(float(np.percentile(differences, 2.5)), 4),
        "ci_high": round(float(np.percentile(differences, 97.5)), 4),
        "p_value": round(float(min(p, 1.0)), 4),
    }
