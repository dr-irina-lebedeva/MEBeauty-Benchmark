"""Point and distribution metrics. Every published number passes through here.

Point metrics (Pearson, Spearman, MAE, RMSE) apply to every method.
Distribution metrics apply only when a method predicts a distribution *and*
the dataset ships a real histogram -- never one reconstructed from a mean.

`paired_bootstrap_difference` is the test that says whether a gap between two
methods is real; on a 250-image split, differences below ~0.04 are not.
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
    """Score predicted rating distributions."""
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
    """Bootstrap 95% interval for Pearson correlation."""
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
    """Is method A's correlation really higher than method B's?"""
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


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    predicted_distributions: np.ndarray | None = None,
    true_distributions: np.ndarray | None = None,
) -> dict[str, float]:
    """Every metric a result carries, in one flat dictionary."""
    scores = point_metrics(np.asarray(predicted), np.asarray(actual)).as_dict()
    if predicted_distributions is None or true_distributions is None:
        return scores
    distribution = distribution_metrics(
        np.asarray(predicted_distributions), np.asarray(true_distributions)
    ).as_dict()
    # Both metric sets report `n`; they are over the same images, so keep one.
    distribution.pop("n", None)
    return scores | distribution
