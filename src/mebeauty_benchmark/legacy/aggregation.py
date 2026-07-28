"""Rater-offset-adjusted image scores, and the model selection behind them.

The primary label stays the plain mean. This module produces a second,
modelled estimate for consumers who want one, and — as much as the estimate
itself — the evidence for the modelling choices.

**The model.** Raters differ in where they sit on the scale: two people can
rank a set of faces identically while one averages 7 and the other 5. With an
unbalanced design (nobody rated everything), a plain mean lets that offset
leak into the image score. So:

    rating(r, i) = quality(i) + offset(r) + noise

fitted by alternating least squares. Offsets are anchored to mean zero, which
is identifiable here because the rater x image graph is a single connected
component — verified, not assumed.

**Per-rater scale, and how it was decided.** The natural extension gives each
rater a slope too, `rating = offset(r) + scale(r) * quality(i)`, with `scale`
ridge-regularised toward 1. It was expected to overfit -- some raters have a
handful of ratings -- so the choice was left to held-out prediction rather
than taste. On this corpus affine wins all 5 folds, so it ships.

One objection was raised against it and then withdrawn on evidence, which is
worth recording because the reasoning is easy to repeat. Fitted `scale`
correlates 0.54 with how much a rater agrees with the consensus, which looks
like the agreement-based re-weighting this project rejects elsewhere. But
simulating a world where the affine model is exactly true -- one shared
quality, raters differing *only* in offset and scale, no taste differences at
all -- produces a correlation of 0.88. The coupling is mechanical: a rater
with a shallow slope necessarily tracks the consensus less closely. The
observed 0.54 is *lower* than that, so it is not evidence of circularity.

What did survive: unconstrained least squares gave a small number of raters a
negative slope, inverting their judgements. `scale` is floored at zero.

**Why not naive z-scoring.** Dividing by each rater's own standard deviation
assumes their sample was representative. Measured on this dataset, the
correlation between a rater's mean and the leave-one-rater-out quality of the
images they were assigned is 0.090 (permutation p = 0.11) — small and not
distinguishable from chance, so the offset model is safe. That diagnostic is
reported alongside the scores rather than being taken on trust, because on a
differently-assigned corpus it could easily fail.

Both fits ship. `score_adjusted` is the selected model; `score_adjusted_offset`
is the offset-only fit, kept because it is the conservative option -- a pure
location shift can never re-weight anyone's opinion, whatever the fit says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Alternating least squares stops when no image score moves more than this.
CONVERGENCE_TOLERANCE = 1e-6

#: Hard cap on iterations, so a pathological input cannot spin forever.
MAX_ITERATIONS = 200

#: Ridge strength pulling per-rater scale toward 1 in the affine model. Scale
#: is the parameter most prone to overfitting on light raters, so it is
#: regularised while the offset is not.
SCALE_RIDGE = 10.0


@dataclass(frozen=True)
class OffsetModel:
    """Fitted image qualities and rater offsets."""

    quality: np.ndarray
    offset: np.ndarray
    iterations: int
    converged: bool


@dataclass(frozen=True)
class AffineModel:
    """Fitted image qualities with per-rater offset *and* scale."""

    quality: np.ndarray
    offset: np.ndarray
    scale: np.ndarray
    iterations: int
    converged: bool


def _group_mean(values: np.ndarray, index: np.ndarray, size: int) -> np.ndarray:
    """Mean of `values` grouped by `index`; groups with no members give 0."""
    totals = np.bincount(index, weights=values, minlength=size)
    counts = np.bincount(index, minlength=size)
    return np.divide(totals, counts, out=np.zeros(size), where=counts > 0)


def fit_offset_model(
    ratings: np.ndarray,
    image_index: np.ndarray,
    rater_index: np.ndarray,
    n_images: int,
    n_raters: int,
) -> OffsetModel:
    """Fit `rating = quality(image) + offset(rater)` by alternating least squares.

    Offsets are re-centred to mean zero every iteration. Without that anchor
    the model is degenerate -- adding a constant to every quality and
    subtracting it from every offset leaves the fit unchanged -- so the
    constraint is what makes the solution unique, not a cosmetic step.
    """
    quality = _group_mean(ratings, image_index, n_images)
    offset = np.zeros(n_raters)

    converged = False
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        offset = _group_mean(ratings - quality[image_index], rater_index, n_raters)
        offset -= offset.mean()
        updated = _group_mean(ratings - offset[rater_index], image_index, n_images)
        shift = np.max(np.abs(updated - quality)) if n_images else 0.0
        quality = updated
        if shift < CONVERGENCE_TOLERANCE:
            converged = True
            break

    return OffsetModel(quality, offset, iterations, converged)


def fit_affine_model(
    ratings: np.ndarray,
    image_index: np.ndarray,
    rater_index: np.ndarray,
    n_images: int,
    n_raters: int,
    ridge: float = SCALE_RIDGE,
) -> AffineModel:
    """Fit `rating = offset(rater) + scale(rater) * quality(image)`.

    `scale` is shrunk toward 1 by `ridge`; a rater with few ratings therefore
    keeps a slope near 1 rather than taking an arbitrary one from noise.
    Offsets are centred on zero and scales on one each iteration, which fixes
    the location and scale degeneracies the model would otherwise have.
    """
    quality = _group_mean(ratings, image_index, n_images)
    offset = np.zeros(n_raters)
    scale = np.ones(n_raters)

    converged = False
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        # Per-rater least squares of their ratings on current quality.
        q = quality[image_index]
        n = np.bincount(rater_index, minlength=n_raters)
        sum_q = np.bincount(rater_index, weights=q, minlength=n_raters)
        sum_r = np.bincount(rater_index, weights=ratings, minlength=n_raters)
        sum_qq = np.bincount(rater_index, weights=q * q, minlength=n_raters)
        sum_qr = np.bincount(rater_index, weights=q * ratings, minlength=n_raters)

        safe = n > 0
        mean_q = np.divide(sum_q, n, out=np.zeros(n_raters), where=safe)
        mean_r = np.divide(sum_r, n, out=np.zeros(n_raters), where=safe)
        cov = sum_qr - n * mean_q * mean_r
        var = sum_qq - n * mean_q * mean_q
        scale = np.divide(
            cov + ridge, var + ridge, out=np.ones(n_raters), where=(var + ridge) > 0
        )
        scale = np.where(safe, scale, 1.0)
        # Floor the slope at zero. Unconstrained least squares hands a small
        # number of raters a *negative* slope, which makes the model treat
        # their judgements as evidence of the opposite -- indefensible in a
        # published label, whatever it does for fit. A rater whose ratings do
        # not track the consensus contributes nothing here; they are never
        # inverted. Affects ~0.2% of ratings on this corpus.
        scale = np.maximum(scale, 0.0)
        scale /= scale.mean()
        offset = np.where(safe, mean_r - scale * mean_q, 0.0)
        offset -= offset.mean()

        # Quality given offsets and scales: weighted least squares.
        b = scale[rater_index]
        num = np.bincount(
            image_index, weights=b * (ratings - offset[rater_index]), minlength=n_images
        )
        den = np.bincount(image_index, weights=b * b, minlength=n_images)
        updated = np.divide(num, den, out=quality.copy(), where=den > 0)
        shift = np.max(np.abs(updated - quality)) if n_images else 0.0
        quality = updated
        if shift < CONVERGENCE_TOLERANCE:
            converged = True
            break

    return AffineModel(quality, offset, scale, iterations, converged)


def held_out_rmse(
    ratings: np.ndarray,
    image_index: np.ndarray,
    rater_index: np.ndarray,
    n_images: int,
    n_raters: int,
    test_mask: np.ndarray,
) -> dict[str, float]:
    """Compare plain mean, offset and affine models on held-out ratings.

    Each model is fitted on the training ratings only, then used to predict
    the held-out ones. This is the evidence for preferring the simpler model:
    a richer model that does not predict better is just overfitting.
    """
    train = ~test_mask
    plain = _group_mean(ratings[train], image_index[train], n_images)

    offset_fit = fit_offset_model(
        ratings[train], image_index[train], rater_index[train], n_images, n_raters
    )
    affine_fit = fit_affine_model(
        ratings[train], image_index[train], rater_index[train], n_images, n_raters
    )

    actual = ratings[test_mask]
    images = image_index[test_mask]
    raters = rater_index[test_mask]

    def rmse(prediction: np.ndarray) -> float:
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))

    return {
        "plain_mean": rmse(plain[images]),
        "offset": rmse(offset_fit.quality[images] + offset_fit.offset[raters]),
        "affine": rmse(
            affine_fit.offset[raters]
            + affine_fit.scale[raters] * affine_fit.quality[images]
        ),
    }


def bootstrap_quality_ci(
    ratings: np.ndarray,
    image_index: np.ndarray,
    rater_index: np.ndarray,
    n_images: int,
    n_raters: int,
    iterations: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile confidence interval for each image's adjusted quality.

    **Raters are resampled, not ratings.** The uncertainty that matters here
    is "would a different panel of people have scored this face differently?",
    and resampling individual ratings would treat one rater's twenty
    judgements as twenty independent draws, understating it.
    """
    rng = np.random.default_rng(seed)
    by_rater = [np.flatnonzero(rater_index == r) for r in range(n_raters)]
    samples = np.full((iterations, n_images), np.nan)

    for step in range(iterations):
        chosen = rng.integers(0, n_raters, size=n_raters)
        positions = np.concatenate([by_rater[r] for r in chosen if by_rater[r].size])
        if positions.size == 0:
            continue
        # Re-index raters so a rater drawn twice is two distinct panellists.
        replicate = np.concatenate(
            [
                np.full(by_rater[r].size, k)
                for k, r in enumerate(chosen)
                if by_rater[r].size
            ]
        )
        fit = fit_offset_model(
            ratings[positions],
            image_index[positions],
            replicate,
            n_images,
            len(chosen),
        )
        seen = np.bincount(image_index[positions], minlength=n_images) > 0
        samples[step, seen] = fit.quality[seen]

    lower = np.nanpercentile(samples, 2.5, axis=0)
    upper = np.nanpercentile(samples, 97.5, axis=0)
    return lower, upper
