"""Rater-offset-adjusted image scores, and the model selection behind them.

The primary label stays the plain mean. This module produces a second,
modelled estimate for consumers who want one, and — as much as the estimate
itself — the evidence for the modelling choices.

**The model.** Raters differ in where they sit on the scale: two people can
rank a set of faces identically while one averages 7 and the other 5. With an
unbalanced design (nobody rated everything), a plain mean lets that offset
leak into the image score. So:

    rating(r, i) = quality(i) + offset(r) + noise

fitted by alternating least squares. Offsets are anchored to a rating-count-
weighted mean of zero, which is identifiable here because the rater x image
graph is a single connected component — verified, not assumed. That weighting
is what keeps the fitted qualities on the scale the ratings were given on.

**Offsets are shrunk.** An offset fitted from one rating absorbs that rating's
deviation exactly, turning noise into a correction. Each offset is therefore
trusted `n / (n + 5)` of the way, matching `normalization.py`.

**Per-rater scale, and how it was decided.** The natural extension gives each
rater a slope too, `rating = offset(r) + scale(r) * quality(i)`, with `scale`
ridge-regularised toward 1. It was expected to overfit -- some raters have a
handful of ratings -- so the choice was left to measurement rather than taste.

That measurement was initially the wrong one. Affine wins all 5 folds on
held-out RMSE of *individual ratings*, and on that basis it shipped. But the
published label is not a prediction of one person's rating; it is a per-image
consensus, and what matters is whether a different panel would reproduce it.
Measured that way (`split_half_reliability`, 40 rater splits) the ranking
inverts:

    plain mean          0.749
    affine              0.803
    offset, unshrunk    0.811
    offset, shrunk      0.811   <- ships

More parameters fit individual ratings better while making the image score
*less* reproducible. Selection now runs on split-half reliability, so the
criterion matches what the number is used for.

A per-rater z-score scores marginally higher still (0.818, winning 40/40
paired splits). It is not used: the two agree at r = 0.992, and z-scoring
divides by a standard deviation estimated from whichever images a rater
happened to see, which is an assumption the offset model does not need. The
0.007 is not worth buying with it.

The affine fit is still computed, because two things about it are worth
keeping on the record. First, unconstrained least squares gave a small number
of raters a *negative* slope, inverting their judgements; `scale` is floored at
zero. Second, fitted `scale` correlates 0.54 with how much a rater agrees with
the consensus, which looks like the agreement-based re-weighting this project
rejects elsewhere -- but simulating a world where the affine model is exactly
true produces a correlation of 0.88, so the coupling is mechanical and 0.54 is
*lower* than chance would give. Neither is a reason to ship it.

**The offset model rests on one assumption, and it is checked.** If raters had
been assigned systematically different images, a rater's mean would reflect
their assignment rather than their generosity, and correcting for it would
remove real signal. Measured here, a rater's mean correlates 0.090 with the
leave-one-rater-out quality of the images they saw (permutation p = 0.11) --
not distinguishable from chance. That diagnostic ships beside the scores
rather than being taken on trust, because on a differently-assigned corpus it
could easily fail. (Computed naively, with the rater's own rating left inside
the image mean, the same figure reads 0.30; that version is mechanically
inflated and should not be quoted.)

Both fits ship. `score_adjusted` is the selected model; `score_adjusted_offset`
is the unshrunk offset fit, kept so the effect of shrinkage stays visible.
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

#: Empirical-Bayes shrinkage on rater offsets: an offset estimated from `n`
#: ratings is trusted `n / (n + k)` of the way. A rater with one rating would
#: otherwise get an offset that exactly absorbs their single deviation, which
#: is noise fitted as if it were scale preference. Matches the `k = 5` used by
#: `normalization.py`, validated there by held-out prediction.
OFFSET_SHRINKAGE = 5.0


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
    shrinkage: float = 0.0,
) -> OffsetModel:
    """Fit `rating = quality(image) + offset(rater)` by alternating least squares.

    Offsets are re-centred every iteration. Without that anchor the model is
    degenerate -- adding a constant to every quality and subtracting it from
    every offset leaves the fit unchanged -- so the constraint is what makes
    the solution unique, not a cosmetic step.

    The centring is **weighted by rating count**, which fixes where the scale
    sits. Any anchor gives the same image *ranking*, but only this one keeps
    the fitted qualities on the scale the ratings were given on: it forces the
    rating-weighted mean of `quality` to equal the grand mean of `ratings`
    exactly. Centring the offsets unweighted instead lets prolific raters pull
    the whole label up or down -- on this corpus, +0.12 -- which matters when
    `score_adjusted` is published beside `score_mean` as though both were on a
    1-10 scale.

    Shrinkage is **opt-in**: with the default the fit is pure least squares and
    recovers known offsets exactly, which is what the tests pin. Pass
    `shrinkage=OFFSET_SHRINKAGE` for the published label, where each offset is
    pulled toward zero by `n / (n + k)` so a rater seen a handful of times
    cannot move an image score far.
    """
    quality = _group_mean(ratings, image_index, n_images)
    offset = np.zeros(n_raters)
    counts = np.bincount(rater_index, minlength=n_raters).astype(float)
    weight = counts / (counts + shrinkage) if shrinkage else np.ones(n_raters)
    anchor = counts if counts.sum() else np.ones(n_raters)

    converged = False
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        offset = _group_mean(ratings - quality[image_index], rater_index, n_raters)
        offset *= weight
        offset -= np.average(offset, weights=anchor)
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


def split_half_reliability(
    ratings: np.ndarray,
    image_index: np.ndarray,
    rater_index: np.ndarray,
    n_images: int,
    n_raters: int,
    seeds: int = 40,
    min_per_half: int = 4,
    shrinkage: float = OFFSET_SHRINKAGE,
) -> dict[str, float]:
    """Reliability of each candidate *image score* under a rater split-half.

    `held_out_rmse` asks which model predicts an individual rating best. That
    is not what the published label is for. The label is a per-image consensus,
    and the question that matters is whether a *different panel of raters*
    would have produced the same one. So: split the raters in two, aggregate
    each half independently, and correlate.

    The distinction is not academic. The affine model wins on held-out RMSE --
    more parameters fit individual ratings better -- while producing a *less*
    reproducible image score than the plain offset model. Selecting on RMSE
    therefore picks the worse label, which is what this function exists to
    stop.
    """
    rng_raters = np.arange(n_raters)
    scores: dict[str, list[float]] = {"plain_mean": [], "offset": [], "affine": []}

    for seed in range(seeds):
        shuffled = np.random.default_rng(seed).permutation(rng_raters)
        first = np.zeros(n_raters, dtype=bool)
        first[shuffled[: n_raters // 2]] = True
        left, right = first[rater_index], ~first[rater_index]

        counts = [
            np.bincount(image_index[side], minlength=n_images) for side in (left, right)
        ]
        # An image needs enough ratings on *both* sides for the comparison to
        # carry information; otherwise the correlation measures sampling noise.
        keep = (counts[0] >= min_per_half) & (counts[1] >= min_per_half)
        if keep.sum() < 2:
            continue

        estimates: dict[str, list[np.ndarray]] = {k: [] for k in scores}
        for side in (left, right):
            values, images, raters = (
                ratings[side],
                image_index[side],
                rater_index[side],
            )
            estimates["plain_mean"].append(_group_mean(values, images, n_images))
            estimates["offset"].append(
                fit_offset_model(
                    values, images, raters, n_images, n_raters, shrinkage
                ).quality
            )
            estimates["affine"].append(
                fit_affine_model(values, images, raters, n_images, n_raters).quality
            )
        for model, (a, b) in estimates.items():
            scores[model].append(float(np.corrcoef(a[keep], b[keep])[0, 1]))

    return {model: float(np.mean(values)) for model, values in scores.items()}


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
