import numpy as np
import pytest

from mebeauty_benchmark.legacy.aggregation import (
    OFFSET_SHRINKAGE,
    bootstrap_quality_ci,
    fit_affine_model,
    fit_offset_model,
    held_out_rmse,
    split_half_reliability,
)


def _synthetic(n_images=40, n_raters=12, offsets=None, scales=None, seed=0):
    """A fully-crossed design with known image qualities and rater offsets."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(2, 9, n_images)
    offset = np.zeros(n_raters) if offsets is None else np.asarray(offsets, float)
    scale = np.ones(n_raters) if scales is None else np.asarray(scales, float)

    image_index = np.repeat(np.arange(n_images), n_raters)
    rater_index = np.tile(np.arange(n_raters), n_images)
    ratings = offset[rater_index] + scale[rater_index] * quality[image_index]
    return ratings, image_index, rater_index, quality, offset


def test_recovers_known_qualities_when_there_is_no_bias():
    ratings, images, raters, quality, _ = _synthetic()

    fit = fit_offset_model(ratings, images, raters, 40, 12)

    assert fit.converged
    assert np.allclose(fit.quality, quality, atol=1e-4)


def test_recovers_rater_offsets():
    offsets = np.linspace(-2, 2, 12)
    offsets -= offsets.mean()
    ratings, images, raters, quality, _ = _synthetic(offsets=offsets)

    fit = fit_offset_model(ratings, images, raters, 40, 12)

    assert np.allclose(fit.offset, offsets, atol=1e-4)
    assert np.allclose(fit.quality, quality, atol=1e-4)


def test_offsets_are_anchored_to_mean_zero():
    # Without the anchor the model is degenerate: any constant can move
    # between quality and offset. The constraint is what makes it unique.
    ratings, images, raters, _, _ = _synthetic(offsets=np.full(12, 3.0))

    fit = fit_offset_model(ratings, images, raters, 40, 12)

    assert fit.offset.mean() == pytest.approx(0.0, abs=1e-9)


def test_a_generous_rater_does_not_inflate_images_only_they_saw():
    # The failure a plain mean has: one rater sits +3 on the scale and is the
    # only person to rate image 0, so its plain mean is inflated by 3.
    ratings = np.array([9.0, 6.0, 6.0, 6.0, 9.0, 6.0])
    images = np.array([0, 1, 1, 2, 2, 3])
    raters = np.array([0, 0, 1, 1, 0, 1])

    fit = fit_offset_model(ratings, images, raters, 4, 2)

    plain_mean_image0 = 9.0
    assert fit.quality[0] < plain_mean_image0
    assert fit.offset[0] > fit.offset[1]


def test_affine_model_recovers_scale():
    scales = np.linspace(0.6, 1.4, 12)
    scales /= scales.mean()
    ratings, images, raters, _, _ = _synthetic(scales=scales)

    fit = fit_affine_model(ratings, images, raters, 40, 12, ridge=0.0)

    assert np.corrcoef(fit.scale, scales)[0, 1] > 0.99


def test_scale_is_shrunk_toward_one_for_light_raters():
    # A rater with two ratings should not be handed an extreme slope.
    ratings = np.array([2.0, 9.0, 2.0, 9.0, 1.0, 10.0])
    images = np.array([0, 1, 0, 1, 0, 1])
    raters = np.array([0, 0, 1, 1, 2, 2])

    strong = fit_affine_model(ratings, images, raters, 2, 3, ridge=100.0)

    assert np.allclose(strong.scale, 1.0, atol=0.05)


def test_held_out_comparison_returns_all_three_models():
    ratings, images, raters, _, _ = _synthetic(offsets=np.linspace(-1, 1, 12))
    rng = np.random.default_rng(1)
    test_mask = rng.random(len(ratings)) < 0.25

    result = held_out_rmse(ratings, images, raters, 40, 12, test_mask)

    assert set(result) == {"plain_mean", "offset", "affine"}
    # With real rater offsets present, modelling them must beat ignoring them.
    assert result["offset"] < result["plain_mean"]


def test_bootstrap_interval_brackets_the_estimate_and_widens_with_noise():
    rng = np.random.default_rng(2)
    ratings, images, raters, _, _ = _synthetic(n_images=15, n_raters=10)
    noisy = ratings + rng.normal(0, 1.5, len(ratings))

    fit = fit_offset_model(noisy, images, raters, 15, 10)
    lower, upper = bootstrap_quality_ci(
        noisy, images, raters, 15, 10, iterations=40, seed=3
    )

    inside = (lower <= fit.quality) & (fit.quality <= upper)
    assert inside.mean() > 0.8
    assert np.all(upper >= lower)


def test_empty_groups_do_not_crash():
    # Image 2 and rater 2 have no ratings at all.
    ratings = np.array([5.0, 7.0])
    images = np.array([0, 1])
    raters = np.array([0, 1])

    fit = fit_offset_model(ratings, images, raters, 3, 3)

    assert len(fit.quality) == 3
    assert np.isfinite(fit.quality).all()


def test_scale_is_never_negative():
    # Unconstrained least squares hands an inversely-correlated rater a
    # negative slope, which would make the model read their judgement as
    # evidence of the opposite. Rater 2 below rates in reverse.
    ratings = np.array([2.0, 9.0, 2.0, 9.0, 9.0, 2.0])
    images = np.array([0, 1, 0, 1, 0, 1])
    raters = np.array([0, 0, 1, 1, 2, 2])

    fit = fit_affine_model(ratings, images, raters, 2, 3, ridge=0.0)

    assert np.all(fit.scale >= 0.0)


def test_a_light_rater_earns_less_offset_than_an_identical_heavy_one():
    # Raters 0 and 1 both sit exactly +2 on the scale; rater 0 rated 40 images
    # and rater 1 rated 3. Only the heavy rater has the evidence to earn the
    # full correction. Rater 2 sits at -2 and anchors the mean-zero constraint,
    # which with only two raters would force the offsets to be symmetric and
    # hide the effect entirely.
    rng = np.random.default_rng(0)
    quality = rng.uniform(3, 8, 40)
    images = np.concatenate([np.arange(40), np.array([0, 1, 2]), np.arange(40)])
    raters = np.concatenate([np.zeros(40, int), np.ones(3, int), np.full(40, 2)])
    true_offset = np.array([2.0, 2.0, -2.0])
    ratings = quality[images] + true_offset[raters]

    unshrunk = fit_offset_model(ratings, images, raters, 40, 3)
    shrunk = fit_offset_model(ratings, images, raters, 40, 3, OFFSET_SHRINKAGE)

    assert unshrunk.offset[0] == pytest.approx(unshrunk.offset[1], abs=1e-4)
    assert shrunk.offset[1] < shrunk.offset[0]


def test_shrinkage_of_zero_reproduces_the_unregularised_fit():
    offsets = np.linspace(-2, 2, 12)
    offsets -= offsets.mean()
    ratings, images, raters, _, _ = _synthetic(offsets=offsets)

    assert np.allclose(
        fit_offset_model(ratings, images, raters, 40, 12, 0.0).quality,
        fit_offset_model(ratings, images, raters, 40, 12).quality,
    )


def test_split_half_reliability_prefers_the_model_that_removes_real_bias():
    # Raters carry large offsets, so a plain mean is polluted by whoever
    # happened to rate each image and the offset model should reproduce
    # better across an independent half of the panel.
    rng = np.random.default_rng(3)
    n_images, n_raters = 120, 24
    quality = rng.uniform(2, 9, n_images)
    offset = rng.normal(0, 2.0, n_raters)
    images = np.repeat(np.arange(n_images), 8)
    raters = np.concatenate(
        [rng.choice(n_raters, 8, replace=False) for _ in range(n_images)]
    )
    ratings = quality[images] + offset[raters] + rng.normal(0, 0.3, len(images))

    result = split_half_reliability(
        ratings, images, raters, n_images, n_raters, seeds=8, min_per_half=2
    )

    assert result["offset"] > result["plain_mean"]


def test_fitted_qualities_stay_on_the_scale_the_ratings_were_given_on():
    # A prolific generous rater and a light harsh one. Whatever the fit does
    # to individual images, the rating-weighted average of the fitted
    # qualities must equal the average of the raw ratings -- otherwise the
    # label drifts off the 1-10 scale it is published on.
    rng = np.random.default_rng(7)
    quality = rng.uniform(3, 8, 30)
    images = np.concatenate([np.arange(30), np.arange(30), np.array([0, 1, 2])])
    raters = np.concatenate([np.zeros(30, int), np.ones(30, int), np.full(3, 2)])
    true_offset = np.array([1.5, 0.0, -3.0])
    ratings = quality[images] + true_offset[raters]

    fit = fit_offset_model(ratings, images, raters, 30, 3, OFFSET_SHRINKAGE)

    assert fit.quality[images].mean() == pytest.approx(ratings.mean(), abs=1e-9)
