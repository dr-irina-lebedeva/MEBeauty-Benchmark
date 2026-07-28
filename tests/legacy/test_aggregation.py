import numpy as np
import pytest

from mebeauty_benchmark.legacy.aggregation import (
    bootstrap_quality_ci,
    fit_affine_model,
    fit_offset_model,
    held_out_rmse,
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
