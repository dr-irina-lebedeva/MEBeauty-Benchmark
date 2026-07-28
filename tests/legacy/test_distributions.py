import math

import pandas as pd
import pytest

from mebeauty_benchmark.legacy.distributions import (
    MAX_ENTROPY_BITS,
    SCORE_BINS,
    build_distributions,
    rating_distribution,
)


def test_counts_land_in_the_right_bins():
    distribution = rating_distribution([1, 1, 5, 10])

    assert distribution.counts[0] == 2  # two 1s
    assert distribution.counts[4] == 1  # one 5
    assert distribution.counts[9] == 1  # one 10
    assert distribution.n_ratings == 4
    assert len(distribution.counts) == len(SCORE_BINS)


def test_probabilities_sum_to_one():
    distribution = rating_distribution([3, 3, 7, 9, 9, 9])

    assert sum(distribution.probabilities) == pytest.approx(1.0)
    assert distribution.probabilities[2] == pytest.approx(2 / 6)


def test_unanimous_ratings_have_zero_entropy():
    distribution = rating_distribution([6, 6, 6, 6])

    assert distribution.entropy_bits == pytest.approx(0.0)
    assert distribution.std == pytest.approx(0.0)


def test_perfectly_split_ratings_have_maximum_entropy():
    distribution = rating_distribution(list(SCORE_BINS))

    assert distribution.entropy_bits == pytest.approx(MAX_ENTROPY_BITS)
    assert distribution.entropy_bits == pytest.approx(math.log2(10))


def test_single_rating_has_no_spread_rather_than_zero():
    distribution = rating_distribution([8])

    assert distribution.n_ratings == 1
    assert distribution.mean == 8.0
    assert distribution.std is None


def test_out_of_range_and_fractional_scores_are_rejected_not_dropped():
    for bad in ([0], [11], [7.5], [-1]):
        with pytest.raises(ValueError, match="not an integer"):
            rating_distribution(bad)


def test_empty_ratings_are_rejected():
    with pytest.raises(ValueError, match="zero ratings"):
        rating_distribution([])


def test_build_makes_one_distribution_per_image():
    ratings = pd.DataFrame(
        {
            "image_id": ["img1", "img1", "img2", "img2"],
            "score": [8, 8, 2, 4],
        }
    )

    out = build_distributions(ratings).set_index("image_id")

    assert len(out) == 2
    assert out.loc["img1", "mean"] == 8.0
    assert out.loc["img1", "n_ratings"] == 2
    assert out.loc["img2", "mean"] == 3.0
    assert out.loc["img2", "n_ratings"] == 2


def test_build_rejects_a_table_missing_required_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_distributions(pd.DataFrame({"image_id": ["a"], "rating": [5]}))


def test_distribution_mean_matches_a_plain_mean_of_the_ratings():
    scores = [1, 4, 4, 9, 10]

    distribution = rating_distribution(scores)

    assert distribution.mean == pytest.approx(sum(scores) / len(scores))
    assert distribution.median == 4.0
