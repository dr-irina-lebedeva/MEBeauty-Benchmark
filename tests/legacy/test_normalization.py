import numpy as np
import pandas as pd
import pytest

from mebeauty_benchmark.legacy.normalization import (
    normalise_ratings,
    normalised_image_scores,
    rater_scales,
)


def _ratings(**raters: list[float]) -> pd.DataFrame:
    rows = [
        {"image_id": f"img{i}", "rater_id": rater, "score": float(score)}
        for rater, scores in raters.items()
        for i, score in enumerate(scores)
    ]
    return pd.DataFrame(rows)


def test_a_single_rating_rater_does_not_produce_a_division_by_zero():
    # The exact case that breaks plain z-scoring: one rating, no spread.
    ratings = _ratings(solo=[7.0], busy=[1.0, 5.0, 9.0, 3.0, 7.0] * 6)

    out = normalise_ratings(ratings)

    assert np.isfinite(out["z"]).all()


def test_a_light_rater_is_barely_adjusted_but_a_heavy_one_is():
    # Shrinkage weight is how much of the global statistics a rater gets.
    # Both raters sit high; a third provides the global spread.
    ratings = _ratings(
        light=[9.0, 9.0],
        heavy=[9.0] * 200,
        spread=[1.0, 3.0, 5.0, 7.0, 9.0] * 8,
    )

    scales = rater_scales(ratings, shrinkage=10.0).set_index("rater_id")

    assert scales.loc["light", "shrinkage_weight"] > 0.8
    assert scales.loc["heavy", "shrinkage_weight"] < 0.1
    # The heavy rater's mean is trusted; the light one's is pulled to global.
    assert abs(scales.loc["heavy", "shrunk_mean"] - 9.0) < abs(
        scales.loc["light", "shrunk_mean"] - 9.0
    )


def test_normalisation_removes_a_pure_offset_between_two_raters():
    # Both raters rank the images identically; one simply sits 3 points higher.
    # After normalising, the image ordering must be preserved and the two
    # raters must agree.
    harsh = [1.0, 2.0, 3.0, 4.0, 5.0] * 8
    generous = [4.0, 5.0, 6.0, 7.0, 8.0] * 8
    ratings = _ratings(harsh=harsh, generous=generous)

    out = normalise_ratings(ratings, shrinkage=1.0)
    by_rater = out.groupby("rater_id")["z"].mean()

    assert abs(by_rater["harsh"] - by_rater["generous"]) < 0.2


def test_image_scores_stay_on_the_one_to_ten_scale():
    rng = np.random.default_rng(0)
    rows = [
        {
            "image_id": f"img{i}",
            "rater_id": f"r{r}",
            "score": float(rng.integers(1, 11)),
        }
        for i in range(40)
        for r in range(12)
    ]

    out = normalised_image_scores(pd.DataFrame(rows))

    assert out["score_normalised"].between(1.0, 10.0).all()
    assert np.isfinite(out["score_normalised"]).all()


def test_min_ratings_drops_thinly_rated_images():
    ratings = pd.concat(
        [
            _ratings(**{f"r{r}": [5.0] * 3 for r in range(12)}),  # 12 raters, 3 images
            pd.DataFrame([{"image_id": "thin", "rater_id": "r0", "score": 8.0}]),
        ]
    )

    out = normalised_image_scores(ratings, min_ratings=10)

    assert "thin" not in set(out["image_id"])
    assert out["n_ratings"].min() >= 10


def test_normalised_score_differs_from_the_raw_mean_when_raters_are_biased():
    # img0 is rated only by the generous rater, img1 only by the harsh one.
    # A raw mean would call img0 better; normalisation should close the gap
    # because both raters gave their own middling score.
    rows = []
    for i in range(30):
        rows.append({"image_id": "warmup", "rater_id": "generous", "score": 8.0})
        rows.append({"image_id": "warmup", "rater_id": "harsh", "score": 3.0})
    rows.append({"image_id": "img0", "rater_id": "generous", "score": 8.0})
    rows.append({"image_id": "img1", "rater_id": "harsh", "score": 3.0})

    out = normalised_image_scores(pd.DataFrame(rows), shrinkage=1.0).set_index(
        "image_id"
    )

    raw_gap = abs(out.loc["img0", "raw_mean"] - out.loc["img1", "raw_mean"])
    normalised_gap = abs(
        out.loc["img0", "score_normalised"] - out.loc["img1", "score_normalised"]
    )
    assert raw_gap == pytest.approx(5.0)
    assert normalised_gap < raw_gap


def test_all_identical_ratings_are_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="nothing to normalise"):
        rater_scales(_ratings(a=[5.0] * 10, b=[5.0] * 10))


def test_negative_shrinkage_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        rater_scales(_ratings(a=[1.0, 5.0, 9.0]), shrinkage=-1.0)
