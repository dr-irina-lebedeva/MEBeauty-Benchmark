import pandas as pd

from mebeauty_benchmark.legacy.validity import (
    MAX_DISTINCT_SCORES,
    MIN_RATINGS,
    attach_validity,
    rater_validity,
)


def _ratings(**raters: list[float]) -> pd.DataFrame:
    rows = []
    for rater, scores in raters.items():
        for index, score in enumerate(scores):
            rows.append({"image_id": f"img{index}", "rater_id": rater, "score": score})
    return pd.DataFrame(rows)


def test_a_rater_with_too_few_ratings_is_invalid():
    frame = rater_validity(_ratings(light=[5.0] * (MIN_RATINGS - 1))).set_index(
        "rater_id"
    )

    assert not frame.loc["light", "rater_valid"]
    assert "fewer than" in frame.loc["light", "invalid_reason"]


def test_a_rater_who_gave_one_score_to_everything_is_invalid():
    frame = rater_validity(_ratings(flat=[7.0] * 40)).set_index("rater_id")

    assert not frame.loc["flat", "rater_valid"]
    assert "distinct" in frame.loc["flat", "invalid_reason"]


def test_alternating_two_values_is_also_invalid():
    # A zero-variance test misses this, but 40 judgements spread over two
    # values still discriminates almost nothing.
    frame = rater_validity(_ratings(binary=[1.0, 2.0] * 20)).set_index("rater_id")

    assert not frame.loc["binary", "rater_valid"]


def test_three_distinct_values_is_enough_to_stay_valid():
    frame = rater_validity(_ratings(varied=[1.0, 2.0, 3.0] * 14)).set_index("rater_id")

    assert frame.loc["varied", "rater_valid"]
    assert frame.loc["varied", "n_distinct_scores"] == MAX_DISTINCT_SCORES + 1


def test_a_harsh_but_discriminating_rater_stays_valid():
    # The point of the design: scoring low is a scale preference, not
    # invalidity. This rater is consistently harsh and still varies.
    frame = rater_validity(_ratings(harsh=[1.0, 2.0, 1.0, 3.0, 2.0] * 6)).set_index(
        "rater_id"
    )

    assert frame.loc["harsh", "rater_valid"]


def test_a_rater_who_disagrees_with_everyone_stays_valid():
    # Validity never consults agreement. This rater is a perfect inversion of
    # the other two and must survive.
    ratings = _ratings(
        a=[1.0, 3.0, 5.0, 7.0, 9.0] * 4,
        b=[1.0, 3.0, 5.0, 7.0, 9.0] * 4,
        contrarian=[9.0, 7.0, 5.0, 3.0, 1.0] * 4,
    )

    frame = rater_validity(ratings).set_index("rater_id")

    assert frame.loc["contrarian", "rater_valid"]


def test_attach_validity_keeps_every_row():
    ratings = _ratings(light=[5.0] * 3, solid=[1.0, 5.0, 9.0] * 10)

    out = attach_validity(ratings)

    assert len(out) == len(ratings)
    assert out.loc[out["rater_id"] == "light", "rater_valid"].eq(False).all()
    assert out.loc[out["rater_id"] == "solid", "rater_valid"].all()


def test_valid_raters_have_no_reason_recorded():
    frame = rater_validity(_ratings(solid=[1.0, 5.0, 9.0] * 10)).set_index("rater_id")

    assert frame.loc["solid", "invalid_reason"] == ""
