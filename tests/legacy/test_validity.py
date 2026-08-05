import pandas as pd

from mebeauty_benchmark.legacy.validity import (
    MIN_DISTINCT_SCORES,
    MIN_RATINGS_PER_IMAGE,
    attach_validity,
    labelled_image_ids,
    rater_validity,
)


def _ratings(**raters: list[float]) -> pd.DataFrame:
    rows = []
    for rater, scores in raters.items():
        for index, score in enumerate(scores):
            rows.append({"image_id": f"img{index}", "rater_id": rater, "score": score})
    return pd.DataFrame(rows)


def test_a_rater_with_few_ratings_is_kept():
    # The rule this policy deliberately dropped. Three judgements are three
    # real judgements; the image-level support rule is what guards the labels.
    frame = rater_validity(_ratings(light=[2.0, 6.0, 9.0])).set_index("rater_id")

    assert frame.loc["light", "rater_valid"]
    assert frame.loc["light", "invalid_reason"] == ""


def test_a_rater_who_gave_one_score_to_everything_is_invalid():
    frame = rater_validity(_ratings(flat=[7.0] * 40)).set_index("rater_id")

    assert not frame.loc["flat", "rater_valid"]
    assert "distinct" in frame.loc["flat", "invalid_reason"]


def test_alternating_two_values_is_also_invalid():
    # A zero-variance test misses this, but 40 judgements spread over two
    # values still discriminates almost nothing.
    frame = rater_validity(_ratings(binary=[1.0, 2.0] * 20)).set_index("rater_id")

    assert not frame.loc["binary", "rater_valid"]


def test_the_distinct_score_rule_applies_at_any_volume():
    # A light rater is not exempt: two values over three ratings is as
    # uninformative as two values over two hundred.
    frame = rater_validity(_ratings(few_flat=[4.0, 4.0, 5.0])).set_index("rater_id")

    assert not frame.loc["few_flat", "rater_valid"]


def test_three_distinct_values_is_enough_to_stay_valid():
    frame = rater_validity(_ratings(varied=[1.0, 2.0, 3.0] * 14)).set_index("rater_id")

    assert frame.loc["varied", "rater_valid"]
    assert frame.loc["varied", "n_distinct_scores"] == MIN_DISTINCT_SCORES


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
    ratings = _ratings(flat=[5.0] * 3, solid=[1.0, 5.0, 9.0] * 10)

    out = attach_validity(ratings)

    assert len(out) == len(ratings)
    assert out.loc[out["rater_id"] == "flat", "rater_valid"].eq(False).all()
    assert out.loc[out["rater_id"] == "solid", "rater_valid"].all()


def test_valid_raters_have_no_reason_recorded():
    frame = rater_validity(_ratings(solid=[1.0, 5.0, 9.0] * 10)).set_index("rater_id")

    assert frame.loc["solid", "invalid_reason"] == ""


def test_an_image_needs_the_threshold_in_ratings_to_be_labelled():
    every = {f"r{i}": [1.0, 5.0, 9.0] for i in range(MIN_RATINGS_PER_IMAGE)}
    ratings = attach_validity(_ratings(**every))
    # img2 loses one rater, leaving it one short of the threshold.
    thin = ratings[~((ratings["image_id"] == "img2") & (ratings["rater_id"] == "r0"))]

    labelled = labelled_image_ids(thin)

    assert "img0" in labelled and "img1" in labelled
    assert "img2" not in labelled


def test_straight_lining_raters_cannot_prop_an_image_over_the_threshold():
    # The rater screen must run first. One short of the threshold in real
    # raters, plus any number of flat ones, is still one short.
    real = _ratings(
        **{f"real{i}": [1.0, 5.0, 9.0] for i in range(MIN_RATINGS_PER_IMAGE - 1)}
    )
    flat = _ratings(**{f"flat{i}": [7.0, 7.0, 7.0] for i in range(20)})
    ratings = attach_validity(pd.concat([real, flat], ignore_index=True))

    assert labelled_image_ids(ratings) == set()
