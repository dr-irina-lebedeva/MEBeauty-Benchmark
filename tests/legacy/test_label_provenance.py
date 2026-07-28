import pandas as pd

from mebeauty_benchmark.legacy.label_provenance import (
    PROVENANCE_COLUMNS,
    attach_label_provenance,
    compute_label_support,
    rater_columns,
    support_frame,
)


def _workbook() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "image": ["a.jpg", "b.jpg"],
            "mean": [4.0, 7.0],
            "path": ["male/asian/a.jpg", "female/asian/b.jpg"],
            "r1": [3.0, 6.0],
            "r2": [5.0, 8.0],
            "r3": [None, 10.0],
        }
    )


def test_rater_columns_excludes_bookkeeping_columns():
    assert rater_columns(_workbook()) == ["r1", "r2", "r3"]


def test_support_counts_only_actual_ratings():
    support = {item.filename: item for item in compute_label_support(_workbook())}

    assert support["a.jpg"].n_ratings == 2
    assert support["a.jpg"].recomputed_score == 4.0
    assert support["b.jpg"].n_ratings == 3
    assert support["b.jpg"].recomputed_score == 8.0


def test_single_rating_has_no_spread_rather_than_zero():
    workbook = pd.DataFrame({"image": ["only.jpg"], "r1": [5.0], "r2": [None]})

    (support,) = compute_label_support(workbook)

    assert support.n_ratings == 1
    # 0.0 would read as perfect agreement between raters that do not exist.
    assert support.score_std is None


def test_ratings_are_pooled_across_duplicate_image_rows():
    workbook = pd.DataFrame(
        {"image": ["dup.jpg", "dup.jpg"], "r1": [2.0, 6.0], "r2": [4.0, None]}
    )

    (support,) = compute_label_support(workbook)

    assert support.n_ratings == 3
    assert support.recomputed_score == 4.0


def test_workbook_without_rater_columns_is_rejected():
    try:
        compute_label_support(pd.DataFrame({"image": ["a.jpg"], "mean": [4.0]}))
    except ValueError as error:
        assert "wide-format" in str(error)
    else:
        raise AssertionError("expected ValueError for a workbook with no raters")


def test_attach_preserves_canonical_score_and_flags_discrepancy():
    ratings = pd.DataFrame({"image_id": ["id_a", "id_b"], "score": [4.0, 6.0]})
    support = support_frame(compute_label_support(_workbook()))

    out = attach_label_provenance(
        ratings, support, {"id_a": "a.jpg", "id_b": "b.jpg"}, threshold=0.25
    )

    assert out["score"].tolist() == [4.0, 6.0]
    assert "legacy_filename" not in out.columns
    # a.jpg matches its raters exactly; b.jpg is canonical 6.0 vs recomputed 8.0.
    assert not out.loc[0, "label_discrepancy"]
    assert out.loc[1, "label_discrepancy"]
    assert out.loc[1, "score_delta"] == -2.0


def test_rerunning_refreshes_columns_instead_of_colliding():
    ratings = pd.DataFrame({"image_id": ["id_a"], "score": [4.0]})
    support = support_frame(compute_label_support(_workbook()))
    mapping = {"id_a": "a.jpg"}

    once = attach_label_provenance(ratings, support, mapping)
    twice = attach_label_provenance(once, support, mapping)

    assert not any(column.endswith(("_x", "_y")) for column in twice.columns)
    assert set(PROVENANCE_COLUMNS).issubset(twice.columns)
    pd.testing.assert_frame_equal(once, twice)


def test_images_missing_from_the_rater_matrix_keep_null_support():
    ratings = pd.DataFrame({"image_id": ["id_missing"], "score": [5.0]})
    support = support_frame(compute_label_support(_workbook()))

    out = attach_label_provenance(ratings, support, {"id_missing": "gone.jpg"})

    assert len(out) == 1
    assert out["score"].tolist() == [5.0]
    assert pd.isna(out.loc[0, "n_ratings"])
    # An unmeasurable delta must not be flagged as a discrepancy.
    assert not out.loc[0, "label_discrepancy"]
