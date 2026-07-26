from math import comb

import numpy as np
import pytest

from mebeauty_benchmark.legacy.geometry import (
    FEATURE_DIM,
    compute_geometric_features,
    parse_landmark_string,
)


def _set_point(coords: np.ndarray, point_index: int, x: float, y: float) -> None:
    coords[2 * (point_index - 1)] = x
    coords[2 * point_index - 1] = y


def test_feature_dim_matches_expected_combination_count():
    # 19 landmark points, C(19, 4) combinations, 3 ratio permutations each.
    assert FEATURE_DIM == comb(19, 4) * 3


def test_first_combination_matches_hand_computed_ratios():
    # Points 18, 22, 23, 27 form a 3-4-5-ish rectangle so every ratio is 1.0.
    coords = np.zeros(136)
    _set_point(coords, 18, 0, 0)
    _set_point(coords, 22, 4, 0)
    _set_point(coords, 23, 0, 3)
    _set_point(coords, 27, 4, 3)

    features = compute_geometric_features(coords)

    assert features.shape == (FEATURE_DIM,)
    assert features[0] == pytest.approx(1.0)  # dist(18,22)/dist(23,27) = 4/4
    assert features[1] == pytest.approx(1.0)  # dist(18,23)/dist(22,27) = 3/3
    assert features[2] == pytest.approx(1.0)  # dist(18,27)/dist(22,23) = 5/5


def test_compute_geometric_features_rejects_wrong_length():
    with pytest.raises(ValueError):
        compute_geometric_features(np.zeros(10))


def test_parse_landmark_string_handles_trailing_comma():
    landmark_string = "1,2,3,4," * 34  # 136 values, legacy format has a trailing comma
    coords = parse_landmark_string(landmark_string)
    assert coords.shape == (136,)
    assert coords[0] == 1.0


def test_parse_landmark_string_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_landmark_string("1,2,3")
