"""Geometric (facial-ratio) features, ported from the legacy notebook.

Faithful re-implementation of ``generateAllFeatures``/``facialRatio`` from
``get_landmarks_geom.features.ipynb``: for 19 fixed landmark points, every
combination of 4 yields 3 ratio features (distance between the first pair
over distance between the second pair). The legacy CSV lost these values
to ``str(numpy_array)`` truncation; this recomputes them directly from the
intact 68-point coordinates in ``landmarks.csv``, so no value here is
guessed — it is the documented formula run over untouched source data.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

# 1-indexed dlib 68-point landmark indices used by the legacy notebook.
_RATIO_POINTS = (
    18,
    22,
    23,
    27,
    37,
    40,
    43,
    46,
    28,
    32,
    34,
    36,
    5,
    9,
    13,
    49,
    55,
    52,
    58,
)

LANDMARK_POINT_COUNT = 68
LANDMARK_VALUE_COUNT = LANDMARK_POINT_COUNT * 2


def _build_index_quadruples() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx1: list[int] = []
    idx2: list[int] = []
    idx3: list[int] = []
    idx4: list[int] = []
    for c0, c1, c2, c3 in itertools.combinations(_RATIO_POINTS, 4):
        idx1 += [c0, c0, c0]
        idx2 += [c1, c2, c3]
        idx3 += [c2, c1, c1]
        idx4 += [c3, c3, c2]
    return (np.array(idx1), np.array(idx2), np.array(idx3), np.array(idx4))


_IDX1, _IDX2, _IDX3, _IDX4 = _build_index_quadruples()
FEATURE_DIM = len(_IDX1)


def _xy(coords: np.ndarray, point_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = coords[..., 2 * (point_index - 1)]
    y = coords[..., 2 * point_index - 1]
    return x, y


def parse_landmark_string(landmark_string: str) -> np.ndarray:
    """Parse the legacy ``"x1,y1,x2,y2,..."`` landmark string (trailing comma ok)."""
    values = [v for v in landmark_string.split(",") if v.strip() != ""]
    coords = np.array([float(v) for v in values], dtype=float)
    if coords.shape[0] != LANDMARK_VALUE_COUNT:
        raise ValueError(
            f"expected {LANDMARK_VALUE_COUNT} landmark values "
            f"({LANDMARK_POINT_COUNT} points), got {coords.shape[0]}"
        )
    return coords


def compute_geometric_features(landmark_coords: Sequence[float]) -> np.ndarray:
    """Compute the 11,628-dimensional facial-ratio feature vector.

    `landmark_coords` is the flat ``[x0, y0, x1, y1, ..., x67, y67]``
    array parsed from a `landmarks.csv` row.
    """
    coords = np.asarray(landmark_coords, dtype=float)
    if coords.shape[-1] != LANDMARK_VALUE_COUNT:
        raise ValueError(
            f"expected {LANDMARK_VALUE_COUNT} landmark values, got {coords.shape[-1]}"
        )
    x1, y1 = _xy(coords, _IDX1)
    x2, y2 = _xy(coords, _IDX2)
    x3, y3 = _xy(coords, _IDX3)
    x4, y4 = _xy(coords, _IDX4)
    dist1 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    dist2 = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dist1 / dist2
