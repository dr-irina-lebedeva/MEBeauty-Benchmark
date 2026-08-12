"""The geometric features the classical era was built on.

Pure functions over a 68-point landmark array, so they can be checked directly
against the properties they are supposed to have -- which is the only way to
know a feature block means what its name says.
"""

from __future__ import annotations

import numpy as np
import pytest

from fbp_benchmark.methods.classical import (
    KEY_POINTS,
    SYMMETRY_PAIRS,
    _normalised,
    pairwise_distances,
    proportion_features,
    symmetry_features,
)


def face(seed: int = 0) -> np.ndarray:
    """A plausible 68-point face: random, but with eyes where eyes go."""
    rng = np.random.default_rng(seed)
    points = rng.normal(scale=20.0, size=(68, 2))
    points[36:42] += np.array([-30.0, -20.0])  # left eye
    points[42:48] += np.array([30.0, -20.0])  # right eye
    return points


def test_normalisation_removes_translation():
    points = face()
    assert np.allclose(_normalised(points), _normalised(points + 100.0), atol=1e-9)


def test_normalisation_removes_scale():
    points = face()
    assert np.allclose(_normalised(points), _normalised(points * 3.0), atol=1e-9)


def test_normalisation_survives_coincident_eyes():
    # Extreme profile views do occur in this dataset; dividing by a zero
    # inter-ocular distance would produce inf and poison the whole feature row.
    points = face()
    points[36:48] = points[36]
    result = _normalised(points)
    assert np.isfinite(result).all()


def test_pairwise_distances_have_the_expected_count_and_are_non_negative():
    n = len(KEY_POINTS)
    distances = pairwise_distances(face())
    assert len(distances) == n * (n - 1) // 2
    assert (distances >= 0).all()


def test_pairwise_distances_are_scale_invariant():
    points = face()
    assert np.allclose(
        pairwise_distances(points), pairwise_distances(points * 5.0), atol=1e-8
    )


def test_a_perfectly_symmetric_face_has_near_zero_symmetry_features():
    # Build a face mirrored about x = 0, then check the block reports it.
    # Eyes are positioned *before* the pairs are mirrored: two of the mirror
    # pairs are eye corners, so doing it the other way round overwrites them.
    rng = np.random.default_rng(3)
    points = rng.normal(scale=20.0, size=(68, 2))
    points[36:42] += np.array([-30.0, -20.0])
    points[42:48] = points[36:42] * np.array([-1.0, 1.0])
    points[27:31, 0] = 0.0  # midline down the nose bridge
    for a, b in SYMMETRY_PAIRS:
        points[b] = np.array([-points[a, 0], points[a, 1]])

    assert symmetry_features(points).max() < 1e-6


def test_an_asymmetric_face_reports_asymmetry():
    points = face()
    points[SYMMETRY_PAIRS[0][1]] += np.array([80.0, 0.0])
    assert symmetry_features(points).max() > 0.05


def test_symmetry_features_have_one_value_per_mirror_pair():
    assert len(symmetry_features(face())) == len(SYMMETRY_PAIRS)


def test_proportion_features_are_finite_and_scale_invariant():
    points = face()
    a = proportion_features(points)
    assert np.isfinite(a).all()
    assert np.allclose(a, proportion_features(points * 7.0), atol=1e-6)


@pytest.mark.parametrize(
    "extract", [pairwise_distances, symmetry_features, proportion_features]
)
def test_every_feature_block_is_deterministic(extract):
    points = face()
    assert np.array_equal(extract(points), extract(points))
