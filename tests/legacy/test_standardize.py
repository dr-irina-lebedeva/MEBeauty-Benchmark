import pytest

from mebeauty_benchmark.legacy.standardize import (
    compute_transform,
    has_out_of_bounds_landmarks,
    transform_landmarks,
)


def test_square_input_gets_no_padding():
    # The dominant case: 2,472 of 2,495 images are square face crops.
    for size in [400, 500, 600, 256]:
        t = compute_transform(size, size)
        assert (t.pad_left, t.pad_top, t.pad_right, t.pad_bottom) == (0, 0, 0, 0)
        assert (t.resized_width, t.resized_height) == (256, 256)


def test_output_is_always_the_target_square():
    for width, height in [(400, 400), (6200, 3600), (2028, 3000), (688, 688), (1, 7)]:
        t = compute_transform(width, height)
        assert t.output_width == t.output_height == 256
        assert t.pad_left + t.resized_width + t.pad_right == 256
        assert t.pad_top + t.resized_height + t.pad_bottom == 256


def test_non_square_preserves_aspect_ratio():
    # 6200x3600 is a real image in the dataset (male/caucasian/guy9.jpg).
    t = compute_transform(6200, 3600)
    assert t.resized_width == 256  # longer side fills the canvas
    assert abs((t.resized_width / t.resized_height) - (6200 / 3600)) < 0.02
    assert t.pad_left == t.pad_right == 0  # padding is on the short axis only
    assert t.pad_top > 0


def test_padding_is_centred_with_the_odd_pixel_at_the_end():
    t = compute_transform(100, 99)
    assert t.pad_top + t.pad_bottom == 256 - t.resized_height
    assert t.pad_bottom >= t.pad_top
    assert t.pad_bottom - t.pad_top <= 1


def test_landmarks_follow_the_pixels():
    t = compute_transform(500, 500)  # scale 0.512, no padding
    assert transform_landmarks([0.0, 0.0, 500.0, 500.0], t) == [0.0, 0.0, 256.0, 256.0]

    t = compute_transform(1000, 500)  # scale 0.256, padded top/bottom
    moved = transform_landmarks([0.0, 0.0], t)
    assert moved[0] == 0.0
    assert moved[1] == pytest.approx(t.pad_top)


def test_transform_introduces_no_new_out_of_bounds_landmarks():
    """The real invariant. 106 native landmark sets are *already* out of
    bounds, so "everything lands inside the frame" is not testable -- what
    must hold is that anything inside stays inside."""
    for width, height in [(400, 400), (600, 600), (6200, 3600), (2028, 3000)]:
        t = compute_transform(width, height)
        corners = [
            0.0,
            0.0,
            float(width),
            0.0,
            0.0,
            float(height),
            float(width),
            float(height),
        ]
        moved = transform_landmarks(corners, t)
        for index in range(0, len(moved), 2):
            assert -0.51 <= moved[index] <= 256.51
            assert -0.51 <= moved[index + 1] <= 256.51


def test_out_of_bounds_landmarks_stay_out_of_bounds():
    # Faithful, not clamped: a chin point extrapolated past the crop edge
    # must remain past the edge after transforming.
    t = compute_transform(400, 400)
    moved = transform_landmarks([420.0, 200.0], t)
    assert moved[0] > 256


def test_out_of_bounds_detection():
    assert not has_out_of_bounds_landmarks([10.0, 10.0, 390.0, 390.0], 400, 400)
    assert has_out_of_bounds_landmarks([10.0, 10.0, 420.0, 390.0], 400, 400)
    assert has_out_of_bounds_landmarks([-1.0, 10.0], 400, 400)


def test_transform_is_deterministic():
    assert compute_transform(3620, 3999) == compute_transform(3620, 3999)


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        compute_transform(0, 100)
    with pytest.raises(ValueError):
        transform_landmarks([1.0, 2.0, 3.0], compute_transform(400, 400))
