"""Geometry for the `standardized_256` image configuration.

Pure arithmetic, deliberately free of any image I/O so the transform can be
tested exhaustively without fixture files -- the same split used by
`splits.py` and `provenance.py`.

The native images are the authoritative version of this dataset (see
docs/DATASET_AUDIT.md, Finding 17: they are heterogeneous 400x400 / 500x500 /
600x600 face crops plus 20 full-size photographs). `standardized_256` is a
derived convenience configuration: every image resized to a uniform 256x256,
aspect ratio preserved, centre-padded rather than stretched, so that faces
are never distorted.

Landmarks must move with the pixels. A consumer of `standardized_256` needs
landmark coordinates in *that* config's pixel space, which is the native
coordinates scaled and then offset by the padding:

    scale = min(256 / width, 256 / height)
    new_x = old_x * scale + pad_left
    new_y = old_y * scale + pad_top

Landmarks that sit outside the native image bounds stay outside after the
transform, proportionally. That is intentional: 106 of 2,495 native landmark
sets already extend past the frame (median 8px, max 65px) because
`face_alignment` extrapolates jaw and chin points beyond a tight crop's edge.
Clamping them to the frame would silently alter the geometry of those images
and make the two configurations encode *different* face shapes rather than
the same shape at two scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_TARGET = 256


@dataclass(frozen=True)
class Transform:
    """How one native image maps into the standardized square canvas."""

    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    output_width: int
    output_height: int


def compute_transform(
    width: int, height: int, target: int = DEFAULT_TARGET
) -> Transform:
    """Aspect-preserving fit of `width` x `height` into a `target` square.

    The longer side lands exactly on `target`; the shorter side is centred
    with the remainder split between the two opposite edges (the extra pixel
    of an odd remainder goes to the right/bottom). A square input therefore
    scales to exactly `target` x `target` with no padding at all.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")

    scale = min(target / width, target / height)
    # The longer side is exactly `target` by construction; round() only ever
    # affects the shorter one. min() guards against a rounding overshoot.
    resized_width = min(round(width * scale), target)
    resized_height = min(round(height * scale), target)

    pad_left = (target - resized_width) // 2
    pad_top = (target - resized_height) // 2
    return Transform(
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=target - resized_width - pad_left,
        pad_bottom=target - resized_height - pad_top,
        output_width=target,
        output_height=target,
    )


def transform_landmarks(
    landmarks: Sequence[float], transform: Transform
) -> list[float]:
    """Map flat [x0, y0, x1, y1, ...] landmarks into standardized coordinates."""
    if len(landmarks) % 2 != 0:
        raise ValueError(
            f"landmarks must be flat (x, y) pairs, got {len(landmarks)} values"
        )

    moved: list[float] = []
    for index in range(0, len(landmarks), 2):
        moved.append(landmarks[index] * transform.scale + transform.pad_left)
        moved.append(landmarks[index + 1] * transform.scale + transform.pad_top)
    return moved


def has_out_of_bounds_landmarks(
    landmarks: Sequence[float], width: int, height: int
) -> bool:
    """True if any landmark falls outside the image frame.

    Reported rather than corrected -- see this module's docstring. Consumers
    who need strictly in-frame landmarks can filter on this flag instead of
    discovering the condition during training.
    """
    for index in range(0, len(landmarks), 2):
        x, y = landmarks[index], landmarks[index + 1]
        if x < 0 or y < 0 or x > width or y > height:
            return True
    return False
