"""Normalize the legacy repository's machine-specific image paths."""

from __future__ import annotations

import re

_LEGACY_PREFIXES = (
    "/home/ubuntu/ME-beautydatabase/images/",
    "/home/ubuntu/MEBeauty-database/images/",
    "/home/ubuntu/crop/",
)

_CROP_PREFIX_RE = re.compile(r"^\./cropped_images/images_crop_align_\w+/")


def normalize_image_path(raw_path: str) -> str:
    """Strip machine-specific prefixes and leading `./`, keep everything else.

    The legacy data files mix at least three absolute-path conventions plus
    one relative convention. This collapses all of them to a path relative
    to the dataset root (e.g. ``female/caucasian/f8.jpg``).
    """
    path = raw_path.strip()

    for prefix in _LEGACY_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]

    if path.startswith("./cropped_images/"):
        return _CROP_PREFIX_RE.sub("", path)

    if path.startswith("cropped_images/"):
        return re.sub(r"^cropped_images/images_crop_align_\w+/", "", path)

    if path.startswith("./"):
        return path[2:]

    return path
