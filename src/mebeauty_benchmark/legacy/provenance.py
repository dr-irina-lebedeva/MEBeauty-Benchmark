"""Best-effort source-platform inference from legacy stock-photo filenames.

Filenames are the only provenance signal the legacy release kept. The
outputs here are unverified inferences (never confirmed against the live
platform) meant to make a future per-image consent/takedown review
possible — not a guarantee of the true source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PEXELS_RE = re.compile(r"^pexels-(?P<slug>.+)-(?P<photo_id>\d+)$")
_PIXABAY_RE = re.compile(r"^(?P<slug>.+)-(?P<photo_id>\d+)_\d+$")

# Trailing "(1)" / " (2)" is a download-manager copy marker, not part of the
# platform's filename. Left in place it defeats every pattern below -- an
# obvious Unsplash file like "gift-habeshaw-KBv5dEN3QtY-unsplash(1).jpg" was
# being tagged "unknown" purely because the stem no longer ends in
# "-unsplash". Confirmed against the dataset: 14 images were misclassified as
# having no provenance signal for this reason alone.
_COPY_MARKER_RE = re.compile(r"\s*\(\d+\)$")


@dataclass(frozen=True)
class ImageProvenance:
    platform: str
    photo_id: str | None
    inferred_source_url: str | None
    confidence: str  # "inferred" or "unknown"


def infer_provenance(filename: str) -> ImageProvenance:
    stem = _COPY_MARKER_RE.sub("", filename.rsplit(".", 1)[0])

    if stem.endswith("-unsplash"):
        body = stem[: -len("-unsplash")]
        # Unsplash photo IDs are a fixed 11 characters and may themselves
        # contain "-" (URL-safe base64-style charset), so splitting on the
        # last hyphen truncates the id whenever it has one internally
        # (e.g. "shivam-singh-2_X6NMP-E_U-unsplash.jpg" -> id "2_X6NMP-E_U",
        # not "E_U"). Confirmed against 195/1325 unsplash-pattern files.
        photo_id = (body[-11:] if len(body) >= 11 else body) or None
        url = f"https://unsplash.com/photos/{photo_id}" if photo_id else None
        return ImageProvenance("unsplash", photo_id, url, "inferred")

    match = _PEXELS_RE.match(stem)
    if match:
        photo_id = match.group("photo_id")
        return ImageProvenance(
            "pexels", photo_id, f"https://www.pexels.com/photo/{photo_id}/", "inferred"
        )

    match = _PIXABAY_RE.match(stem)
    if match:
        photo_id = match.group("photo_id")
        return ImageProvenance(
            "pixabay", photo_id, f"https://pixabay.com/photos/{stem}/", "inferred"
        )

    return ImageProvenance("unknown", None, None, "unknown")
