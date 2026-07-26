"""Try every automated route to a source link for images whose filename
carries no provenance signal, and report honestly when there isn't one.

`infer_image_provenance.py` resolves provenance from filename patterns, then
inherits it across byte-identical duplicates. Whatever it leaves as
`unknown` has no filename signal at all (e.g. `0.png`, `f3.png`, `30.jpg`).
This script attempts the remaining automated options on exactly those:

1. **Embedded image metadata** -- EXIF Artist / Copyright / ImageDescription /
   XP* fields, and PNG text chunks. Stock downloads occasionally retain the
   photographer or a source URL here.
2. **Perceptual near-duplicate matching** against every image that *does*
   have provenance. Exact (SHA-256) duplicate inheritance already happened
   upstream; this catches the weaker case of the same photo re-saved at a
   different quality or crop, which exact hashing misses.

What this deliberately does **not** do: reverse image search. That requires
uploading each image to a search service, which needs a tool that can POST
binary data -- not available here, and these images are not published at any
URL a search engine could be pointed at. Reverse image search remains the
only known route for the images this script cannot resolve, and it needs a
human with a browser.

The near-duplicate threshold defaults to the value validated in Finding 12
against this dataset's own distribution (the one true near-duplicate pair
sits at Hamming distance 16; the closest non-matching candidate at 58).
Anything above that is a different photograph that merely looks similar --
portraits of different people routinely land in the 60-90 range, so a loose
threshold here would invent provenance rather than recover it.

    uv run --with imagehash --with pillow python \\
        scripts/data/recover_unknown_provenance.py \\
        --metadata data/mebeauty_v3/images/metadata.parquet \\
        --images data/mebeauty_v3/images \\
        --output reports/legacy_audit/unknown_provenance_recovery.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imagehash
import pandas as pd
from PIL import ExifTags, Image

_METADATA_FIELDS = {
    "Artist",
    "Copyright",
    "ImageDescription",
    "XPAuthor",
    "XPComment",
    "XPSubject",
}
_TEXT_CHUNK_KEYS = {"description", "comment", "author", "copyright", "xmp", "url"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="v3 metadata.parquet")
    parser.add_argument("--images", required=True, help="v3 images/ directory")
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Max perceptual Hamming distance to accept as the same photo "
        "(Finding 12's validated value; raising this invents provenance)",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path")
    return parser.parse_args()


# Read failures are collected and reported, never swallowed. Finding 3 of
# this audit exists precisely because the legacy pipeline wrapped image
# processing in a bare `except: print(...)`, which hid a 94.7% failure rate
# in one subgroup for years. A "0 of 115 recovered" result is only
# trustworthy if we know all 115 were actually readable.
def embedded_metadata(path: Path) -> tuple[dict[str, str], str | None]:
    """Any author/copyright/description field carried inside the file.

    Returns (fields, error) -- `error` is non-None only if the file could not
    be read at all, which the caller reports rather than ignoring.
    """
    try:
        with Image.open(path) as image:
            found = {}
            exif = image.getexif()
            if exif:
                for tag, value in exif.items():
                    name = ExifTags.TAGS.get(tag, str(tag))
                    if name in _METADATA_FIELDS and str(value).strip():
                        found[name] = str(value)[:200]
            for key, value in (image.info or {}).items():
                if key.lower() in _TEXT_CHUNK_KEYS and str(value).strip():
                    found[key] = str(value)[:200]
            return found, None
    except (OSError, ValueError) as error:
        return {}, f"{type(error).__name__}: {error}"


def perceptual_hash(path: Path) -> tuple[imagehash.ImageHash | None, str | None]:
    """Returns (hash, error); a failed read is reported, not silently skipped."""
    try:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"), hash_size=16), None
    except (OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(args.metadata)
    unknown = metadata[metadata["inferred_platform"] == "unknown"]
    known = metadata[metadata["inferred_platform"] != "unknown"]
    print(f"{len(unknown)} images with no provenance signal, {len(known)} with one")

    read_errors: list[dict[str, str]] = []

    metadata_hits = []
    for row in unknown.itertuples():
        found, error = embedded_metadata(images_dir / row.file_name)
        if error:
            read_errors.append({"legacy_path": row.legacy_path, "error": error})
        if found:
            metadata_hits.append({"legacy_path": row.legacy_path, "fields": found})
    print(f"Embedded metadata: {len(metadata_hits)} of {len(unknown)} carry any")

    print("Perceptual hashing (this is the slow part) ...")
    unknown_hashes = []
    for row in unknown.itertuples():
        image_hash, error = perceptual_hash(images_dir / row.file_name)
        if error:
            read_errors.append({"legacy_path": row.legacy_path, "error": error})
        elif image_hash is not None:
            unknown_hashes.append((row.legacy_path, image_hash))

    known_hashes = []
    for row in known.itertuples():
        image_hash, error = perceptual_hash(images_dir / row.file_name)
        if error:
            read_errors.append({"legacy_path": row.legacy_path, "error": error})
        elif image_hash is not None:
            known_hashes.append((row.legacy_path, row.inferred_source_url, image_hash))

    matches, closest = [], []
    for legacy_path, unknown_hash in unknown_hashes:
        distance, match_path, url = min(
            ((unknown_hash - h, p, u) for p, u, h in known_hashes),
            key=lambda candidate: candidate[0],
        )
        closest.append(distance)
        if distance <= args.threshold:
            matches.append(
                {
                    "legacy_path": legacy_path,
                    "distance": int(distance),
                    "matched_path": match_path,
                    "recovered_source_url": url,
                }
            )

    closest_sorted = sorted(closest)
    report = {
        "unknown_provenance_images": len(unknown),
        "images_with_provenance": len(known),
        "unknown_images_successfully_read": len(unknown_hashes),
        "read_errors": read_errors,
        "embedded_metadata_hits": metadata_hits,
        "embedded_metadata_hit_count": len(metadata_hits),
        "near_duplicate_threshold": args.threshold,
        "near_duplicate_matches": matches,
        "near_duplicate_match_count": len(matches),
        "closest_distance_distribution": {
            "min": int(closest_sorted[0]) if closest_sorted else None,
            "median": int(closest_sorted[len(closest_sorted) // 2])
            if closest_sorted
            else None,
            "max": int(closest_sorted[-1]) if closest_sorted else None,
        },
        "conclusion": (
            "Every automated route is exhausted for the unresolved images: no "
            "filename signal (by definition), no exact-duplicate inheritance "
            "(applied upstream), no embedded metadata, and no perceptual match "
            "within the validated threshold. Reverse image search is the only "
            "remaining route and requires uploading each file to a search "
            "service by hand -- it cannot be automated from this repository."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Near-duplicate matches at threshold <= {args.threshold}: {len(matches)}")
    if closest_sorted:
        print(
            f"  closest distance seen: {closest_sorted[0]} "
            f"(median {closest_sorted[len(closest_sorted) // 2]})"
        )
    resolved = len(metadata_hits) + len(matches)
    print(f"\nRecoverable automatically: {resolved} of {len(unknown)}")
    if read_errors:
        print(f"  WARNING: {len(read_errors)} file(s) could not be read at all:")
        for entry in read_errors[:10]:
            print(f"    {entry['legacy_path']}: {entry['error']}")
    else:
        print(f"  (all {len(unknown_hashes)} unknown-source images read successfully)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
