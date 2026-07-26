"""Tag every original image with a best-effort source-platform inference.

Filenames are the only per-file provenance signal the legacy release kept,
so most images are tagged by filename pattern alone. A second pass then
propagates provenance across byte-identical duplicates (see
`find_duplicate_images.py`): several images were saved twice, once under a
platform-slug filename and once under a bare numeric name that carries no
signal on its own -- e.g. `female/asian/30.jpg` is pixel-identical to
`asian-girl-4819726_1920.jpg`. Those rows are tagged `inferred-via-duplicate`
rather than `inferred`, so it stays clear the platform came from a content
match, not the filename itself. Nothing here is confirmed against the live
platforms.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from mebeauty_benchmark.legacy.checksums import group_files_by_sha256
from mebeauty_benchmark.legacy.provenance import ImageProvenance, infer_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def propagate_across_duplicates(
    provenance_by_path: dict[str, ImageProvenance], images_root: Path
) -> int:
    """Fill in `unknown` rows whose file is byte-identical to a resolved one.

    Mutates `provenance_by_path` in place. Returns how many rows were
    resolved this way.
    """
    resolved = 0
    for paths in group_files_by_sha256(images_root).values():
        relative_paths = [p.relative_to(images_root).as_posix() for p in paths]
        known = next(
            (
                provenance_by_path[p]
                for p in relative_paths
                if provenance_by_path[p].platform != "unknown"
            ),
            None,
        )
        if known is None:
            continue
        for relative_path in relative_paths:
            if provenance_by_path[relative_path].platform == "unknown":
                provenance_by_path[relative_path] = ImageProvenance(
                    platform=known.platform,
                    photo_id=known.photo_id,
                    inferred_source_url=known.inferred_source_url,
                    confidence="inferred-via-duplicate",
                )
                resolved += 1
    return resolved


def main() -> None:
    args = parse_args()
    images_root = Path(args.legacy_copy).expanduser().resolve() / "original_images"
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    provenance_by_path: dict[str, ImageProvenance] = {}
    for path in sorted(images_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(images_root).as_posix()
        provenance_by_path[relative] = infer_provenance(path.name)

    resolved_via_duplicate = propagate_across_duplicates(
        provenance_by_path, images_root
    )

    rows = []
    platform_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    for relative, provenance in sorted(provenance_by_path.items()):
        platform_counts[provenance.platform] += 1
        confidence_counts[provenance.confidence] += 1
        rows.append(
            {
                "image": relative,
                "inferred_platform": provenance.platform,
                "inferred_photo_id": provenance.photo_id or "",
                "inferred_source_url": provenance.inferred_source_url or "",
                "confidence": provenance.confidence,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "inferred_platform",
                "inferred_photo_id",
                "inferred_source_url",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} images tagged")
    for platform, count in platform_counts.most_common():
        print(f"  {platform}: {count}")
    print(
        f"\n{resolved_via_duplicate} additionally resolved via a content-duplicate match"
    )
    for confidence, count in confidence_counts.most_common():
        print(f"  {confidence}: {count}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
