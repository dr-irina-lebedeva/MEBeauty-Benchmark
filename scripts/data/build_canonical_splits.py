"""Build a canonical train/val/test split from a legacy split generation.

Of the three split generations in the legacy repo (`train/val/test.txt`,
`train/val/test_2022.txt`, `train_crop.csv`/`test_crop.csv`), `--generation
2022` (the default) covers the most images and is the one used for the
`benchmark-v1` canonical split in `data/mebeauty_v3/`. The other two are
supported here too (`--generation original`, `--generation crop`) so all
three get the same rigor rather than the informal one-off counts in
`docs/DATASET_AUDIT.md`'s Finding 6 comparison table.

Three checks run against whichever generation is selected, in order:

1. Existence resolution (`resolve_to_existing_images`) -- ratings are raw
   text, never checked against the actual `original_images/` tree. Some
   rated images have since been reclassified into a different
   ethnicity/gender folder (remapped to their current path); some no
   longer exist anywhere (dropped, since there is nothing to join the
   rating to). This also depends on `parse_split_lines` stripping quotes
   from paths with spaces (`"foo (1).jpg" 5.5`) -- unstripped, the leading
   `"` breaks every prefix check silently.
2. Exact-path dedup (`dedupe_across_splits`) -- catches the same normalized
   image path appearing twice, whether within one split or across splits.
   Keyed on the full path, not the bare basename: this dataset has one
   confirmed case of two different photos sharing an identical filename in
   different folders, which a basename-only key would wrongly conflate.
3. Content-hash dedup (`dedupe_by_content_hash`) -- catches the same photo
   saved under two *different* filenames (e.g. a stock image re-downloaded
   with its site slug instead of its original numeric name, or the same
   photo cross-filed under two label folders), which the path check cannot
   see at all. Comparing SHA-256 hashes of every file under
   `original_images/` against this split found leakage invisible to a
   path-only check. See `docs/DATASET_AUDIT.md` for the full breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from mebeauty_benchmark.legacy.checksums import group_files_by_sha256
from mebeauty_benchmark.legacy.paths import normalize_image_path
from mebeauty_benchmark.legacy.splits import (
    Rating,
    dedupe_across_splits,
    dedupe_by_content_hash,
    parse_split_lines,
    resolve_to_existing_images,
)

GENERATIONS: dict[str, dict[str, object]] = {
    "2022": {
        "files": {
            "train": "train_2022.txt",
            "val": "val_2022.txt",
            "test": "test_2022.txt",
        },
        "format": "txt",
    },
    "original": {
        "files": {"train": "train.txt", "val": "val.txt", "test": "test.txt"},
        "format": "txt",
    },
    "crop": {
        "files": {"train": "train_crop.csv", "test": "test_crop.csv"},
        "format": "csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--generation",
        default="2022",
        choices=sorted(GENERATIONS),
        help="Which split generation to build",
    )
    parser.add_argument(
        "--output", required=True, help="Directory for the canonical split CSVs"
    )
    parser.add_argument(
        "--report-out", required=True, help="Where to write the dedup report"
    )
    return parser.parse_args()


def read_split_lines(path: Path, file_format: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if file_format == "csv":
        # "image,score" header + comma-separated rows -> reuse the same
        # "<path> <score>" parser by swapping the last comma for a space.
        lines = lines[1:]  # drop header
        converted = []
        for line in lines:
            if not line.strip():
                continue
            image_path, _, score = line.rpartition(",")
            converted.append(f"{image_path} {score}")
        lines = converted
    return lines


def build_path_to_hash(
    splits: dict[str, list[Rating]], images_root: Path
) -> dict[str, str]:
    """Map each raw split path to the SHA-256 of its `original_images/` file, where found."""
    hash_by_relative_path: dict[str, str] = {}
    for content_hash, paths in group_files_by_sha256(images_root).items():
        for path in paths:
            hash_by_relative_path[path.relative_to(images_root).as_posix()] = (
                content_hash
            )

    path_to_hash: dict[str, str] = {}
    for ratings in splits.values():
        for rating in ratings:
            content_hash = hash_by_relative_path.get(normalize_image_path(rating.path))
            if content_hash is not None:
                path_to_hash[rating.path] = content_hash
    return path_to_hash


def build_image_lookup(images_root: Path) -> tuple[set[str], dict[str, list[str]]]:
    valid_paths: set[str] = set()
    basename_to_paths: dict[str, list[str]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(images_root).as_posix()
        valid_paths.add(relative)
        basename_to_paths[path.name.lower()].append(relative)
    return valid_paths, dict(basename_to_paths)


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    scores_dir = legacy_root / "scores"
    images_root = legacy_root / "original_images"
    output_dir = Path(args.output).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    generation = GENERATIONS[args.generation]
    split_files: dict[str, str] = generation["files"]  # type: ignore[assignment]
    file_format: str = generation["format"]  # type: ignore[assignment]
    priority = list(split_files)

    raw_splits: dict[str, list[Rating]] = {}
    for split_name, filename in split_files.items():
        path = scores_dir / filename
        raw_splits[split_name] = parse_split_lines(read_split_lines(path, file_format))
    raw_count = sum(len(ratings) for ratings in raw_splits.values())

    valid_paths, basename_to_paths = build_image_lookup(images_root)
    existing, existence_issues = resolve_to_existing_images(
        raw_splits, valid_paths, basename_to_paths
    )

    by_path, removed_path = dedupe_across_splits(existing, priority=priority)

    path_to_hash = build_path_to_hash(by_path, images_root)
    deduped, removed_content = dedupe_by_content_hash(
        by_path, priority=priority, path_to_hash=path_to_hash
    )

    counts = {}
    for split_name in priority:
        dest = output_dir / f"{split_name}.csv"
        with dest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "score"])
            for rating in deduped[split_name]:
                writer.writerow([normalize_image_path(rating.path), rating.score])
        counts[split_name] = len(deduped[split_name])
        print(f"{split_name}: {counts[split_name]} rows -> {dest}")

    report = {
        "source_generation": args.generation,
        "priority_order": priority,
        "raw_row_count": raw_count,
        "counts": counts,
        "existence_resolution": existence_issues,
        "existence_resolution_summary": {
            "relabeled": sum(1 for i in existence_issues if i["status"] == "relabeled"),
            "ambiguous_dropped": sum(
                1 for i in existence_issues if i["status"] == "ambiguous"
            ),
            "not_found_dropped": sum(
                1 for i in existence_issues if i["status"] == "not_found"
            ),
        },
        "removed_for_exact_path_leakage": [
            {"normalized_image": normalized_image, "dropped_from_split": split_name}
            for normalized_image, split_name in removed_path
        ],
        "removed_for_content_duplicate_leakage": [
            {
                "path": path,
                "normalized_image": normalize_image_path(path),
                "sha256": content_hash,
                "dropped_from_split": split_name,
            }
            for path, content_hash, split_name in removed_content
        ],
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nExistence resolution: {report['existence_resolution_summary']}")
    print(f"Removed {len(removed_path)} exact-path-duplicate row(s): {removed_path}")
    print(
        f"Removed {len(removed_content)} content-duplicate row(s) (same photo, different filename)"
    )
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
