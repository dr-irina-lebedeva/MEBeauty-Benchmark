"""Find byte-identical images filed under different filenames.

`find_label_collisions.py` only catches the same *filename* appearing in
two label folders. It misses a stock photo saved twice under different
names (e.g. a numeric legacy name and a re-downloaded site slug) -- those
are undetectable by filename alone. This hashes every file under
`--root` and reports any group of two or more byte-identical files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mebeauty_benchmark.legacy.checksums import group_files_by_sha256


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root, so committed reports carry no absolute paths."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Path to original_images/ in the copied legacy repo",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = group_files_by_sha256(root)
    duplicate_groups = sorted(
        (paths for paths in groups.values() if len(paths) > 1),
        key=lambda paths: paths[0].relative_to(root).as_posix(),
    )

    report = {
        # Recorded relative to the repository root, never as an absolute path:
        # this report is committed, and an absolute path would leak the
        # maintainer's home directory into the public repository.
        "root": _repo_relative(root),
        "duplicate_group_count": len(duplicate_groups),
        "redundant_file_count": sum(len(paths) - 1 for paths in duplicate_groups),
        "groups": [
            {
                "sha256": next(h for h, p in groups.items() if p == paths),
                "paths": [path.relative_to(root).as_posix() for path in paths],
            }
            for paths in duplicate_groups
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{len(duplicate_groups)} content-duplicate group(s), {report['redundant_file_count']} redundant file(s)"
    )
    for entry in report["groups"]:
        print(f"  {entry['paths']}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
