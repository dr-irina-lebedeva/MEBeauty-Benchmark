"""Report images filed under more than one ethnicity/gender label folder.

The legacy `original_images/` tree encodes ethnicity and gender as folder
names. A handful of files exist twice, under the same filename, in two
different label folders. Which label is correct is a judgment call for the
dataset maintainer, so this only reports the conflicts -- it never picks one
automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mebeauty_benchmark.legacy.labels import find_cross_label_collisions


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

    conflicts = find_cross_label_collisions(root)

    report = {
        # Recorded relative to the repository root, never as an absolute path:
        # this report is committed, and an absolute path would leak the
        # maintainer's home directory into the public repository.
        "root": _repo_relative(root),
        "conflict_count": len(conflicts),
        "conflicts": [
            {
                "filename": conflict.filename,
                "label_paths": [
                    path.relative_to(root).as_posix() for path in conflict.label_paths
                ],
            }
            for conflict in conflicts
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{len(conflicts)} cross-label collision(s) found")
    for conflict in conflicts:
        print(
            f"  {conflict.filename}: {[p.relative_to(root).as_posix() for p in conflict.label_paths]}"
        )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
