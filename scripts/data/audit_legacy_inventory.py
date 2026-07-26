"""Create a read-only inventory of the legacy MEBeauty repository."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Legacy repository path")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Legacy directory not found: {root}")

    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    extensions: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    total_size = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue

        relative = path.relative_to(root)
        size = path.stat().st_size
        suffix = path.suffix.lower() or "<no-extension>"

        total_size += size
        extensions[suffix] += 1
        top_level[relative.parts[0]] += 1

        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": size,
                "extension": suffix,
            }
        )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["relative_path", "size_bytes", "extension"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "file_count_excluding_git": len(rows),
        "total_size_bytes": total_size,
        "top_level_file_counts": dict(sorted(top_level.items())),
        "extension_counts": dict(
            sorted(extensions.items(), key=lambda item: (-item[1], item[0]))
        ),
    }

    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
