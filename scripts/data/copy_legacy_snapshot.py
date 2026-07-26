"""Copy the read-only legacy MEBeauty repository into the working data dir.

The source at ~/Research/MEBeauty-Legacy is never modified. This writes a
plain file-for-file copy plus a SHA-256 manifest so later steps can prove
what they started from.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mebeauty_benchmark.legacy.checksums import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Legacy repository path")
    parser.add_argument(
        "--dest", required=True, help="Destination directory (will be created)"
    )
    return parser.parse_args()


def legacy_commit_sha(source: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Legacy directory not found: {source}")

    if dest.exists():
        raise FileExistsError(
            f"{dest} already exists — remove it first to make a fresh, verifiable copy"
        )

    def ignore_git(directory: str, names: list[str]) -> list[str]:
        return [".git"] if ".git" in names else []

    print(f"Copying {source} -> {dest}")
    shutil.copytree(source, dest, ignore=ignore_git)

    manifest_rows = []
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(dest).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    manifest_path = dest.parent / f"{dest.name}.manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["relative_path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    provenance = {
        "source": str(source),
        "source_git_commit": legacy_commit_sha(source),
        "copied_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(manifest_rows),
        "manifest": manifest_path.name,
    }
    provenance_path = dest.parent / f"{dest.name}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(json.dumps(provenance, indent=2))
    print(f"\nManifest:   {manifest_path}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
