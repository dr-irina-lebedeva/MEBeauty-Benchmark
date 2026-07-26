"""File hashing for the copy manifest."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_files_by_sha256(root: Path) -> dict[str, list[Path]]:
    """Hash every file under `root` and group paths that are byte-identical.

    Filename-based duplicate checks miss copies of the same image saved
    under a different name (e.g. a stock photo re-downloaded with its
    site slug instead of its original numeric name). Grouping by content
    hash catches those regardless of filename.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            groups[sha256_file(path)].append(path)
    return dict(groups)
