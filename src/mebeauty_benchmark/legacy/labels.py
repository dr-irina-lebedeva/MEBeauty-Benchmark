"""Detect images filed under more than one ethnicity/gender label.

The legacy `original_images/` tree encodes ethnicity and gender as folder
names. A handful of files exist twice, under the same filename, in two
different label folders. Which label is correct is a judgment call for
the dataset maintainer, so this only reports the conflicts — it never
picks one automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelConflict:
    filename: str
    label_paths: tuple[Path, ...]


def find_cross_label_collisions(images_root: Path) -> list[LabelConflict]:
    by_filename: dict[str, list[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if path.is_file():
            by_filename[path.name.lower()].append(path)

    conflicts = [
        LabelConflict(filename=paths[0].name, label_paths=tuple(sorted(paths)))
        for paths in by_filename.values()
        if len(paths) > 1
    ]
    return sorted(conflicts, key=lambda c: c.filename)
