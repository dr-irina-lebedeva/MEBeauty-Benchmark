"""Guard: no raw AMT Worker ID may ever appear in committed content.

This exists because it already happened. `tests/legacy/test_raters.py` used
five *real* Worker IDs as fixtures -- each resolving to a real rater in
`data/rater_mapping.LOCAL_ONLY.csv` -- which directly contradicted the
project's own published guarantee that raw Worker IDs are pseudonymized and
never committed. It was caught only by scanning staged content against the
mapping immediately before the first commit, i.e. by luck and timing rather
than by any check.

The mapping file is gitignored and local-only, so this test is a no-op for
anyone who does not have it (CI, a fresh clone). That is deliberate: the test
cannot leak what it is protecting. On a machine that *does* have the mapping
-- the only machine where a fixture could be copied from real data in the
first place -- it fails loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING = REPO_ROOT / "data" / "rater_mapping.LOCAL_ONLY.csv"

# AMT Worker IDs: "A" followed by ten or more uppercase alphanumerics.
_WORKER_ID = re.compile(r"\bA[A-Z0-9]{10,}\b")

_SEARCH_ROOTS = ("reports", "docs", "scripts", "src", "tests")
_SEARCH_FILES = ("README.md", "CITATION.cff", "Makefile", "pyproject.toml")


def _committed_files() -> list[Path]:
    found = [
        path
        for root in _SEARCH_ROOTS
        for path in (REPO_ROOT / root).rglob("*")
        if path.is_file() and path.suffix not in {".parquet", ".pyc"}
    ]
    found += [
        REPO_ROOT / name for name in _SEARCH_FILES if (REPO_ROOT / name).is_file()
    ]
    return found


def _real_worker_ids() -> set[str]:
    import csv

    with MAPPING.open(encoding="utf-8") as handle:
        return {row[0].strip() for row in csv.reader(handle) if row and row[0].strip()}


@pytest.mark.skipif(
    not MAPPING.is_file(),
    reason="local-only rater mapping absent; nothing to compare against",
)
def test_no_real_worker_id_appears_in_committed_content():
    real_ids = _real_worker_ids()
    assert real_ids, "mapping file present but empty -- the guard would be vacuous"

    offenders: list[str] = []
    for path in _committed_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for candidate in set(_WORKER_ID.findall(text)):
            if candidate in real_ids:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {candidate}")

    assert not offenders, (
        "Real AMT Worker IDs found in committed content. Replace them with "
        "synthetic fixtures (e.g. AEXAMPLEWORKER01):\n  " + "\n  ".join(offenders)
    )
