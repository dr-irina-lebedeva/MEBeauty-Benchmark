"""Pseudonymize raw Amazon Mechanical Turk Worker IDs.

Worker IDs show up in the legacy scores in two shapes:

* wide format — one column per rater, headed by the raw Worker ID
  (``generic_scores_all*.xlsx``, ``date_scores_all*.xlsx``,
  ``public_generic_all.xlsx``, ``public_date_all.xlsx``);
* long format — a ``rater`` column holding the raw Worker ID as a row
  value (``scores/public_generic/*.xlsx``, ``scores/public_date/*.xlsx``).

The small in-house panel's columns are demographic codes like
``asian_female_25`` and are already anonymous. Worker IDs are persistent,
potentially re-identifiable participant identifiers and must never be
published in either shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

_AMT_WORKER_ID_RE = re.compile(r"^A[A-Z0-9]{9,}$")

PSEUDONYM_PREFIX = "rater_"


def is_amt_worker_id(column_name: str) -> bool:
    """True if `column_name` looks like a raw AMT Worker ID.

    AMT Worker IDs start with ``A`` followed by ten or more uppercase
    letters/digits. The legacy in-house panel columns (e.g.
    ``asian_female_25``, ``mean``, ``path``) are lowercase or mixed and
    never match.
    """
    return bool(_AMT_WORKER_ID_RE.match(column_name.strip()))


def collect_worker_ids(column_names: Iterable[str]) -> set[str]:
    return {name for name in column_names if is_amt_worker_id(name)}


def build_pseudonym_mapping(worker_ids: Iterable[str]) -> dict[str, str]:
    """Deterministic worker_id -> rater_XXXX mapping, sorted for stability.

    Sorting before assigning numbers means the same input set always
    produces the same mapping, independent of dict/set iteration order.
    """
    ordered = sorted(set(worker_ids))
    width = max(4, len(str(len(ordered))))
    return {
        worker_id: f"{PSEUDONYM_PREFIX}{index:0{width}d}"
        for index, worker_id in enumerate(ordered, start=1)
    }


def pseudonymize_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Return a copy of `frame` with AMT Worker ID columns renamed via `mapping`."""
    rename_map = {
        column: mapping[column]
        for column in frame.columns
        if isinstance(column, str) and column in mapping
    }
    return frame.rename(columns=rename_map)


def collect_worker_ids_from_value_column(series: pd.Series) -> set[str]:
    """Collect Worker IDs stored as row values (long-format ``rater`` columns)."""
    return {value for value in series.dropna().astype(str) if is_amt_worker_id(value)}


def pseudonymize_value_column(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Replace Worker-ID values in a long-format ``rater`` column via `mapping`."""
    return series.astype(str).map(lambda value: mapping.get(value, value))
