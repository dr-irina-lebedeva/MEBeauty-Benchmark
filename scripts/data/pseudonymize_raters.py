"""Scrub raw Mechanical Turk Worker IDs out of the copied legacy scores.

Worker IDs appear as wide-format column headers in the four
``*_scores_all*.xlsx`` workbooks and as long-format ``rater`` column
values in ``scores/public_generic/*.xlsx`` and ``scores/public_date/*.xlsx``.
This builds one global worker_id -> rater_XXXX mapping across every file
so the same person gets the same pseudonym everywhere, writes
pseudonymized copies of the workbooks, and keeps the real mapping in a
local-only file that is never committed or uploaded (see Finding 1 of the
dataset audit).

Run `copy_legacy_snapshot.py` first; this script only ever reads from the
working copy, never from ~/Research/MEBeauty-Legacy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.raters import (
    build_pseudonym_mapping,
    collect_worker_ids,
    collect_worker_ids_from_value_column,
    pseudonymize_columns,
    pseudonymize_value_column,
)

WIDE_FORMAT_FILES = [
    "generic_scores_all_2022.xlsx",
    "date_scores_all_2022.xlsx",
    "generic_scores_all.xlsx",
    "date_scores_all.xlsx",
    "public_generic_all.xlsx",
    "public_date_all.xlsx",
]
LONG_FORMAT_DIRS = ["public_generic", "public_date"]
LONG_FORMAT_VALUE_COLUMN = "rater"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--output", required=True, help="Where to write pseudonymized workbooks"
    )
    parser.add_argument(
        "--mapping-out",
        required=True,
        help="Local-only file for the real worker_id -> pseudonym mapping (never commit this)",
    )
    parser.add_argument(
        "--report-out", required=True, help="Where to write the summary report"
    )
    return parser.parse_args()


def load_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None)


def main() -> None:
    args = parse_args()
    legacy_scores = Path(args.legacy_copy).expanduser().resolve() / "scores"
    output_scores = Path(args.output).expanduser().resolve()
    mapping_out = Path(args.mapping_out).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()

    output_scores.mkdir(parents=True, exist_ok=True)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    wide_workbooks: dict[str, dict[str, pd.DataFrame]] = {}
    long_workbooks: dict[str, pd.DataFrame] = {}
    all_worker_ids: set[str] = set()

    for name in WIDE_FORMAT_FILES:
        path = legacy_scores / name
        if not path.is_file():
            print(f"skip (not found): {name}")
            continue
        sheets = load_all_sheets(path)
        wide_workbooks[name] = sheets
        for sheet_name, frame in sheets.items():
            found = collect_worker_ids(str(c) for c in frame.columns)
            all_worker_ids |= found
            print(f"{name}::{sheet_name}: {len(found)} worker-id columns")

    for sub_dir in LONG_FORMAT_DIRS:
        for path in sorted((legacy_scores / sub_dir).glob("*.xlsx")):
            frame = pd.read_excel(path)
            relative_key = f"{sub_dir}/{path.name}"
            long_workbooks[relative_key] = frame
            if LONG_FORMAT_VALUE_COLUMN in frame.columns:
                found = collect_worker_ids_from_value_column(
                    frame[LONG_FORMAT_VALUE_COLUMN]
                )
                all_worker_ids |= found

    mapping = build_pseudonym_mapping(all_worker_ids)
    print(f"\n{len(mapping)} unique worker IDs found across all files")

    report_rows = []

    for name, sheets in wide_workbooks.items():
        dest = output_scores / name
        with pd.ExcelWriter(dest) as writer:
            for sheet_name, frame in sheets.items():
                before = collect_worker_ids(str(c) for c in frame.columns)
                cleaned = pseudonymize_columns(frame, mapping)
                cleaned.to_excel(writer, sheet_name=sheet_name, index=False)
                report_rows.append(
                    {
                        "file": name,
                        "sheet": sheet_name,
                        "worker_ids_replaced": len(before),
                    }
                )
        print(f"wrote {dest}")

    for relative_key, frame in long_workbooks.items():
        dest = output_scores / relative_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        cleaned = frame.copy()
        replaced = 0
        if LONG_FORMAT_VALUE_COLUMN in cleaned.columns:
            before = collect_worker_ids_from_value_column(
                cleaned[LONG_FORMAT_VALUE_COLUMN]
            )
            replaced = len(before)
            cleaned[LONG_FORMAT_VALUE_COLUMN] = pseudonymize_value_column(
                cleaned[LONG_FORMAT_VALUE_COLUMN], mapping
            )
        cleaned.to_excel(dest, index=False)
        report_rows.append(
            {"file": relative_key, "sheet": "Sheet1", "worker_ids_replaced": replaced}
        )

    mapping_out.write_text(
        "\n".join(
            f"{worker_id},{pseudonym}" for worker_id, pseudonym in mapping.items()
        ),
        encoding="utf-8",
    )

    report = {
        "total_unique_worker_ids": len(mapping),
        "files_processed": report_rows,
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    remaining = set()
    for sheets in wide_workbooks.values():
        for frame in sheets.values():
            remaining |= collect_worker_ids(str(c) for c in frame.columns) - set(
                mapping
            )
    assert not remaining, f"unmapped worker IDs left behind: {remaining}"

    print(f"\nPseudonymized workbooks: {output_scores}")
    print(f"Local-only mapping (do not commit/upload): {mapping_out}")
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
