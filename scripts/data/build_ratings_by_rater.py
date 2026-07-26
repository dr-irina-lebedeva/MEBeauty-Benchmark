"""Build a per-rater ratings table, reconciled against the v3 image identity.

The pseudonymized per-rater score files (`data/pseudonymized_scores/`, from
`pseudonymize_raters.py`) key images by **bare basename only** -- no
folder, no path. That's the same ambiguity that turned out to matter for
the aggregate ratings (Finding 6). Checked directly against v3's metadata:
only 1 basename in the whole dataset is genuinely ambiguous
(`shivam-singh-2_X6NMP-E_U-unsplash.jpg`, the same two-different-photos
case already documented) -- so basename joining is safe here, with that one
exception dropped explicitly rather than guessed at.

Source selection, after checking for overlap/redundancy between files:

- `public_generic/*.xlsx`, `public_date/*.xlsx` (long format: image, rater,
  label) -- used. These are the most complete source: `public_generic_all.xlsx`
  (the corresponding wide-format aggregate) turned out to be a near-total
  subset of the long-format files (61,300 of 61,372 rows overlap; the long
  format has 76,617), so using both would double-count. `public_generic_all.xlsx`
  / `public_date_all.xlsx` are therefore skipped.
- `generic_scores_all.xlsx` (wide format) -- used; a distinct rating batch,
  not redundant with `public_generic/*`. `generic_scores_all_2022.xlsx` is
  skipped: confirmed to be a 100% subset of it (53,895 of 53,895 rows
  overlap).
- `date_scores_all.xlsx` / `date_scores_all_2022.xlsx` -- **skipped
  entirely**. Both have corrupted column headers (e.g. a column literally
  named `labelcaucasian_female_29.xlsx`, not a rater ID) -- a legacy data
  problem, not a parsing bug here. `public_date/*.xlsx` already covers
  "date"-context ratings cleanly, so this is a loss of one redundant,
  broken source, not a coverage gap.

    uv run python scripts/data/build_ratings_by_rater.py \\
        --pseudonymized data/pseudonymized_scores \\
        --v3-metadata data/mebeauty_v3/metadata.parquet \\
        --output data/mebeauty_v3/ratings/by_rater
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pseudonymized", required=True, help="data/pseudonymized_scores directory"
    )
    parser.add_argument(
        "--v3-metadata", required=True, help="Path to v3 metadata.parquet"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def build_basename_lookup(metadata_df: pd.DataFrame) -> dict[str, str | None]:
    """basename (lowercased) -> image_id, or None if the basename is ambiguous."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in metadata_df.itertuples():
        candidates[row.legacy_filename.lower()].add(row.image_id)
        if row.other_legacy_paths:
            for path in row.other_legacy_paths.split(", "):
                candidates[path.rsplit("/", 1)[-1].lower()].add(row.image_id)
    return {
        basename: (next(iter(ids)) if len(ids) == 1 else None)
        for basename, ids in candidates.items()
    }


def load_long_format(directory: Path, source_label: str) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.xlsx")):
        df = pd.read_excel(path)[["image", "rater", "label"]].rename(
            columns={"label": "score"}
        )
        df["image"] = df["image"].astype(str).str.rsplit("/", n=1).str[-1]
        frames.append(df)
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["image", "rater", "score"])
    )
    result["source"] = source_label
    return result


def load_wide_format(path: Path, source_label: str) -> pd.DataFrame:
    wide = pd.read_excel(path)
    rater_cols = [c for c in wide.columns if str(c).startswith("rater_")]
    melted = wide.melt(
        id_vars=["image"], value_vars=rater_cols, var_name="rater", value_name="score"
    )
    melted = melted.dropna(subset=["score"])
    melted["source"] = source_label
    return melted


def main() -> None:
    args = parse_args()
    pseudonymized_dir = Path(args.pseudonymized).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_parquet(args.v3_metadata)
    basename_to_image_id = build_basename_lookup(metadata_df)

    frames = [
        load_long_format(pseudonymized_dir / "public_generic", "public_generic"),
        load_long_format(pseudonymized_dir / "public_date", "public_date"),
        load_wide_format(
            pseudonymized_dir / "generic_scores_all.xlsx", "generic_scores_all"
        ),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["image"] = combined["image"].str.lower()
    print(f"Raw rows from selected sources: {len(combined)}")

    # The legacy collection ran TWO distinct rating tasks over the same images,
    # and their score distributions differ by about a full point (generic mean
    # 5.99, date mean 5.02). Pooling them into one undifferentiated `score`
    # column would silently mix two different questions -- see
    # docs/DATASET_AUDIT.md Finding 16.
    combined["rating_type"] = combined["source"].map(
        {
            "public_generic": "generic",
            "generic_scores_all": "generic",
            "public_date": "date",
        }
    )
    unmapped_sources = combined.loc[combined["rating_type"].isna(), "source"].unique()
    if len(unmapped_sources):
        raise ValueError(
            f"source with no rating_type mapping: {list(unmapped_sources)}"
        )

    combined["image_id"] = combined["image"].map(basename_to_image_id)
    not_found = combined[~combined["image"].isin(basename_to_image_id)]
    ambiguous = combined[
        combined["image"].isin(basename_to_image_id) & combined["image_id"].isna()
    ]
    resolved = combined.dropna(subset=["image_id"])

    # Dedupe within a rating task, not across tasks: one rater answering both
    # the generic and the date question about the same image gave two distinct
    # answers, not a duplicate. Keying on (image_id, rater) alone -- as this
    # previously did -- discarded whichever task came second in `frames` order,
    # silently keeping a generic-vs-date mixture that depended on load order.
    deduped = resolved.drop_duplicates(
        subset=["image_id", "rater", "rating_type"], keep="first"
    )
    dupes_dropped = len(resolved) - len(deduped)

    out = deduped[["image_id", "rater", "rating_type", "score"]].rename(
        columns={"rater": "rater_id"}
    )
    out.to_parquet(output_dir / "ratings_by_rater.parquet", index=False)

    summary = {
        "raw_rows": len(combined),
        "not_found_dropped": len(not_found),
        "ambiguous_dropped": len(ambiguous),
        "within_task_duplicate_rows_dropped": int(dupes_dropped),
        "final_rows": len(out),
        "unique_images": int(out["image_id"].nunique()),
        "unique_raters": int(out["rater_id"].nunique()),
        "by_rating_type": {
            str(k): {
                "rows": len(g),
                "images": int(g["image_id"].nunique()),
                "raters": int(g["rater_id"].nunique()),
                "mean_score": round(float(g["score"].mean()), 4),
            }
            for k, g in out.groupby("rating_type")
        },
        "excluded_sources": [
            "date_scores_all.xlsx (corrupted column headers)",
            "date_scores_all_2022.xlsx (corrupted column headers)",
            "generic_scores_all_2022.xlsx (100% subset of generic_scores_all.xlsx)",
            "public_generic_all.xlsx (near-total subset of public_generic/*.xlsx)",
            "public_date_all.xlsx (near-total subset of public_date/*.xlsx)",
        ],
    }
    (output_dir / "reconciliation_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
