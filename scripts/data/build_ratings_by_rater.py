"""Build a per-rater ratings table, reconciled against the v3 image identity.

The pseudonymized per-rater score files (`data/pseudonymized_scores/`, from
`pseudonymize_raters.py`) key images by **bare basename only** -- no
folder, no path. That's the same ambiguity that turned out to matter for
the aggregate ratings (Finding 6). Checked directly against v3's metadata:
only 1 basename in the whole dataset is genuinely ambiguous
(`shivam-singh-2_X6NMP-E_U-unsplash.jpg`, the same two-different-photos
case already documented) -- so basename joining is safe here, with that one
exception dropped explicitly rather than guessed at.

**Only the `generic` attractiveness task ships.** The legacy collection also
ran a `date` task ("would you date this person"), a different question whose
mean sits about a point lower. It never enters a label, so carrying it in the
released table only invited the pooling mistake Finding 16 already caught
once. Its raw files remain in the legacy repository for anyone who wants them.

Two sources, chosen after measuring overlap rather than assuming it:

- `public_generic/*.xlsx` (long format: image, rater, label) -- the crowd
  pool. `public_generic_all.xlsx` is a near-total subset of these and is
  skipped, as is `generic_scores_all.xlsx`: 56,751 of its 56,816 image-rater
  pairs already appear here and **no rater is unique to it**, so it adds 65
  ratings and a second identifier convention.
- `private_generic/*.xlsx` -- the in-house panel, one file per rater, read
  from the legacy snapshot. Their identifiers are demographic codes, which is
  exactly why earlier builds' `rater_`-prefix filter dropped them silently.
  `private_generic_all.xlsx` is a wide merge of these and is skipped.

Outputs `ratings_by_rater.parquet` (image_id, rater_id, score) and
`rater_demographics.parquet` (panel members only -- crowd workers have no
demographic data).

    uv run python scripts/data/build_ratings_by_rater.py \\
        --pseudonymized data/pseudonymized_scores \\
        --v3-metadata data/mebeauty_v3/images/metadata.parquet \\
        --legacy-copy data/legacy_snapshot \\
        --output data/mebeauty_v3/ratings/by_rater
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.panel import (
    PanelRater,
    assign_pseudonyms,
    build_roster,
    canonical_panel_id,
)
from mebeauty_benchmark.legacy.validity import attach_validity

#: The in-house panel's own per-rater files for the generic task. Never
#: ingested before: panel identifiers are demographic codes, not `rater_`
#: pseudonyms, so the previous prefix filter skipped them silently.
PANEL_DIR = "private_generic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pseudonymized", required=True, help="data/pseudonymized_scores directory"
    )
    parser.add_argument(
        "--v3-metadata", required=True, help="Path to v3 metadata.parquet"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--legacy-copy",
        required=True,
        help="data/legacy_snapshot -- source of the in-house panel's own files",
    )
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


def load_panel_long_format(
    directory: Path,
    source_label: str,
    roster: dict[str, PanelRater],
    pseudonyms: dict[str, str],
) -> pd.DataFrame:
    """Load the in-house panel's per-rater files, where the *filename* is the rater.

    Only the individual per-rater files are read. `private_date_female.xlsx`
    and `private_date_male.xlsx` are wide merges of those same files and would
    double-count every rating in them.
    """
    frames = []
    for path in sorted(directory.glob("*.xlsx")):
        canonical = canonical_panel_id(path.name, roster)
        if canonical is None:
            print(f"  skipping non-panel file {path.name}")
            continue
        df = pd.read_excel(path)
        if not {"image", "label"} <= set(df.columns):
            raise ValueError(f"{path.name}: expected long-format image/label columns")
        df = df[["image", "label"]].rename(columns={"label": "score"}).dropna()
        df["rater"] = pseudonyms[canonical]
        frames.append(df)
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["image", "rater", "score"])
    )
    result["source"] = source_label
    return result


def resolve_stems_to_filenames(
    images: pd.Series, basename_lookup: dict[str, str | None]
) -> pd.Series:
    """Map extension-less image stems onto the basenames the lookup expects.

    The panel files store `girl-1763686_1920`, the lookup is keyed by
    `girl-1763686_1920.jpg`. Bare numeric stems like `0` are real filenames in
    this dataset (`0.jpg`), not row indices -- see Finding 17.
    """
    known = set(basename_lookup)

    def resolve(stem: str) -> str:
        if stem in known:
            return stem
        for extension in (".jpg", ".png", ".jpeg"):
            candidate = f"{stem}{extension}"
            if candidate in known:
                return candidate
        return stem

    return images.map(resolve)


def main() -> None:
    args = parse_args()
    pseudonymized_dir = Path(args.pseudonymized).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_parquet(args.v3_metadata)
    basename_to_image_id = build_basename_lookup(metadata_df)

    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    panel_root = legacy_root / "scores" / PANEL_DIR
    roster = build_roster([path.name for path in panel_root.glob("*.xlsx")])
    pseudonyms = assign_pseudonyms(roster)
    print(f"In-house panel: {len(roster)} members")

    frames = [
        load_long_format(pseudonymized_dir / "public_generic", "public_generic"),
        load_panel_long_format(panel_root, PANEL_DIR, roster, pseudonyms),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["image"] = combined["image"].astype(str).str.lower()
    combined["image"] = resolve_stems_to_filenames(
        combined["image"], basename_to_image_id
    )
    print(f"Raw rows from selected sources: {len(combined)}")

    combined["image_id"] = combined["image"].map(basename_to_image_id)
    not_found = combined[~combined["image"].isin(basename_to_image_id)]
    ambiguous = combined[
        combined["image"].isin(basename_to_image_id) & combined["image_id"].isna()
    ]
    resolved = combined.dropna(subset=["image_id"])

    # One rating per (image, rater). Only the generic task is ingested, so
    # unlike the earlier generic+date build there is no second answer from the
    # same rater to preserve -- a repeat here is a genuine duplicate.
    deduped = resolved.drop_duplicates(subset=["image_id", "rater"], keep="first")
    dupes_dropped = len(resolved) - len(deduped)

    out = deduped[["image_id", "rater", "score"]].rename(columns={"rater": "rater_id"})
    out = out.sort_values(["image_id", "rater_id"]).reset_index(drop=True)
    # Flag rather than drop: every rating ships, and the filter behind the
    # canonical score stays auditable and reversible.
    out = attach_validity(out)
    out.to_parquet(output_dir / "ratings_by_rater.parquet", index=False)

    # Panel demographics, kept separate from the identifier so the id stays
    # opaque. MTurk raters have no demographic data of any kind, so they are
    # absent here rather than carrying null columns.
    demographics = pd.DataFrame(
        [
            {
                "rater_id": pseudonyms[canonical],
                "ethnicity": rater.ethnicity,
                "gender": rater.gender,
                "age": rater.age,
            }
            for canonical, rater in sorted(roster.items())
            if pseudonyms[canonical] in set(out["rater_id"])
        ]
    ).sort_values("rater_id")
    demographics.to_parquet(output_dir / "rater_demographics.parquet", index=False)
    print(f"Panel demographics: {len(demographics)} raters")

    summary = {
        "raw_rows": len(combined),
        "not_found_dropped": len(not_found),
        "ambiguous_dropped": len(ambiguous),
        "within_task_duplicate_rows_dropped": int(dupes_dropped),
        "final_rows": len(out),
        "unique_images": int(out["image_id"].nunique()),
        "unique_raters": int(out["rater_id"].nunique()),
        "rating_task": "generic",
        "mean_score": round(float(out["score"].mean()), 4),
        "crowd_raters": int(
            out.loc[out["rater_id"].str.startswith("rater_"), "rater_id"].nunique()
        ),
        "panel_raters": int(
            out.loc[out["rater_id"].str.startswith("panel_"), "rater_id"].nunique()
        ),
        "in_house_panel": {
            "members": len(roster),
            "pseudonym_prefix": "panel_",
            "demographics": "rater_demographics.parquet (ethnicity, gender, age)",
            "note": (
                "Not present in any earlier release: panel identifiers are "
                "demographic codes, not `rater_` pseudonyms, so the previous "
                "wide-format loader's prefix filter skipped them silently."
            ),
        },
        "excluded_sources": [
            (
                "Every `date` source (public_date/, private_date/, "
                "date_scores_all*.xlsx) -- the date task is a different question "
                "and no longer ships. The raw files remain in the legacy "
                "repository for anyone who wants them."
            ),
            (
                "generic_scores_all.xlsx / generic_scores_all_2022.xlsx -- "
                "99.9% redundant with public_generic/*.xlsx (56,751 of 56,816 "
                "image-rater pairs overlap, 0 raters unique to it)."
            ),
            "public_generic_all.xlsx (near-total subset of public_generic/*.xlsx)",
            "private_generic_all.xlsx (wide merge of private_generic/*.xlsx)",
        ],
    }
    (output_dir / "reconciliation_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
