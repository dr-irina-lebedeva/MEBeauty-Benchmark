"""Build per-rater ratings for **both** tasks, with a shared panel roster.

`build_ratings_by_rater.py` ingests only the `generic` task, because that is
the only one behind the canonical label. This builds the full per-rater record
the release ships: `generic` *and* `date`, crowd *and* in-house panel, in one
table with a `task` column.

**Why the roster must be shared.** The panel is identified by filename
(`asian_female_17.xlsx`), and 4 of them rated in both tasks. Building each task
separately would give the same person two different `panel_XXXX` ids and make
them look like two raters. The roster is therefore built from the union of both
directories before any pseudonym is assigned.

**Two people can share a demographic triple.** `caucasian_female_34.xlsx` and
`caucasian_female_34_2..xlsx` are different panel members, which is exactly why
the pseudonym comes from `legacy/panel.py`'s canonical id and not from the
demographics. Matching across tasks is by canonical id, so it inherits that
distinction rather than collapsing on age+gender+ethnicity.

**Crowd raters have no demographics.** The MTurk pool supplied none, so they
carry nulls rather than being silently attributed characteristics. The
pseudonym mapping is global across every source file, so `rater_0002` is the
same worker in both tasks.

**A rater may answer both tasks about the same image**, and both answers are
kept -- they are different questions. Only a repeat of the *same* task by the
same rater on the same image is a duplicate, and only those are dropped.

    uv run python scripts/data/build_all_ratings.py \\
        --pseudonymized data/pseudonymized_scores \\
        --legacy-copy data/legacy_snapshot \\
        --v3 data/mebeauty_v3 \\
        --output data/mebeauty_v3/ratings/by_rater \\
        --report-out reports/legacy_audit/all_ratings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.panel import (
    assign_pseudonyms,
    build_roster,
    canonical_panel_id,
)

TASKS = {
    "generic": ("public_generic", "private_generic"),
    "date": ("public_date", "private_date"),
}

#: Panel ages that the release publishes differently from the source filename.
#:
#: Two panel files are named `*_17`, which under the roster's
#: `ethnicity_gender_age` convention reads as a 17-year-old rater. At the
#: maintainer's direction the released age is 18.
#:
#: **This is applied to the published `age` value only, not to the rater's
#: identity.** A literal rename would collide: `asian_female_17` (date task)
#: renamed to `asian_female_18` clashes with an existing, *different*
#: `asian_female_18` in the generic task, and because the roster keys on
#: `ethnicity_gender_age` the two people would be merged into one rater and
#: their ratings pooled. Overriding the value keeps them distinct.
#:
#: The source filenames under `data/legacy_snapshot/` are never modified --
#: the legacy snapshot is read-only by project rule -- so the original values
#: remain recoverable in the private archive.
PUBLISHED_AGE_OVERRIDES = {
    "caucasian_male_17": 18,
    "asian_female_17": 18,
}

#: Wide merges of the per-rater panel files. Reading them alongside the
#: individual files would double-count every rating they contain.
PANEL_MERGE_FILES = {
    "private_date_female.xlsx",
    "private_date_male.xlsx",
    "private_generic_all.xlsx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudonymized", required=True)
    parser.add_argument("--legacy-copy", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--consolidation-report",
        required=True,
        help="reports/legacy_audit/consolidation.json -- the 25 duplicate "
        "photographs merged after the original ingest. Ratings must be moved "
        "onto the surviving image_id or they end up split across two copies "
        "of the same photo.",
    )
    parser.add_argument("--report-out", required=True)
    return parser.parse_args()


def merge_map(report_path: Path) -> dict[str, str]:
    """merged-away image_id -> surviving image_id."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = {}
    for merge in report["merges"]:
        for gone in merge["merged_away"]:
            mapping[gone] = merge["kept"]
    return mapping


def merged_basenames(report_path: Path) -> dict[str, str]:
    """Basename of a merged-away photograph -> surviving image_id.

    The merged-away files were deleted, so their basenames are absent from
    `metadata.parquet` and every rating naming them would be dropped as
    unresolvable -- 24 of the 25 merges, several thousand ratings. The
    consolidation report is the only remaining record of which survivor they
    belong to, so it is consulted directly rather than inferred.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = {}
    for merge in report["merges"]:
        for path in merge["merged_paths"]:
            mapping[path.rsplit("/", 1)[-1].lower()] = merge["kept"]
    return mapping


def build_basename_lookup(metadata: pd.DataFrame) -> dict[str, str | None]:
    """Legacy basename -> image_id, or None where the name is ambiguous."""
    from collections import defaultdict

    candidates: dict[str, set[str]] = defaultdict(set)
    for row in metadata.itertuples():
        candidates[row.legacy_filename.lower()].add(row.image_id)
        # `other_legacy_paths` is a *string*, not a list. Splatting it with `*`
        # unpacks it character by character, filling the lookup with 41
        # single-letter keys and silently losing every alternate filename --
        # which cost two images their entire rating history, because the
        # rating files name them by a clean spelling while `legacy_filename`
        # carries a mojibake one.
        alternates = row.other_legacy_paths
        if isinstance(alternates, str):
            alternates = [alternates] if alternates else []
        elif alternates is None:
            alternates = []
        for path in (row.legacy_path, *alternates):
            if path:
                candidates[path.rsplit("/", 1)[-1].lower()].add(row.image_id)
    return {
        basename: (next(iter(ids)) if len(ids) == 1 else None)
        for basename, ids in candidates.items()
    }


def resolve_stems(images: pd.Series, lookup: dict[str, str | None]) -> pd.Series:
    """Panel files store `girl-1763686_1920`; the lookup wants a basename."""
    known = set(lookup)

    def resolve(name: str) -> str:
        if name in known:
            return name
        for extension in (".jpg", ".jpeg", ".png"):
            candidate = f"{name}{extension}"
            if candidate in known:
                return candidate
        return name

    return images.map(resolve)


def load_crowd(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.xlsx")):
        frame = pd.read_excel(path)[["image", "rater", "label"]].rename(
            columns={"label": "score"}
        )
        frame["image"] = frame["image"].astype(str).str.rsplit("/", n=1).str[-1]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["image", "rater", "score"])
    out = pd.concat(frames, ignore_index=True)
    out["pool"] = "crowd"
    return out


def load_panel(directory: Path, roster, pseudonyms) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.xlsx")):
        if path.name in PANEL_MERGE_FILES:
            continue
        canonical = canonical_panel_id(path.name, roster)
        if canonical is None:
            print(f"    skipping unrecognised panel file {path.name}")
            continue
        frame = pd.read_excel(path)
        if not {"image", "label"} <= set(frame.columns):
            raise ValueError(f"{path.name}: expected image/label columns")
        frame = frame[["image", "label"]].rename(columns={"label": "score"}).dropna()
        frame["rater"] = pseudonyms[canonical]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["image", "rater", "score"])
    out = pd.concat(frames, ignore_index=True)
    out["pool"] = "panel"
    return out


def main() -> None:
    args = parse_args()
    pseudonymized = Path(args.pseudonymized).expanduser().resolve()
    legacy = Path(args.legacy_copy).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(Path(args.v3) / "images" / "metadata.parquet")
    lookup = build_basename_lookup(metadata)

    consolidation = Path(args.consolidation_report).expanduser().resolve()
    # Merged-away photographs no longer exist on disk, so their basenames must
    # be pointed at the survivor explicitly or their ratings are lost.
    recovered = 0
    for basename, survivor in merged_basenames(consolidation).items():
        if lookup.get(basename) != survivor:
            lookup[basename] = survivor
            recovered += 1
    print(f"Merged-away basenames redirected to their survivor: {recovered}")

    # One roster across both tasks, so a panel member who rated in both keeps
    # a single identity.
    panel_names: list[str] = []
    for _, panel_dir in TASKS.values():
        panel_names += [p.name for p in (legacy / "scores" / panel_dir).glob("*.xlsx")]
    roster = build_roster([n for n in panel_names if n not in PANEL_MERGE_FILES])
    pseudonyms = assign_pseudonyms(roster)
    print(f"Panel roster across both tasks: {len(roster)} members")

    per_task = []
    report: dict[str, object] = {"tasks": {}}
    for task, (crowd_dir, panel_dir) in TASKS.items():
        print(f"\n{task}:")
        crowd = load_crowd(pseudonymized / crowd_dir)
        panel = load_panel(legacy / "scores" / panel_dir, roster, pseudonyms)
        combined = pd.concat([crowd, panel], ignore_index=True)
        combined["image"] = combined["image"].astype(str).str.lower()
        combined["image"] = resolve_stems(combined["image"], lookup)
        combined["image_id"] = combined["image"].map(lookup)

        unresolved = int(combined["image_id"].isna().sum())
        resolved = combined.dropna(subset=["image_id"])
        # Within a task, the same rater answering the same image twice is a
        # genuine duplicate. Across tasks it is two different questions.
        deduped = resolved.drop_duplicates(subset=["image_id", "rater"], keep="first")
        dropped = len(resolved) - len(deduped)

        frame = deduped[["image_id", "rater", "score", "pool"]].rename(
            columns={"rater": "rater_id"}
        )
        frame["task"] = task
        per_task.append(frame)
        print(
            f"  crowd {len(crowd)} + panel {len(panel)} -> {len(frame)} ratings, "
            f"{frame['rater_id'].nunique()} raters, "
            f"{frame['image_id'].nunique()} images"
        )
        report["tasks"][task] = {
            "crowd_rows": len(crowd),
            "panel_rows": len(panel),
            "unresolved_image_dropped": unresolved,
            "duplicate_rows_dropped": int(dropped),
            "final_rows": len(frame),
            "raters": int(frame["rater_id"].nunique()),
            "images": int(frame["image_id"].nunique()),
        }

    ratings = pd.concat(per_task, ignore_index=True)

    # Move ratings off merged-away duplicates onto the surviving photograph.
    # A rater who scored both copies leaves two judgements of the same face;
    # both are kept, because that is a genuine test-retest signal and
    # averaging them would invent a score nobody gave.
    keep_of = merge_map(consolidation)
    before = ratings["image_id"].nunique()
    ratings["image_id"] = ratings["image_id"].map(lambda i: keep_of.get(i, i))
    print(
        f"\nMerged duplicate photographs: {before} -> "
        f"{ratings['image_id'].nunique()} images ({len(keep_of)} merged away)"
    )

    # Only images that survived into the released set.
    known = set(metadata["image_id"])
    unknown = int((~ratings["image_id"].isin(known)).sum())
    if unknown:
        print(f"  dropping {unknown} ratings for images not in the release")
        ratings = ratings[ratings["image_id"].isin(known)]

    ratings = ratings.sort_values(["task", "image_id", "rater_id"]).reset_index(
        drop=True
    )
    ratings["score"] = ratings["score"].astype("float64")

    # Raters table: demographics for the panel, nulls for the crowd. Kept
    # apart from the ratings so the identifier stays opaque and a consumer
    # cannot accidentally join demographics onto a rating they then publish.
    demographics = {pseudonyms[canonical]: rater for canonical, rater in roster.items()}

    def published_age(rater) -> int | None:
        if rater is None:
            return None
        return PUBLISHED_AGE_OVERRIDES.get(rater.canonical_id, rater.age)

    overridden = sum(
        1 for r in demographics.values() if r.canonical_id in PUBLISHED_AGE_OVERRIDES
    )
    if overridden:
        print(f"Published-age overrides applied: {overridden}")

    raters = (
        pd.DataFrame(
            [
                {
                    "rater_id": rater_id,
                    "pool": "panel" if rater_id.startswith("panel_") else "crowd",
                    "gender": getattr(demographics.get(rater_id), "gender", None),
                    "ethnicity": getattr(demographics.get(rater_id), "ethnicity", None),
                    "age": published_age(demographics.get(rater_id)),
                }
                for rater_id in sorted(ratings["rater_id"].unique())
            ]
        )
        .astype({"age": "Int64"})
        .sort_values("rater_id")
        .reset_index(drop=True)
    )

    ratings.to_parquet(output / "ratings_all_tasks.parquet", index=False)
    raters.to_parquet(output / "raters.parquet", index=False)

    # The generic-only view the label pipeline consumes, derived from the same
    # table so the two can never disagree about a rater id. Rebuilding it here
    # matters: the panel roster now spans both tasks, so `panel_0001` is a
    # different person than it was under the generic-only roster.
    from mebeauty_benchmark.legacy.validity import attach_validity

    generic = ratings[ratings["task"] == "generic"][
        ["image_id", "rater_id", "score"]
    ].reset_index(drop=True)
    generic = attach_validity(generic)
    generic.to_parquet(output / "ratings_by_rater.parquet", index=False)
    print(
        f"Generic view for the label pipeline: {len(generic)} ratings, "
        f"{int(generic['rater_valid'].sum())} from valid raters"
    )
    print(f"\nWrote {len(ratings)} ratings over {len(raters)} raters to {output}")

    both = ratings.groupby("rater_id")["task"].nunique().pipe(lambda s: (s > 1).sum())
    report["totals"] = {
        "ratings": len(ratings),
        "raters": len(raters),
        "raters_in_both_tasks": int(both),
        "panel_raters": int((raters["pool"] == "panel").sum()),
        "crowd_raters": int((raters["pool"] == "crowd").sum()),
        "raters_with_demographics": int(raters["age"].notna().sum()),
        "images": int(ratings["image_id"].nunique()),
    }
    print(json.dumps(report["totals"], indent=2))

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
