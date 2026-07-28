"""Merge duplicate photographs and keep each person inside one split.

Two fixes, from `classify_duplicate_pairs.py`, applied in one pass because
both change which `image_id` a rating belongs to.

**1. Duplicate photographs are merged, not deleted.** The same photo saved
twice under different names carries two independent sets of ratings. Deleting
a copy would discard real annotation work, so instead one copy is kept and
the other's ratings move onto it. The merged image ends up with roughly twice
the ratings, which makes its label *better* estimated, and the duplicate can
no longer sit on both sides of a split.

Where one rater scored both copies, **both scores are kept**: they saw the
photograph twice under different filenames and answered twice, so these are
two genuine judgements, not a duplicated row. Averaging them was tried and is
wrong -- it invents a score off the 1-10 integer scale and destroys the
test-retest signal.

The surviving copy is the one whose source image has the most pixels, so the
merge never discards resolution; ties break on `image_id` so the choice is
deterministic across runs.

**2. Every photo of one person goes into one split.** Different photographs
of the same face are legitimate, separate observations and are all kept --
but split across train and test they let a model score by recognising the
person instead of judging the face. Each identity group is assigned whole to
the split where most of its ratings already are, so the reassignment moves as
few rows as possible.

    uv run python scripts/data/consolidate_identities.py \\
        --v3 data/mebeauty_v3 \\
        --classification reports/legacy_audit/duplicate_classification.json \\
        --report-out reports/legacy_audit/consolidation.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")

#: Face similarity at or above which two *different* photographs are treated
#: as the same person. Validated visually: below ~0.7 the model starts
#: grouping people who merely look alike (blondes with similar styling).
SAME_PERSON_SIMILARITY = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument(
        "--classification", required=True, help="duplicate_classification.json"
    )
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=SAME_PERSON_SIMILARITY,
        help=f"Face similarity for same-person grouping (default {SAME_PERSON_SIMILARITY})",
    )
    return parser.parse_args()


def connected_groups(pairs: list[tuple[str, str]]) -> list[set[str]]:
    """Collapse pairwise links into groups (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict[str, set[str]] = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    return [g for g in groups.values() if len(g) > 1]


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    classification = json.loads(Path(args.classification).read_text())

    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    by_rater_path = v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    ratings = pd.read_parquet(by_rater_path)

    pixels = dict(zip(metadata["image_id"], metadata["width"] * metadata["height"]))
    path_by_id = dict(zip(metadata["image_id"], metadata["legacy_path"]))

    # --- 1. merge duplicate photographs -------------------------------------
    duplicate_groups = connected_groups(
        [(p["image_a"], p["image_b"]) for p in classification["same_photograph_pairs"]]
    )

    # Keep the largest source image; `-image_id` only to break exact ties
    # deterministically rather than by dict ordering.
    keep_of: dict[str, str] = {}
    merge_records = []
    for group in duplicate_groups:
        survivor = max(sorted(group), key=lambda i: pixels.get(i, 0))
        for member in group:
            keep_of[member] = survivor
        merge_records.append(
            {
                "kept": survivor,
                "kept_path": path_by_id.get(survivor),
                "merged_away": sorted(m for m in group if m != survivor),
                "merged_paths": [
                    path_by_id.get(m) for m in sorted(group) if m != survivor
                ],
            }
        )

    before = len(ratings)
    ratings["image_id"] = ratings["image_id"].map(lambda i: keep_of.get(i, i))

    # A rater who scored both copies gave two genuine, independent judgements
    # of the same photograph -- they saw it twice, under different filenames,
    # and answered twice. Both rows are kept.
    #
    # Averaging them was tried and is wrong twice over: it invents a score
    # (3 and 4 average to 3.5) that no one gave and that does not exist on the
    # 1-10 integer scale the distributions are built over, and it discards the
    # test-retest signal that makes these pairs interesting. The cost is that
    # such a rater contributes twice to that one image's label -- 246 rows out
    # of ~69,000, and each is a real observation.
    repeat_judgements = int(
        ratings.duplicated(subset=["image_id", "rater_id"], keep=False).sum()
    )
    ratings = ratings.sort_values(["image_id", "rater_id"]).reset_index(drop=True)
    ratings.to_parquet(by_rater_path, index=False)

    removed = set(keep_of) - set(keep_of.values())
    extension_of = dict(zip(metadata["image_id"], metadata["extension"]))
    metadata = metadata[~metadata["image_id"].isin(removed)].reset_index(drop=True)
    metadata.to_parquet(v3_dir / "images" / "metadata.parquet", index=False)

    # Delete the merged-away image files too. Dropping only the metadata row
    # would leave the file loadable through ImageFolder, so the duplicate
    # would keep shipping -- the same failure the v3 builder's orphan sweep
    # exists to prevent.
    for image_id in sorted(removed):
        for directory, suffix in (
            (v3_dir / "images", extension_of.get(image_id, ".jpg")),
            (v3_dir / "cropped_256" / "images", ".jpg"),
            (v3_dir / "standardized_256" / "images", ".jpg"),
        ):
            path = directory / f"{image_id}{suffix}"
            if path.exists():
                path.unlink()

    landmarks_path = v3_dir / "landmarks.parquet"
    landmarks = pd.read_parquet(landmarks_path)
    landmarks[~landmarks["image_id"].isin(removed)].to_parquet(
        landmarks_path, index=False
    )

    # --- 2. keep each person inside one split -------------------------------
    identity_groups = connected_groups(
        [
            (
                keep_of.get(p["image_a"], p["image_a"]),
                keep_of.get(p["image_b"], p["image_b"]),
            )
            for p in classification["same_person_pairs"]
            if p["face_similarity"] >= args.identity_threshold
        ]
    )

    split_frames = {
        split: pd.read_parquet(v3_dir / "ratings" / "aggregate" / f"{split}.parquet")
        for split in SPLITS
    }
    # Map split membership through the merge before anything else. A merged-away
    # copy that was in a split hands its place to the survivor along with its
    # ratings; without this the survivor inherits the ratings but no split, and
    # those ratings silently stop counting toward the benchmark.
    split_of: dict[str, str] = {}
    for split, frame in split_frames.items():
        for image_id in frame["image_id"]:
            split_of[keep_of.get(image_id, image_id)] = split
    rating_counts = ratings.groupby("image_id").size()

    moves = []
    for group in identity_groups:
        members = {m for m in group if m in split_of}
        if len(members) < 2:
            continue
        present = {split_of[m] for m in members}
        if len(present) < 2:
            continue
        # Send the whole group where most of its *ratings* already are, so the
        # reassignment disturbs the least data.
        weight: dict[str, int] = defaultdict(int)
        for member in members:
            weight[split_of[member]] += int(rating_counts.get(member, 0))
        target = max(sorted(weight), key=lambda s: weight[s])
        for member in members:
            if split_of[member] != target:
                moves.append(
                    {
                        "image_id": member,
                        "legacy_path": path_by_id.get(member),
                        "from": split_of[member],
                        "to": target,
                    }
                )
                split_of[member] = target

    rebuilt = {
        split: sorted(i for i, s in split_of.items() if s == split) for split in SPLITS
    }
    for split in SPLITS:
        pd.DataFrame({"image_id": rebuilt[split]}).to_parquet(
            v3_dir / "ratings" / "aggregate" / f"{split}.parquet", index=False
        )

    report = {
        "duplicate_photograph_groups": len(duplicate_groups),
        "images_merged_away": len(removed),
        "ratings_before_merge": before,
        "ratings_after_merge": len(ratings),
        "repeat_judgements_kept": repeat_judgements,
        "identity_threshold": args.identity_threshold,
        "identity_groups_spanning_splits": len({m["image_id"] for m in moves}),
        "images_moved_between_splits": len(moves),
        "split_sizes": {s: len(rebuilt[s]) for s in SPLITS},
        "merges": merge_records,
        "moves": moves,
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"merged {len(duplicate_groups)} duplicate groups, {len(removed)} images away"
    )
    print(
        f"  ratings {before} -> {len(ratings)} ({repeat_judgements} repeat judgements kept)"
    )
    print(f"moved {len(moves)} images so each person sits in one split")
    print(f"  split sizes: {report['split_sizes']}")
    print(f"Report: {report_path}")
    print("\nNow re-run: build_rating_distributions.py, build_labels.py,")
    print("            standardize_images.py, build_face_crops.py")


if __name__ == "__main__":
    main()
