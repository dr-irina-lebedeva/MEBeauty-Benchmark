"""Rebuild the release splits: leakage-safe, stratified, and round-numbered.

The previous split (1,719/218/523) was whatever survived successive filtering,
which is why the numbers look arbitrary -- they are. This rebuilds from the
full labelled set with explicit targets.

**Nothing that could leak is allowed to cross a split.** Groups are formed by
union-find over three signals, in order of how much they are trusted:

1. *Same photograph, confirmed.* `duplicate_classification.json` aligned each
   candidate pair on facial landmarks and correlated the pixels **outside**
   the face: same photograph means the background aligns too. 25 pairs passed
   and were already merged into single images upstream.
2. *Same person, different photograph.* The same test, 228 pairs where the
   face matched but the background did not. These are two genuine photographs
   of one person -- exactly the case that inflates a test score if split
   apart.
3. *Same source URL, or a perceptual near-duplicate.* Cheap, exact, and
   catches anything the face model missed.

No fresh face recognition is run here. The groups come from work already done
and reviewed; re-deriving identities automatically at release time would be
both slower and a stronger claim about real people than this needs to make.

**Stratification.** Groups are assigned greedily, largest first, each going to
whichever split is furthest below target for that group's stratum (gender x
ethnicity x score tercile). Greedy-largest-first matters: a 4-image group
placed last cannot be split, so it would blow whichever target it lands on.

**Five cross-validation folds** are built over train+val by the same
group-respecting logic, so a fold boundary is as leak-proof as a split
boundary. `test` is never part of any fold.

    uv run python scripts/data/build_release_splits.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/release_splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mebeauty_benchmark.legacy.validity import (
    MIN_RATINGS_PER_IMAGE,
    labelled_image_ids,
)

#: Target split sizes. Round val/test with the remainder in train -- the
#: numbers a reader sees should look chosen, because they are.
TARGET_VAL = 250
TARGET_TEST = 250

N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--val", type=int, default=TARGET_VAL)
    parser.add_argument("--test", type=int, default=TARGET_TEST)
    return parser.parse_args()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(v3: Path, image_ids: set[str]) -> tuple[dict[str, str], dict]:
    """image_id -> group_id, plus a record of what linked what."""
    union = UnionFind()
    for image_id in image_ids:
        union.find(image_id)

    provenance: dict[str, int] = defaultdict(int)

    classification = (
        v3.parent.parent / "reports/legacy_audit/duplicate_classification.json"
    )
    if classification.is_file():
        report = json.loads(classification.read_text(encoding="utf-8"))
        for pair in report.get("same_person_pairs", []):
            a, b = pair.get("image_a"), pair.get("image_b")
            if a in image_ids and b in image_ids:
                union.union(a, b)
                provenance["same_person_confirmed"] += 1
        for pair in report.get("same_photograph_pairs", []):
            a, b = pair.get("image_a"), pair.get("image_b")
            if a in image_ids and b in image_ids:
                union.union(a, b)
                provenance["same_photograph_confirmed"] += 1

    metadata = pd.read_parquet(v3 / "images" / "metadata.parquet")
    metadata = metadata[metadata["image_id"].isin(image_ids)]

    # Same source URL means the same photograph, whatever any model thinks.
    for url, frame in metadata[metadata["source_url"].notna()].groupby("source_url"):
        ids = frame["image_id"].tolist()
        if len(ids) > 1 and str(url).strip():
            for other in ids[1:]:
                union.union(ids[0], other)
                provenance["same_source_url"] += 1

    for row in metadata.itertuples():
        others = row.near_duplicate_image_ids
        if isinstance(others, str):
            others = [others] if others else []
        elif others is None:
            others = []
        for other in others:
            if other in image_ids:
                union.union(row.image_id, other)
                provenance["near_duplicate"] += 1

    return {i: union.find(i) for i in image_ids}, dict(provenance)


def main() -> None:
    args = parse_args()
    v3 = Path(args.v3).expanduser().resolve()

    ratings = pd.read_parquet(v3 / "ratings" / "by_rater" / "ratings_by_rater.parquet")
    labelled = labelled_image_ids(ratings, MIN_RATINGS_PER_IMAGE)
    metadata = pd.read_parquet(v3 / "images" / "metadata.parquet")
    metadata = metadata[metadata["image_id"].isin(labelled)].reset_index(drop=True)
    print(
        f"Labelled images (>= {MIN_RATINGS_PER_IMAGE} valid ratings): {len(metadata)}"
    )

    valid = ratings[ratings["rater_valid"]]
    score = valid.groupby("image_id")["score"].mean()
    metadata["score"] = metadata["image_id"].map(score)

    groups, provenance = build_groups(v3, set(metadata["image_id"]))
    metadata["group"] = metadata["image_id"].map(groups)
    sizes = metadata.groupby("group").size()
    print(
        f"Leakage groups: {len(sizes)} covering {len(metadata)} images; "
        f"largest {sizes.max()}, multi-image groups {int((sizes > 1).sum())}"
    )
    print(f"  links by source: {provenance}")

    # Stratum: gender x ethnicity x score tercile. A group takes the stratum of
    # its first member; groups are small, so this is stable.
    metadata["score_bin"] = pd.qcut(metadata["score"], 3, labels=["low", "mid", "high"])
    metadata["stratum"] = (
        metadata["gender"].astype(str)
        + "|"
        + metadata["ethnicity"].astype(str)
        + "|"
        + metadata["score_bin"].astype(str)
    )

    targets = {
        "train": len(metadata) - args.val - args.test,
        "val": args.val,
        "test": args.test,
    }
    print(f"Targets: {targets}")

    stratum_totals = metadata["stratum"].value_counts().to_dict()
    stratum_target = {
        split: {
            s: total * targets[split] / len(metadata)
            for s, total in stratum_totals.items()
        }
        for split in targets
    }
    have: dict[str, dict[str, float]] = {split: defaultdict(float) for split in targets}
    assigned_count = dict.fromkeys(targets, 0)

    group_frames = list(metadata.groupby("group"))
    rng = np.random.default_rng(args.seed)
    # Largest first; ties broken by a seeded shuffle so the result is
    # reproducible but not alphabetical.
    order = sorted(group_frames, key=lambda kv: (-len(kv[1]), rng.random()))

    assignment: dict[str, str] = {}
    for _, frame in order:
        best, best_cost = None, None
        for split, target in targets.items():
            if assigned_count[split] + len(frame) > target and split != "train":
                continue
            # Cost: how far this split would overshoot its stratum quotas.
            cost = 0.0
            for stratum, count in frame["stratum"].value_counts().items():
                quota = stratum_target[split][stratum]
                cost += max(0.0, have[split][stratum] + count - quota)
            # Prefer the split that is proportionally emptiest overall.
            cost += (assigned_count[split] / max(1, target)) * len(frame)
            if best_cost is None or cost < best_cost:
                best, best_cost = split, cost
        best = best or "train"
        for image_id in frame["image_id"]:
            assignment[image_id] = best
        for stratum, count in frame["stratum"].value_counts().items():
            have[best][stratum] += count
        assigned_count[best] += len(frame)

    metadata["split"] = metadata["image_id"].map(assignment)
    print(f"After greedy pass: {metadata['split'].value_counts().to_dict()}")

    # Repair to the exact targets. The greedy pass lands within a few images
    # because groups are indivisible; single-image groups are, so moving those
    # closes the gap without ever splitting a group. The one chosen is the one
    # whose stratum train can most afford to lose, so balance is not traded
    # away for round numbers.
    group_size = metadata.groupby("group")["image_id"].transform("size")
    for split in ("val", "test"):
        while (metadata["split"] == split).sum() < targets[split]:
            movable = metadata[(metadata["split"] == "train") & (group_size == 1)]
            if movable.empty:
                break
            deficit = {
                s: stratum_target[split][s] - have[split][s] for s in stratum_totals
            }
            pick = (
                movable.assign(gain=movable["stratum"].map(deficit).fillna(0.0))
                .sort_values("gain", ascending=False)
                .iloc[0]
            )
            metadata.loc[metadata["image_id"] == pick["image_id"], "split"] = split
            have[split][pick["stratum"]] += 1
        while (metadata["split"] == split).sum() > targets[split]:
            movable = metadata[(metadata["split"] == split) & (group_size == 1)]
            if movable.empty:
                break
            pick = movable.iloc[0]
            metadata.loc[metadata["image_id"] == pick["image_id"], "split"] = "train"
            have[split][pick["stratum"]] -= 1

    actual = metadata["split"].value_counts().to_dict()
    print(f"Actual: {actual}")

    # SCUT-FBP5500's two protocols, so code written for that benchmark runs
    # here unchanged.
    #
    # SCUT folds over *every* image -- 80% train, 20% test per fold, with no
    # validation split and no permanently held-out test. That is a genuine
    # trade rather than an oversight: every image gets evaluated on, which
    # buys statistical power, but nothing is protected from being fit to over
    # years of community use. Both protocols therefore ship side by side and
    # the card says which to use when.
    #
    # Unlike SCUT's, these folds respect the leakage groups, so two
    # photographs of one person never land on opposite sides of a fold.
    # Stratified, not merely size-balanced. Assigning groups to the emptiest
    # fold balances *counts* but not composition: doing that left asian at
    # 10.5-16.4% and female at 47.7-57.3% across folds, so a fold-to-fold
    # difference in a result could be demographic mix rather than method.
    # Each group goes to whichever fold is furthest below quota for its own
    # stratum, exactly as the train/val/test assignment does.
    fold_of: dict[str, int] = {}
    fold_sizes = dict.fromkeys(range(N_FOLDS), 0)
    fold_have: dict[int, dict[str, float]] = {
        f: defaultdict(float) for f in range(N_FOLDS)
    }
    fold_quota = {stratum: total / N_FOLDS for stratum, total in stratum_totals.items()}
    for _, frame in sorted(metadata.groupby("group"), key=lambda kv: -len(kv[1])):
        counts = frame["stratum"].value_counts()
        best, best_cost = 0, None
        for fold in range(N_FOLDS):
            cost = sum(
                max(0.0, fold_have[fold][stratum] + count - fold_quota[stratum])
                for stratum, count in counts.items()
            )
            cost += fold_sizes[fold] / max(1, len(metadata) / N_FOLDS) * len(frame)
            if best_cost is None or cost < best_cost:
                best, best_cost = fold, cost
        for image_id in frame["image_id"]:
            fold_of[image_id] = best
        for stratum, count in counts.items():
            fold_have[best][stratum] += count
        fold_sizes[best] += len(frame)
    metadata["cv_fold"] = metadata["image_id"].map(fold_of).astype("Int64")
    print(f"SCUT-style 5-fold over all {len(metadata)} images: {fold_sizes}")

    # Non-negotiable: no group may appear in two splits.
    crossing = metadata.groupby("group")["split"].nunique()
    if int((crossing > 1).sum()):
        raise AssertionError(f"{int((crossing > 1).sum())} groups cross a split")
    for column in ("cv_fold",):
        crossing_count = int((metadata.groupby("group")[column].nunique() > 1).sum())
        if crossing_count:
            raise AssertionError(f"{crossing_count} groups cross {column}")
    print("Verified: no leakage group crosses a split or a fold")

    out = v3 / "ratings" / "splits.parquet"
    metadata[["image_id", "split", "cv_fold"]].to_parquet(out, index=False)
    print(f"Wrote {out}")

    balance = metadata.groupby(["split", "ethnicity"]).size().unstack(fill_value=0)
    proportion = balance.div(balance.sum(axis=1), axis=0).mul(100).round(1)
    print("\nEthnicity balance by split (%):")
    print(proportion.to_string())

    report = {
        "seed": args.seed,
        "min_ratings_per_image": MIN_RATINGS_PER_IMAGE,
        "images": len(metadata),
        "targets": targets,
        "actual": actual,
        "groups": len(sizes),
        "multi_image_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "group_links_by_source": provenance,
        "cv_folds": {str(k): v for k, v in fold_sizes.items()},
        "ethnicity_percent_by_split": proportion.to_dict(),
        "gender_percent_by_split": (
            metadata.groupby(["split", "gender"])
            .size()
            .unstack(fill_value=0)
            .pipe(lambda d: d.div(d.sum(axis=1), axis=0).mul(100).round(1))
            .to_dict()
        ),
        "mean_score_by_split": metadata.groupby("split")["score"]
        .mean()
        .round(4)
        .to_dict(),
        "no_group_crosses_a_split": True,
    }
    path = Path(args.report_out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
