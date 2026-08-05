"""Compute the canonical `score` label for each split.

`score` is the **rater-normalised mean of every generic rating from a valid
rater**, over images with at least 10 such ratings. Each part of that was
decided rather than inherited:

- **Normalised per rater, then averaged.** Raters use the scale differently:
  two people can rank a set of faces identically while one averages 6.1 and
  the other 4.0. Nobody rated every image, so with a plain mean that habit
  leaks into the image -- a face scored by generous raters comes out higher
  for reasons that have nothing to do with the face. Each rater's scores are
  standardised in their own (shrunken) units before averaging, and the result
  is mapped back onto 1-10. See `legacy/normalization.py` for why the
  shrinkage is not optional here.
- **Not a trimmed or robust mean.** Trimming needs an arbitrary percentage
  and buys nothing once rater scale is handled directly.
- **Validity screening, not agreement screening.** One behavioural rule:
  fewer than 3 distinct scores. Filtering on rater/consensus correlation is
  deliberately *not* applied -- it is circular, and it deletes the minority
  aesthetic variation this dataset exists to study. `score_all_raters` ships
  beside `score` so the effect of every screen stays visible.
- **Rating volume is not a rater rule.** A rater who scored three faces is
  kept; the image-level support rule is what protects the labels.
- **Generic ratings only.** The legacy `date` task asked a different
  question and never enters a label.

Three columns ship beside it, so no consumer is forced to accept this choice:

    score_raw_mean    plain mean over the same valid raters, no normalisation
    score_all_raters  plain mean over every rater, no screening at all
    score             the canonical label

`score_raw_mean` -- not `score` -- is what equals the mean of the shipped
distribution, and that is asserted below. The distributions stay counts of
the integer scores raters actually gave; normalising them would fabricate
ratings on a scale nobody used.

This **replaces** the legacy 2021 label, which could not be recomputed from
any released file because its cleaning pipeline's inputs no longer exist
(Finding 20). The legacy values remain in git history and in the legacy
snapshot for anyone reproducing pre-2026 results.

    uv run python scripts/data/build_labels.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.normalization import (
    DEFAULT_SHRINKAGE,
    normalised_image_scores,
)
from mebeauty_benchmark.legacy.validity import MIN_RATINGS_PER_IMAGE

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=DEFAULT_SHRINKAGE,
        help="Ratings at which a rater is trusted halfway (see normalization.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()

    per_rater = pd.read_parquet(
        v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    )
    valid = per_rater[per_rater["rater_valid"]]
    print(
        f"{len(valid)} of {len(per_rater)} ratings from "
        f"{valid['rater_id'].nunique()} of {per_rater['rater_id'].nunique()} raters"
    )

    # Each rater's scale is estimated from *all* their valid ratings, including
    # those on images that will not be labelled. Their habit is a property of
    # the person, and throwing away evidence about it would only make the
    # estimate noisier.
    scored = normalised_image_scores(
        valid[["image_id", "rater_id", "score"]],
        shrinkage=args.shrinkage,
        min_ratings=MIN_RATINGS_PER_IMAGE,
    ).set_index("image_id")
    print(
        f"Labels for {len(scored)} of {valid['image_id'].nunique()} images "
        f"(>= {MIN_RATINGS_PER_IMAGE} valid ratings)"
    )

    means = scored["score_normalised"]
    raw_means = scored["raw_mean"]
    means_all = per_rater.groupby("image_id")["score"].mean()

    shift = (means - raw_means).abs()
    print(
        f"  normalisation moves a label by {shift.mean():.3f} on average, "
        f"{shift.max():.3f} at most"
    )

    ratings_dir = v3_dir / "ratings" / "aggregate"
    report: dict[str, object] = {
        "method": (
            "per-rater shrinkage-normalised mean over valid raters, "
            f"images with >= {MIN_RATINGS_PER_IMAGE} such ratings"
        ),
        "shrinkage": args.shrinkage,
        "min_ratings_per_image": MIN_RATINGS_PER_IMAGE,
        "rater_screening": "fewer than 3 distinct scores (see legacy/validity.py)",
        "rating_task": "generic",
        "raters_kept": int(valid["rater_id"].nunique()),
        "raters_total": int(per_rater["rater_id"].nunique()),
        "ratings_kept": len(valid),
        "ratings_total": len(per_rater),
        "normalisation_shift": {
            "mean_abs": round(float(shift.mean()), 4),
            "max_abs": round(float(shift.max()), 4),
            "correlation_with_raw_mean": round(float(means.corr(raw_means)), 4),
        },
        "splits": {},
    }
    # Split membership comes from `splits.parquet`, which the release split
    # builder writes. Reading it from the existing aggregate files instead --
    # as this did -- silently preserves whatever membership was there before,
    # so a rebuilt split never actually takes effect.
    assignment = pd.read_parquet(v3_dir / "ratings" / "splits.parquet")
    ratings_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        path = ratings_dir / f"{split}.parquet"
        rows = assignment[assignment["split"] == split][["image_id"]].copy()
        rows = rows.sort_values("image_id").reset_index(drop=True)
        rows["score"] = rows["image_id"].map(means)
        # The unnormalised and unscreened means, so the effect of each step
        # stays visible and the pre-filter label remains recoverable.
        rows["score_raw_mean"] = rows["image_id"].map(raw_means)
        rows["score_all_raters"] = rows["image_id"].map(means_all)

        # An image without enough support cannot be scored, and a silent NaN
        # label would train quietly and wrongly.
        unscored = rows[rows["score"].isna()]
        if not unscored.empty:
            print(f"  {split}: dropping {len(unscored)} rows with too few ratings")
        rows = rows.dropna(subset=["score"]).reset_index(drop=True)

        rows.to_parquet(path, index=False)
        report["splits"][split] = {
            "rows": len(rows),
            "dropped_unscored": len(unscored),
            "mean_score": round(float(rows["score"].mean()), 4),
            "min_score": round(float(rows["score"].min()), 4),
            "max_score": round(float(rows["score"].max()), 4),
        }
        print(
            f"  {split}: {len(rows)} rows, mean {report['splits'][split]['mean_score']}"
        )

    # `score_raw_mean` must equal the mean of the shipped distribution, or the
    # two artifacts disagree about which ratings they were built from. `score`
    # is deliberately *not* checked against it -- normalisation is exactly the
    # difference between them.
    distributions = pd.read_parquet(v3_dir / "ratings" / "distributions.parquet")
    merged = (
        pd.concat([pd.read_parquet(ratings_dir / f"{s}.parquet") for s in SPLITS])
        .merge(distributions[["image_id", "mean"]], on="image_id", how="left")
        .assign(diff=lambda d: (d["score_raw_mean"] - d["mean"]).abs())
    )
    if merged["diff"].isna().any():
        raise AssertionError("a labelled image has no shipped distribution")
    max_diff = float(merged["diff"].max())
    if max_diff > 1e-9:
        raise AssertionError(
            f"score_raw_mean disagrees with the distribution mean by {max_diff}"
        )
    report["consistent_with_distribution"] = True
    print(f"Verified score_raw_mean == distribution mean (max diff {max_diff:.1e})")

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
