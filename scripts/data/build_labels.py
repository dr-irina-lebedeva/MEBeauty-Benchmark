"""Compute the canonical `score` label for each split, SCUT-FBP5500 style.

`score` is the **plain unweighted mean of every generic rating from a valid
rater**, with no weighting. Validity is decided from a rater's own behaviour
alone -- see `legacy/validity.py` -- and never from whether they agreed with
anyone. Each part of that was tested rather than assumed:

- **Plain mean, not a model.** Model-based aggregation (Dawid-Skene and
  relatives) outperforms a mean mainly when spam dominates the pool; it also
  bakes a modelling assumption into what is supposed to be data, and produces
  a number no one else can reproduce without rerunning the model. SCUT's
  `All_labels.txt` is a plain mean of its released per-rater ratings, and that
  is the convention this dataset follows.
- **Not a trimmed or robust mean.** Trimming needs an arbitrary percentage,
  and it breaks the property that `score` equals the mean of the shipped
  distribution.
- **Validity screening, not agreement screening.** Two rules apply, both
  behavioural: fewer than 10 ratings, or 2 or fewer distinct scores across
  10+ ratings. Together they remove 0.9% of ratings and no image loses its
  label. Filtering on rater/consensus correlation is deliberately *not*
  applied: it is circular, it is the most expensive rule measured (1.4%),
  and it deletes the minority aesthetic variation this dataset exists to
  study. `score_all_raters` ships beside `score` so the effect is visible.
- **Generic ratings only.** The legacy `date` task asked a different
  question and never enters a label.

The result is reproducible in one line from the ratings this dataset ships:

    ratings[ratings.rater_valid].groupby("image_id")["score"].mean()

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

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()

    per_rater = pd.read_parquet(
        v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    )
    valid = per_rater[per_rater["rater_valid"]]
    means = valid.groupby("image_id")["score"].mean()
    means_all = per_rater.groupby("image_id")["score"].mean()
    print(f"Labels from {len(per_rater)} ratings over {len(means)} images")

    ratings_dir = v3_dir / "ratings" / "aggregate"
    report: dict[str, object] = {
        "method": "unweighted mean over valid raters (behavioural screening only)",
        "reproduce": 'ratings_by_rater.groupby("image_id")["score"].mean()',
        "rating_task": "generic",
        "splits": {},
    }
    for split in SPLITS:
        path = ratings_dir / f"{split}.parquet"
        rows = pd.read_parquet(path)[["image_id"]].copy()
        rows["score"] = rows["image_id"].map(means)
        # The unfiltered mean, so the effect of the validity filter stays
        # visible and the pre-filter label remains recoverable.
        rows["score_all_raters"] = rows["image_id"].map(means_all)

        # An image in a split with no surviving rating cannot be scored, and a
        # silent NaN label would train quietly and wrongly.
        unscored = rows[rows["score"].isna()]
        if not unscored.empty:
            print(f"  {split}: dropping {len(unscored)} rows with no generic rating")
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

    # The label must equal the mean of the shipped distribution, or the two
    # artifacts disagree about the same ratings.
    distributions = pd.read_parquet(v3_dir / "ratings" / "distributions.parquet")
    merged = (
        pd.concat([pd.read_parquet(ratings_dir / f"{s}.parquet") for s in SPLITS])
        .merge(distributions[["image_id", "mean"]], on="image_id", how="left")
        .assign(diff=lambda d: (d["score"] - d["mean"]).abs())
    )
    max_diff = float(merged["diff"].max())
    if max_diff > 1e-9:
        raise AssertionError(
            f"score disagrees with the shipped distribution mean by {max_diff}"
        )
    report["consistent_with_distribution"] = True
    print(f"Verified score == distribution mean (max diff {max_diff:.1e})")

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
