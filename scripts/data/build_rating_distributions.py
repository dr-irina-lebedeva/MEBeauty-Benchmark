"""Build per-image rating distributions (soft labels) for the v3 dataset.

Writes `ratings/distributions.parquet`: one row per (image_id, rating_type)
with raw `counts` and normalized `probabilities` over the 1-10 scale, plus
mean/median/std/entropy. This is what makes MEBeauty usable for label
distribution learning, the dominant paradigm in facial beauty prediction --
the canonical single `score` alone cannot support it.

The distributions are built from `ratings/by_rater/ratings_by_rater.parquet`
and are **unfiltered**: every rater contributes, matching Finding 14's policy
and deliberately unlike the canonical `score`, which inherits the 2021
pipeline's consensus-based rater filtering (Finding 20). The report records
how far apart the two end up.

    uv run python scripts/data/build_rating_distributions.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/rating_distributions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.distributions import (
    MAX_ENTROPY_BITS,
    SCORE_BINS,
    build_distributions,
)

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
    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    print(f"Source: {len(per_rater)} per-rater ratings, {len(metadata)} images")

    distributions = build_distributions(per_rater)

    # Every distribution must be a real probability distribution, and must
    # account for exactly the ratings it was built from. Checked here rather
    # than trusted, because a silently malformed soft label would train
    # quietly and wrongly.
    probability_sums = distributions["probabilities"].apply(sum)
    if not probability_sums.sub(1.0).abs().lt(1e-9).all():
        raise AssertionError("some probability vectors do not sum to 1")
    if not distributions["counts"].apply(sum).equals(distributions["n_ratings"]):
        raise AssertionError("counts do not sum to n_ratings")
    if not distributions["counts"].apply(len).eq(len(SCORE_BINS)).all():
        raise AssertionError(f"expected {len(SCORE_BINS)} bins per distribution")

    output_path = v3_dir / "ratings" / "distributions.parquet"
    distributions.to_parquet(output_path, index=False)
    print(f"Wrote {len(distributions)} distributions to {output_path}")

    report: dict[str, object] = {
        "score_bins": list(SCORE_BINS),
        "max_entropy_bits": round(MAX_ENTROPY_BITS, 5),
        "rows": len(distributions),
        "unfiltered": True,
        "note": (
            "Built from all raters, none excluded. The canonical `score` "
            "inherits the 2021 pipeline's consensus-based rater filtering "
            "(Finding 20), so these means differ from it by design."
        ),
        "by_rating_type": {},
    }
    for rating_type, group in distributions.groupby("rating_type"):
        report["by_rating_type"][rating_type] = {
            "images": int(group["image_id"].nunique()),
            "ratings": int(group["n_ratings"].sum()),
            "min_ratings_per_image": int(group["n_ratings"].min()),
            "median_ratings_per_image": float(group["n_ratings"].median()),
            "max_ratings_per_image": int(group["n_ratings"].max()),
            "mean_entropy_bits": round(float(group["entropy_bits"].mean()), 5),
            "images_with_single_rating": int((group["n_ratings"] == 1).sum()),
        }

    # How far the unfiltered soft label sits from the canonical hard label.
    canonical = pd.concat(
        [
            pd.read_parquet(v3_dir / "ratings" / "aggregate" / f"{split}.parquet")[
                ["image_id", "score"]
            ]
            for split in SPLITS
        ]
    )
    generic = distributions[distributions["rating_type"] == "generic"]
    merged = canonical.merge(
        generic[["image_id", "mean", "n_ratings"]], on="image_id", how="left"
    )
    delta = (merged["score"] - merged["mean"]).abs()
    report["canonical_vs_unfiltered_generic"] = {
        "canonical_rows": len(merged),
        "with_distribution": int(merged["mean"].notna().sum()),
        "without_distribution": int(merged["mean"].isna().sum()),
        "mean_abs_delta": round(float(delta.mean()), 5),
        "max_abs_delta": round(float(delta.max()), 5),
        "correlation": round(float(merged["score"].corr(merged["mean"])), 5),
    }

    # entropy_bits is sample-size dependent and must not be used to rank
    # images by disagreement; std is. Measured every build rather than
    # asserted once in prose, so a future rebuild cannot quietly drift.
    report["disagreement_measure_bias"] = {
        "note": (
            "Correlation of each disagreement measure with n_ratings. "
            "entropy_bits is biased upward by sample size (an image with 9 "
            "ratings cannot fill 10 bins); use std to compare images."
        ),
        "by_rating_type": {
            str(rating_type): {
                "entropy_bits_vs_n_ratings": round(
                    float(group["entropy_bits"].corr(group["n_ratings"])), 4
                ),
                "std_vs_n_ratings": round(
                    float(group["std"].corr(group["n_ratings"])), 4
                ),
                "mean_std": round(float(group["std"].mean()), 4),
            }
            for rating_type, group in distributions.groupby("rating_type")
        },
    }

    coverage = distributions["image_id"].nunique()
    report["image_coverage"] = {
        "images_with_any_distribution": int(coverage),
        "images_in_dataset": len(metadata),
        "images_with_no_ratings": len(metadata) - int(coverage),
    }

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
