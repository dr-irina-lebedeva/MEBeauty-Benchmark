"""Attach rater-support columns to the canonical aggregate ratings.

The canonical `score` is inherited from the legacy release and cannot be
recomputed exactly -- the 2021 cleaning pipeline's intermediate files are
gone (`docs/DATASET_AUDIT.md`, Finding 20). This script does not change it.
It attaches the rater support behind each label, and a second, reproducible
label beside it:

    n_ratings                 generic ratings backing this image
    score_std                 how much those raters disagreed
    score_mean                plain unweighted mean of them, no rater excluded
    score_delta               score - score_mean
    diverges_from_score_mean  |score_delta| > 0.5

`score_mean` is the SCUT-FBP5500-style label: a plain mean of the released
per-rater ratings, reproducible in one line and equal to the mean of
`ratings/distributions.parquet`. `score` stays byte-identical and remains the
label comparable with the published paper.

Run with `--from-raw` (recommended) to derive support from the complete
per-rater layer, which includes the in-house panel. Without it, support comes
from the legacy workbook instead -- kept only for reproducing older reports.

    uv run python scripts/data/enrich_label_provenance.py \\
        --legacy-copy data/legacy_snapshot \\
        --v3 data/mebeauty_v3 --from-raw \\
        --report-out reports/legacy_audit/label_provenance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.label_provenance import (
    DIVERGENCE_THRESHOLD,
    attach_label_provenance,
    compute_label_support,
    support_frame,
)

SPLITS = ("train", "val", "test")

#: The post-cleaning generic rater matrix. Verified to be the closest
#: surviving source to the canonical labels (mad 0.012); the `public_*`
#: workbooks are a different, larger rater pool and match far worse.
SCORES_WORKBOOK = "scores/generic_scores_all_2022.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="data/legacy_snapshot directory"
    )
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Derive support from ratings_by_rater.parquet (the complete raw "
            "layer, including the in-house panel) rather than from the legacy "
            "workbook. This is the SCUT-style label source."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DIVERGENCE_THRESHOLD,
        help=f"Discrepancy flag threshold (default {DIVERGENCE_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    v3_dir = Path(args.v3).expanduser().resolve()

    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    filename_by_image_id = dict(zip(metadata["image_id"], metadata["legacy_filename"]))

    if args.from_raw:
        # Preferred source: the complete per-rater layer, which now includes
        # the in-house panel. `score_mean` here is the SCUT-style label --
        # a plain unweighted mean, reproducible in one line from the ratings
        # this dataset ships, with no rater excluded.
        per_rater = pd.read_parquet(
            v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
        )
        generic = per_rater[per_rater["rating_type"] == "generic"]
        stats = generic.groupby("image_id")["score"].agg(["count", "mean", "std"])
        support = pd.DataFrame(
            {
                "legacy_filename": stats.index.map(filename_by_image_id),
                "n_ratings": stats["count"].to_numpy(),
                "score_mean": stats["mean"].to_numpy(),
                "score_std": stats["std"].to_numpy(),
            }
        )
        print(
            f"Rater support from the raw per-rater layer: {len(support)} images, "
            f"{len(generic)} generic ratings, {generic['rater_id'].nunique()} raters"
        )
    else:
        workbook = pd.read_excel(legacy_root / SCORES_WORKBOOK)
        support = support_frame(compute_label_support(workbook))
        print(f"Rater support computed for {len(support)} images in {SCORES_WORKBOOK}")

    report: dict[str, object] = {
        "support_source": (
            "ratings/by_rater/ratings_by_rater.parquet (generic, unfiltered)"
            if args.from_raw
            else SCORES_WORKBOOK
        ),
        "score_mean_is_filtered": False,
        "divergence_threshold": args.threshold,
        "images_with_support": len(support),
        "note": (
            "`score` is the legacy consensus-filtered label; `score_mean` is a "
            "plain unweighted mean with no rater excluded. They differ "
            "systematically by construction -- score_delta measures that, it "
            "is not an error term."
        ),
        "splits": {},
    }
    ratings_dir = v3_dir / "ratings" / "aggregate"
    for split in SPLITS:
        path = ratings_dir / f"{split}.parquet"
        original = pd.read_parquet(path)
        enriched = attach_label_provenance(
            original, support, filename_by_image_id, threshold=args.threshold
        )

        # The canonical label is the one thing this script must not touch.
        if not enriched["score"].equals(original["score"]):
            raise AssertionError(f"{split}: canonical score column was modified")

        enriched.to_parquet(path, index=False)
        delta = enriched["score_delta"].abs()
        report["splits"][split] = {
            "rows": len(enriched),
            "with_support": int(enriched["n_ratings"].notna().sum()),
            "without_support": int(enriched["n_ratings"].isna().sum()),
            "diverging_from_score_mean": int(
                enriched["diverges_from_score_mean"].sum()
            ),
            "mean_abs_delta": round(float(delta.mean()), 5),
            "max_abs_delta": round(float(delta.max()), 5),
            "median_n_ratings": float(enriched["n_ratings"].median()),
        }
        print(
            f"  {split}: {len(enriched)} rows, "
            f"{report['splits'][split]['diverging_from_score_mean']} flagged, "
            f"mad {report['splits'][split]['mean_abs_delta']}"
        )

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
