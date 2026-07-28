"""Attach rater-support columns to the canonical aggregate ratings.

The canonical scores are inherited from the legacy release and cannot be
recomputed exactly -- the 2021 cleaning pipeline's intermediate files are
gone (`docs/DATASET_AUDIT.md`, Finding 20). This script does not try to
recompute them. It measures the support behind each label from the
surviving post-cleaning rater matrix and ships that alongside:

    n_ratings          how many ratings back this label
    score_std          how much those raters disagreed
    recomputed_score   what a plain mean of them would give
    score_delta        canonical - recomputed
    label_discrepancy  |score_delta| > 0.25

`score` itself is passed through byte-identical. A consumer who recomputes
labels and gets different numbers can now see why, per image, instead of
filing it as a bug.

    uv run python scripts/data/enrich_label_provenance.py \\
        --legacy-copy data/legacy_snapshot \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/label_provenance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.label_provenance import (
    DISCREPANCY_THRESHOLD,
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
        "--threshold",
        type=float,
        default=DISCREPANCY_THRESHOLD,
        help=f"Discrepancy flag threshold (default {DISCREPANCY_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    v3_dir = Path(args.v3).expanduser().resolve()

    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    filename_by_image_id = dict(zip(metadata["image_id"], metadata["legacy_filename"]))

    workbook = pd.read_excel(legacy_root / SCORES_WORKBOOK)
    support = support_frame(compute_label_support(workbook))
    print(f"Rater support computed for {len(support)} images in {SCORES_WORKBOOK}")

    report: dict[str, object] = {
        "source_workbook": SCORES_WORKBOOK,
        "discrepancy_threshold": args.threshold,
        "images_in_workbook": len(support),
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
            "flagged_discrepancy": int(enriched["label_discrepancy"].sum()),
            "mean_abs_delta": round(float(delta.mean()), 5),
            "max_abs_delta": round(float(delta.max()), 5),
            "median_n_ratings": float(enriched["n_ratings"].median()),
        }
        print(
            f"  {split}: {len(enriched)} rows, "
            f"{report['splits'][split]['flagged_discrepancy']} flagged, "
            f"mad {report['splits'][split]['mean_abs_delta']}"
        )

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
