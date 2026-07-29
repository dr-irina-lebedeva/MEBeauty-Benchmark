"""Build per-image rating distributions (soft labels) for the v3 dataset.

Writes `ratings/distributions.parquet`: one row per image
with raw `counts` and normalized `probabilities` over the 1-10 scale, plus
mean/median/std/entropy. This is what makes MEBeauty usable for label
distribution learning, the dominant paradigm in facial beauty prediction --
the canonical single `score` alone cannot support it.

Built from the **valid** raters in `ratings/by_rater/ratings_by_rater.parquet`
-- the same rows behind `score`, so the point label and the soft label can
never disagree; `build_labels.py` asserts it against `score_raw_mean`.
Validity is behavioural only (see `legacy/validity.py`); it never depends on
agreeing with anyone.

Only images with at least `MIN_RATINGS_PER_IMAGE` valid ratings get a
distribution. A ten-bin soft label estimated from three ratings is mostly
zeros and a rounding artefact, and shipping it would invite training on it.

Run this **before** `build_labels.py`, which verifies against the output.

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
    build_soft_distributions,
)
from mebeauty_benchmark.legacy.normalization import (
    DEFAULT_SHRINKAGE,
    normalise_ratings,
)
from mebeauty_benchmark.legacy.validity import (
    MIN_RATINGS_PER_IMAGE,
    labelled_image_ids,
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

    # The canonical score is built from valid raters over sufficiently rated
    # images, so the distribution must be built from exactly those rows.
    valid = per_rater[per_rater["rater_valid"]]
    print(
        f"  valid raters: {valid['rater_id'].nunique()} of {per_rater['rater_id'].nunique()}"
    )
    labelled = labelled_image_ids(per_rater, MIN_RATINGS_PER_IMAGE)
    supported = valid[valid["image_id"].isin(labelled)]
    print(
        f"  images with >={MIN_RATINGS_PER_IMAGE} valid ratings: {len(labelled)} "
        f"of {valid['image_id'].nunique()} "
        f"({len(valid) - len(supported)} ratings left unlabelled)"
    )
    distributions = build_distributions(supported)

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
    if int(distributions["n_ratings"].min()) < MIN_RATINGS_PER_IMAGE:
        raise AssertionError(
            f"a distribution rests on fewer than {MIN_RATINGS_PER_IMAGE} ratings"
        )

    output_path = v3_dir / "ratings" / "distributions.parquet"
    distributions.to_parquet(output_path, index=False)
    print(f"Wrote {len(distributions)} distributions to {output_path}")

    # The soft label that matches the canonical `score`. `distributions.parquet`
    # counts the integer scores raters gave, so its mean is `score_raw_mean`;
    # a method trained on it and scored against `score` is being asked for one
    # quantity and graded on another. Measured on this dataset that cost the
    # LDL entry ~0.09 MAE. This one bins the *normalised* ratings, so its mean
    # is `score` exactly.
    # Rater scales are estimated from *all* of a rater's valid ratings, exactly
    # as `build_labels.py` does -- including their ratings on images that will
    # not be labelled, because a rater's scale habit is a property of the
    # person. Normalising over only the supported subset instead gives every
    # rater slightly different statistics, and the resulting soft label no
    # longer has `score` as its mean. That mismatch is not theoretical: it was
    # caught here by the protocol's expectation check, at 0.09.
    normalised_rows = normalise_ratings(
        valid[["image_id", "rater_id", "score"]], shrinkage=DEFAULT_SHRINKAGE
    )
    soft = build_soft_distributions(
        normalised_rows[normalised_rows["image_id"].isin(labelled)]
    )
    probability_sums = soft["probabilities"].apply(sum)
    if not probability_sums.sub(1.0).abs().lt(1e-9).all():
        raise AssertionError("some normalised probability vectors do not sum to 1")
    expectation = soft["probabilities"].apply(
        lambda p: sum(b * w for b, w in zip(SCORE_BINS, p, strict=True))
    )
    drift = float((expectation - soft["mean"]).abs().max())
    if drift > 1e-9:
        raise AssertionError(
            f"soft distribution expectation drifts from its mean by {drift}"
        )
    soft_path = v3_dir / "ratings" / "distributions_normalised.parquet"
    soft.to_parquet(soft_path, index=False)
    print(f"Wrote {len(soft)} normalised distributions to {soft_path}")
    print(f"  expectation == mean to {drift:.1e}")

    report: dict[str, object] = {
        "score_bins": list(SCORE_BINS),
        "max_entropy_bits": round(MAX_ENTROPY_BITS, 5),
        "rows": len(distributions),
        "rater_screening": "behavioural validity only (see legacy/validity.py)",
        "min_ratings_per_image_threshold": MIN_RATINGS_PER_IMAGE,
        "rating_task": "generic",
        "note": (
            "Built from valid raters over images with at least "
            f"{MIN_RATINGS_PER_IMAGE} such ratings -- the same rows as "
            "`score`, so the two always agree. Screening is behavioural, "
            "never based on agreement with other raters."
        ),
        "images": int(distributions["image_id"].nunique()),
        "ratings": int(distributions["n_ratings"].sum()),
        "observed_min_ratings_per_image": int(distributions["n_ratings"].min()),
        "median_ratings_per_image": float(distributions["n_ratings"].median()),
        "max_ratings_per_image": int(distributions["n_ratings"].max()),
        "mean_entropy_bits": round(float(distributions["entropy_bits"].mean()), 5),
        "normalised_distribution": {
            "file": "ratings/distributions_normalised.parquet",
            "rows": len(soft),
            "expectation_vs_mean_max_drift": drift,
            "note": (
                "Soft-binned normalised ratings. Its expectation is `score`; "
                "distributions.parquet's expectation is `score_raw_mean`. A "
                "benchmark must use the one matching the label it scores on."
            ),
        },
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
        "entropy_bits_vs_n_ratings": round(
            float(distributions["entropy_bits"].corr(distributions["n_ratings"])), 4
        ),
        "std_vs_n_ratings": round(
            float(distributions["std"].corr(distributions["n_ratings"])), 4
        ),
        "mean_std": round(float(distributions["std"].mean()), 4),
    }

    coverage = distributions["image_id"].nunique()
    report["image_coverage"] = {
        "images_with_a_distribution": int(coverage),
        "images_in_dataset": len(metadata),
        "images_without_a_distribution": len(metadata) - int(coverage),
        "note": (
            "Images below the support threshold keep their pixels and "
            "metadata but carry no label and enter no split."
        ),
    }

    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
