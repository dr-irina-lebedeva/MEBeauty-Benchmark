"""Choose the shrinkage constant by measurement rather than by analogy.

`legacy/normalization.py` shrinks each rater's mean and variance toward the
global values, weighted by how much evidence that rater provides. The strength
`k` is the number of ratings at which a rater is trusted halfway.

`k = 10` was originally chosen *by analogy* -- images need 10 ratings to be
labelled, so 10 felt like where a rater becomes measurable. That is a story,
not evidence, and the same file already selects between the offset and affine
models by held-out prediction. This script applies the same standard to `k`.

**The test.** Hold out a random 20% of ratings. Fit rater scales and per-image
normalised scores on the other 80%. Then predict each held-out rating as

    predicted_ij = shrunk_mean_j + shrunk_std_j * z_i

-- rater j's own scale applied to image i's normalised quality -- and measure
RMSE against what that rater actually gave. A `k` that is too small lets noisy
light raters distort their own scale; too large and every rater is dragged to
the global mean and genuine differences are erased. The minimum is the value
the data supports.

Repeated over several folds, because a single split can favour a value by
luck.

    uv run python scripts/data/validate_shrinkage.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/shrinkage_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mebeauty_benchmark.legacy.normalization import DEFAULT_SHRINKAGE, rater_scales

#: Values to compare. 0 is plain z-scoring (no shrinkage at all), which is the
#: thing shrinkage exists to avoid; it is included so its cost is visible
#: rather than asserted.
CANDIDATES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def held_out_rmse(
    train: pd.DataFrame, test: pd.DataFrame, shrinkage: float
) -> float | None:
    """RMSE predicting held-out ratings from scales fitted on `train`."""
    scales = rater_scales(train, shrinkage).set_index("rater_id")

    merged = train.merge(
        scales[["shrunk_mean", "shrunk_std"]], left_on="rater_id", right_index=True
    )
    # Guard: with shrinkage 0 a single-rating rater has zero spread, so their
    # z is undefined. Those rows cannot inform the image score.
    merged["z"] = (merged["score"] - merged["shrunk_mean"]) / merged["shrunk_std"]
    merged = merged[np.isfinite(merged["z"])]
    quality = merged.groupby("image_id")["z"].mean()

    evaluated = test.join(scales, on="rater_id").join(
        quality.rename("z_image"), on="image_id"
    )
    # A held-out rating whose rater or image was never seen in training cannot
    # be predicted by any value of k, so it is excluded from all of them
    # equally rather than imputed.
    evaluated = evaluated.dropna(subset=["shrunk_mean", "shrunk_std", "z_image"])
    if evaluated.empty:
        return None

    predicted = (
        evaluated["shrunk_mean"] + evaluated["shrunk_std"] * evaluated["z_image"]
    )
    predicted = np.clip(predicted, 1.0, 10.0)
    residual = predicted - evaluated["score"]
    return float(np.sqrt((residual**2).mean()))


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    ratings = pd.read_parquet(
        v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    )
    ratings = ratings[ratings["rater_valid"]][["image_id", "rater_id", "score"]]
    print(f"{len(ratings)} valid ratings, {ratings['rater_id'].nunique()} raters")

    results: dict[float, list[float]] = {k: [] for k in CANDIDATES}
    for fold in range(args.folds):
        rng = np.random.default_rng(args.seed + fold)
        held_out = rng.random(len(ratings)) < 0.2
        train, test = ratings[~held_out], ratings[held_out]
        for k in CANDIDATES:
            score = held_out_rmse(train, test, k)
            if score is not None:
                results[k].append(score)
        print(f"  fold {fold + 1}/{args.folds} done", flush=True)

    means = {k: float(np.mean(v)) for k, v in results.items() if v}
    best = min(means, key=means.get)
    print("\nHeld-out RMSE by shrinkage:")
    for k in CANDIDATES:
        if k in means:
            marker = "  <-- best" if k == best else ""
            current = "  (current default)" if k == DEFAULT_SHRINKAGE else ""
            print(f"  k={k:<6} {means[k]:.5f}{marker}{current}")

    penalty = means[DEFAULT_SHRINKAGE] - means[best]
    report = {
        "method": (
            "5-fold held-out rating prediction; predicted_ij = "
            "shrunk_mean_j + shrunk_std_j * z_i"
        ),
        "folds": args.folds,
        "candidates": list(CANDIDATES),
        "held_out_rmse": {str(k): round(v, 5) for k, v in means.items()},
        "best_k": best,
        "current_default_k": DEFAULT_SHRINKAGE,
        "penalty_of_current_default": round(penalty, 5),
        "no_shrinkage_penalty": round(means[0.0] - means[best], 5)
        if 0.0 in means
        else None,
        "note": (
            "k=0 is plain z-scoring. Its gap from the best value is the "
            "measured cost of not shrinking, which is the reason shrinkage "
            "is in the pipeline at all."
        ),
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nBest k={best}; current default k={DEFAULT_SHRINKAGE} costs {penalty:+.5f}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
