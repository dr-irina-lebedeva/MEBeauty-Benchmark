"""Add rater-offset-adjusted scores and per-label uncertainty to the splits.

`score` -- the plain unweighted mean -- stays the primary label and is not
touched. It is reproducible in one line, matches the SCUT-FBP5500 convention,
and equals the mean of the shipped distribution exactly. This script adds
columns beside it:

    score_clean      plain mean, clearly-invalid raters dropped
    score_adjusted   fitted quality from the rater-offset model
    adjusted_lo/hi   95% rater-bootstrap interval on score_adjusted
    ci95             half-width of the t-interval on `score`
    n_ratings, std   support behind the label

"adjusted", not "debiased": the column names the operation performed, not a
claim that bias has been eliminated.

**Model selection is done, not assumed.** Plain mean, offset-only and affine
(per-rater scale) models are all fitted on 80% of ratings and compared on the
held-out 20%. The offset-only model ships unless affine measurably predicts
better -- a richer model that does not predict better is overfitting.

**Two assumptions are tested rather than trusted**, and both results are
written to the report:

- *Identifiability.* Anchoring offsets at mean zero is only meaningful if the
  rater x image graph is connected; otherwise separate components float
  independently and their offsets are not comparable.
- *Assignment confounding.* Offset subtraction assumes a rater's mean reflects
  their generosity, not which images they happened to receive. The diagnostic
  is the correlation between each rater's mean and the leave-one-rater-out
  quality of their assigned images, with a permutation test. Computing that
  quality *without* the rater's own ratings matters: including them
  self-correlates and roughly triples the apparent effect.

    uv run --with scipy python scripts/data/build_adjusted_scores.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/adjusted_scores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mebeauty_benchmark.legacy.aggregation import (
    OffsetModel,
    bootstrap_quality_ci,
    fit_affine_model,
    fit_offset_model,
    held_out_rmse,
)

SPLITS = ("train", "val", "test")

#: A rater below this many ratings cannot be characterised, so their offset
#: would be noise. Used only for `score_clean`; they still count in `score`.
MIN_RATINGS_FOR_VALIDITY = 10

#: Zero spread over at least this many ratings means the rater gave one number
#: to everything, which carries no information by construction. This is judged
#: without reference to whether they agree with anyone.
STRAIGHT_LINING_MIN_RATINGS = 5

#: RMSE improvement on held-out ratings below which the extra per-rater scale
#: parameter is treated as not earning its keep.
AFFINE_MARGIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap draws")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def connectivity(image_index, rater_index, n_images, n_raters) -> dict:
    """Is every rater's offset comparable to every other's?"""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    incidence = sp.coo_matrix(
        (np.ones(len(image_index)), (rater_index, image_index)),
        shape=(n_raters, n_images),
    )
    graph = sp.bmat(
        [
            [sp.csr_matrix((n_raters, n_raters)), incidence],
            [incidence.T, sp.csr_matrix((n_images, n_images))],
        ]
    )
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {
        "components": int(count),
        "largest_component_fraction": round(float(sizes.max() / sizes.sum()), 4),
        "anchor_identifiable": bool(count == 1),
    }


def assignment_confounding(
    ratings, image_index, rater_index, n_images, n_raters, seed, draws=300
) -> dict:
    """Do rater means reflect generosity, or the images they were given?"""
    rng = np.random.default_rng(seed)
    totals = np.bincount(image_index, weights=ratings, minlength=n_images)
    counts = np.bincount(image_index, minlength=n_images)
    # Leave-one-rating-out image mean: excludes the rating being explained.
    leave_out = np.divide(
        totals[image_index] - ratings,
        counts[image_index] - 1,
        out=np.full(len(ratings), np.nan),
        where=counts[image_index] > 1,
    )

    per_rater = np.bincount(rater_index, minlength=n_raters)
    eligible = per_rater >= 30
    valid = ~np.isnan(leave_out)

    def correlation(sample_quality: np.ndarray) -> float:
        mean_rating = np.divide(
            np.bincount(rater_index[valid], weights=ratings[valid], minlength=n_raters),
            np.bincount(rater_index[valid], minlength=n_raters),
            out=np.zeros(n_raters),
            where=np.bincount(rater_index[valid], minlength=n_raters) > 0,
        )
        mean_quality = np.divide(
            np.bincount(rater_index[valid], weights=sample_quality, minlength=n_raters),
            np.bincount(rater_index[valid], minlength=n_raters),
            out=np.zeros(n_raters),
            where=np.bincount(rater_index[valid], minlength=n_raters) > 0,
        )
        return float(np.corrcoef(mean_rating[eligible], mean_quality[eligible])[0, 1])

    observed = correlation(leave_out[valid])
    null = np.array(
        [correlation(rng.permutation(leave_out[valid])) for _ in range(draws)]
    )
    return {
        "correlation": round(observed, 4),
        "permutation_null_mean": round(float(null.mean()), 4),
        "permutation_null_sd": round(float(null.std()), 4),
        "p_value": round(float((np.abs(null) >= abs(observed)).mean()), 4),
        "note": (
            "Leave-one-rater-out. Using image means that include the rater's "
            "own ratings self-correlates and inflates this substantially."
        ),
    }


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    ratings_df = pd.read_parquet(
        v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    )

    images = sorted(ratings_df["image_id"].unique())
    raters = sorted(ratings_df["rater_id"].unique())
    image_lookup = {v: i for i, v in enumerate(images)}
    rater_lookup = {v: i for i, v in enumerate(raters)}

    values = ratings_df["score"].to_numpy(dtype=float)
    image_index = ratings_df["image_id"].map(image_lookup).to_numpy()
    rater_index = ratings_df["rater_id"].map(rater_lookup).to_numpy()
    n_images, n_raters = len(images), len(raters)
    print(f"{len(values)} ratings, {n_images} images, {n_raters} raters")

    graph = connectivity(image_index, rater_index, n_images, n_raters)
    confound = assignment_confounding(
        values, image_index, rater_index, n_images, n_raters, args.seed
    )
    print(f"  connectivity: {graph['components']} component(s)")
    print(
        f"  assignment confounding: r={confound['correlation']} "
        f"(p={confound['p_value']})"
    )

    # Repeat the comparison over several splits: a single random split can
    # favour either model by luck, and the choice should not hinge on a seed.
    folds = []
    for fold_seed in range(args.seed, args.seed + 5):
        rng = np.random.default_rng(fold_seed)
        folds.append(
            held_out_rmse(
                values,
                image_index,
                rater_index,
                n_images,
                n_raters,
                rng.random(len(values)) < 0.2,
            )
        )
    comparison = {
        model: float(np.mean([f[model] for f in folds]))
        for model in ("plain_mean", "offset", "affine")
    }
    gains = [f["offset"] - f["affine"] for f in folds]
    affine_gain = float(np.mean(gains))
    # Require the win in every fold, not just on average, so a single lucky
    # split cannot promote the more complex model.
    use_affine = affine_gain > AFFINE_MARGIN and all(g > 0 for g in gains)
    print(f"  held-out RMSE over 5 splits: {comparison}")
    print(f"  affine gain {affine_gain:+.4f}, wins {sum(g > 0 for g in gains)}/5 folds")
    print(f"  shipping {'affine' if use_affine else 'offset-only'} model")

    # Both are fitted and both ship. The selected model becomes
    # `score_adjusted`; the offset-only fit is kept beside it as the
    # conservative option, because a pure location shift cannot re-weight
    # anyone's opinion whatever the fit prefers.
    offset_fit = fit_offset_model(values, image_index, rater_index, n_images, n_raters)
    affine = fit_affine_model(values, image_index, rater_index, n_images, n_raters)
    fit = (
        OffsetModel(affine.quality, affine.offset, affine.iterations, affine.converged)
        if use_affine
        else offset_fit
    )
    scale_summary = {
        "min": round(float(affine.scale.min()), 4),
        "median": round(float(np.median(affine.scale)), 4),
        "max": round(float(affine.scale.max()), 4),
        "floored_at_zero": int((affine.scale == 0).sum()),
    }
    lower, upper = bootstrap_quality_ci(
        values,
        image_index,
        rater_index,
        n_images,
        n_raters,
        iterations=args.bootstrap,
        seed=args.seed,
    )

    stats = ratings_df.groupby("image_id")["score"].agg(["count", "mean", "std"])

    # score_clean: drop only raters whose ratings carry no information, judged
    # without reference to whether they agree with the majority.
    rater_stats = ratings_df.groupby("rater_id")["score"].agg(["count", "std"])
    invalid = set(
        rater_stats.index[
            (rater_stats["count"] < MIN_RATINGS_FOR_VALIDITY)
            | (
                (rater_stats["std"].fillna(0) == 0)
                & (rater_stats["count"] >= STRAIGHT_LINING_MIN_RATINGS)
            )
        ]
    )
    clean = ratings_df[~ratings_df["rater_id"].isin(invalid)]
    clean_mean = clean.groupby("image_id")["score"].mean()

    from scipy import stats as scipy_stats

    adjusted = pd.DataFrame(
        {
            "image_id": images,
            "score_adjusted": fit.quality,
            "score_adjusted_offset": offset_fit.quality,
            "adjusted_lo": lower,
            "adjusted_hi": upper,
        }
    )
    adjusted["n_ratings"] = adjusted["image_id"].map(stats["count"]).astype("Int64")
    adjusted["std"] = adjusted["image_id"].map(stats["std"])
    adjusted["score_clean"] = adjusted["image_id"].map(clean_mean)
    counts = adjusted["n_ratings"].astype(float)
    adjusted["ci95"] = np.where(
        counts > 1,
        scipy_stats.t.ppf(0.975, np.maximum(counts - 1, 1))
        * adjusted["std"]
        / np.sqrt(counts),
        np.nan,
    )

    written = {}
    for split in SPLITS:
        path = v3_dir / "ratings" / "aggregate" / f"{split}.parquet"
        # Keep whatever build_labels.py wrote (notably `score_all_raters`) and
        # drop only this script's own columns, so re-running is idempotent
        # without silently discarding an upstream column.
        frame = pd.read_parquet(path)
        frame = frame.drop(
            columns=[c for c in adjusted.columns if c != "image_id"], errors="ignore"
        )
        merged = frame.merge(adjusted, on="image_id", how="left")
        merged.to_parquet(path, index=False)
        written[split] = len(merged)

    report = {
        "primary_label": "score (plain unweighted mean, unchanged)",
        "model_shipped": "affine" if use_affine else "offset-only",
        "model_selection": {
            "held_out_rmse_mean_of_5_splits": {
                k: round(v, 5) for k, v in comparison.items()
            },
            "affine_gain_over_offset": round(affine_gain, 5),
            "affine_wins_folds": f"{sum(g > 0 for g in gains)}/5",
            "rater_scale_distribution": scale_summary,
            "margin_required": AFFINE_MARGIN,
            "note": (
                "Affine adds a per-rater scale. It ships only if it predicts "
                "held-out ratings better; otherwise the extra freedom is "
                "fitting noise from light raters."
            ),
        },
        "identifiability": graph,
        "assignment_confounding": confound,
        "offset_distribution": {
            "min": round(float(fit.offset.min()), 4),
            "p25": round(float(np.percentile(fit.offset, 25)), 4),
            "median": round(float(np.median(fit.offset)), 4),
            "p75": round(float(np.percentile(fit.offset, 75)), 4),
            "max": round(float(fit.offset.max()), 4),
        },
        "score_vs_adjusted": {
            "mean_abs_difference": round(
                float(
                    np.abs(
                        adjusted["score_adjusted"]
                        - adjusted["image_id"].map(stats["mean"])
                    ).mean()
                ),
                4,
            ),
            "max_abs_difference": round(
                float(
                    np.abs(
                        adjusted["score_adjusted"]
                        - adjusted["image_id"].map(stats["mean"])
                    ).max()
                ),
                4,
            ),
        },
        "score_clean": {
            "raters_dropped": len(invalid),
            "ratings_dropped": int(len(ratings_df) - len(clean)),
            "fraction_dropped": round(1 - len(clean) / len(ratings_df), 5),
            "criteria": (
                f"fewer than {MIN_RATINGS_FOR_VALIDITY} ratings, or zero spread "
                f"over at least {STRAIGHT_LINING_MIN_RATINGS} ratings"
            ),
        },
        "bootstrap_draws": args.bootstrap,
        "rows_written": written,
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
