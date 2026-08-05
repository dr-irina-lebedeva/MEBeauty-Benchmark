"""Add rater-offset-adjusted scores and per-label uncertainty to the splits.

`score` -- the rater-normalised mean from `build_labels.py` -- stays the
primary label and is not touched. This script adds columns beside it:

    score_adjusted   fitted quality from the rater-offset model
    adjusted_lo/hi   95% rater-bootstrap interval on score_adjusted
    ci95             half-width of the t-interval on the raw mean
    n_ratings, std   support behind the label

"adjusted", not "debiased": the column names the operation performed, not a
claim that bias has been eliminated.

**This is a cross-check on `score`, not a competitor.** Both correct for the
same thing -- raters using the scale differently -- by different routes:
`score` standardises each rater independently, this model estimates offsets
(and optionally scales) jointly by alternating least squares. They should
agree closely; the report measures how closely, and a divergence would mean
one of them is wrong. Every rating on a labelled image is used, with no rater
screened out, so `score_adjusted` is recomputable from exactly the ratings the
release ships.

**Model selection is done, not assumed -- and on the right criterion.** Plain
mean, offset and affine models are compared by *split-half reliability of the
image score*: split the raters in two, aggregate each half, correlate. Held-out
RMSE on individual ratings is also reported, but it is not what selects, because
it answers a different question and here it picks the worse label -- affine wins
on RMSE and loses on reliability. See `legacy/aggregation.py`.

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
    OFFSET_SHRINKAGE,
    OffsetModel,
    bootstrap_quality_ci,
    fit_affine_model,
    fit_offset_model,
    held_out_rmse,
    split_half_reliability,
)
from mebeauty_benchmark.legacy.validity import (
    MIN_RATINGS_PER_IMAGE,
    labelled_image_ids,
)

SPLITS = ("train", "val", "test")

#: Split-half reliability gain below which the extra per-rater scale parameter
#: is treated as not earning its keep.
AFFINE_MARGIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap draws")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reliability-splits",
        type=int,
        default=40,
        help="Rater split-halves used for model selection",
    )
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
    all_ratings = pd.read_parquet(
        v3_dir / "ratings" / "by_rater" / "ratings_by_rater.parquet"
    )
    # **Every rating, no rater screened out.** `score_mean` is published as the
    # plain mean over all raters, and `score_adjusted` sits beside it in the
    # same files; fitting this one on a subset would publish a label no user
    # could recompute from the ratings they were given. The `rater_valid` flag
    # is still carried in the private tier for anyone who wants to apply it.
    labelled = labelled_image_ids(all_ratings, MIN_RATINGS_PER_IMAGE)
    ratings_df = all_ratings[all_ratings["image_id"].isin(labelled)].reset_index(
        drop=True
    )
    print(
        f"Screened {len(all_ratings)} -> {len(ratings_df)} ratings "
        f"({all_ratings['image_id'].nunique()} -> {len(labelled)} images)"
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
    print(f"  held-out RMSE over 5 splits: {comparison}")

    # **Selection runs on reliability, not on held-out RMSE.** RMSE asks which
    # model predicts one person's rating; the label is a consensus, so the
    # question is whether another panel would reproduce it. The two criteria
    # disagree here: affine wins on RMSE and loses on reliability. See the
    # `aggregation` module docstring.
    reliability = split_half_reliability(
        values,
        image_index,
        rater_index,
        n_images,
        n_raters,
        seeds=args.reliability_splits,
    )
    print(f"  split-half reliability: {reliability}")
    use_affine = reliability["affine"] > reliability["offset"] + AFFINE_MARGIN
    print(f"  shipping {'affine' if use_affine else 'offset'} model")

    # Both are fitted and both ship. The selected model becomes
    # `score_adjusted`; the unshrunk offset fit is kept beside it so the
    # effect of shrinkage stays visible to anyone auditing the label.
    offset_fit = fit_offset_model(
        values, image_index, rater_index, n_images, n_raters, OFFSET_SHRINKAGE
    )
    unshrunk = fit_offset_model(values, image_index, rater_index, n_images, n_raters)
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

    from scipy import stats as scipy_stats

    adjusted = pd.DataFrame(
        {
            "image_id": images,
            "score_adjusted": fit.quality,
            "score_adjusted_offset": unshrunk.quality,
            "adjusted_lo": lower,
            "adjusted_hi": upper,
        }
    )
    adjusted["n_ratings"] = adjusted["image_id"].map(stats["count"]).astype("Int64")
    adjusted["std"] = adjusted["image_id"].map(stats["std"])
    counts = adjusted["n_ratings"].astype(float)
    adjusted["ci95"] = np.where(
        counts > 1,
        scipy_stats.t.ppf(0.975, np.maximum(counts - 1, 1))
        * adjusted["std"]
        / np.sqrt(counts),
        np.nan,
    )

    # How closely the joint model agrees with the independently-normalised
    # `score`. These are two different routes to the same correction, so
    # strong agreement is evidence for both and a divergence indicts one.
    canonical = pd.concat(
        [
            pd.read_parquet(v3_dir / "ratings" / "aggregate" / f"{s}.parquet")[
                ["image_id", "score"]
            ]
            for s in SPLITS
        ]
    )
    paired = adjusted.merge(canonical, on="image_id", how="inner")
    difference = (paired["score_adjusted"] - paired["score"]).abs()
    agreement = {
        "images_compared": len(paired),
        "pearson": round(float(paired["score_adjusted"].corr(paired["score"])), 4),
        "spearman": round(
            float(paired["score_adjusted"].corr(paired["score"], method="spearman")), 4
        ),
        "mean_abs_difference": round(float(difference.mean()), 4),
        "max_abs_difference": round(float(difference.max()), 4),
        "note": (
            "score normalises each rater independently; score_adjusted fits "
            "offsets jointly. High agreement means the correction is a "
            "property of the ratings, not of either method."
        ),
    }
    print(
        f"  agreement with score: r={agreement['pearson']}, "
        f"mean |diff| {agreement['mean_abs_difference']}"
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
        "primary_label": "score (rater-normalised mean, unchanged by this script)",
        "model_shipped": "affine" if use_affine else "offset",
        "model_selection": {
            "criterion": "split_half_reliability_of_image_score",
            "split_half_reliability": {k: round(v, 5) for k, v in reliability.items()},
            "affine_gain_over_offset": round(
                reliability["affine"] - reliability["offset"], 5
            ),
            "held_out_rmse_mean_of_5_splits": {
                k: round(v, 5) for k, v in comparison.items()
            },
            "rater_scale_distribution": scale_summary,
            "margin_required": AFFINE_MARGIN,
            "offset_shrinkage": OFFSET_SHRINKAGE,
            "note": (
                "Selection is on split-half reliability of the image score, "
                "not on held-out RMSE of individual ratings. The two disagree "
                "here: affine predicts single ratings better while producing a "
                "less reproducible image score, so RMSE would pick the worse "
                "label. Both figures are reported so the disagreement is "
                "visible rather than buried."
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
        "agreement_with_score": agreement,
        "bootstrap_draws": args.bootstrap,
        "rows_written": written,
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
