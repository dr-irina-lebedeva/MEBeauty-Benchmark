"""Measure per-rater scoring quality in the per-rater ratings table.

Identity reconciliation (`build_ratings_by_rater.py`) checked *who* rated
what; this measures whether individual raters' scores carry signal.

**This reports, it does not filter.** No rater is excluded from the
canonical aggregate on the basis of anything computed here -- see
`docs/DATASET_AUDIT.md` Finding 14 for the evidence behind that choice.
The short version: filtering raters by agreement-with-consensus is
circular (it defines "good rater" as "agrees with the majority", inflates
apparent inter-rater reliability, and deletes minority aesthetic
viewpoints), and this dataset's stated research purpose includes studying
exactly that variation. There is also no principled threshold to filter
at -- the discrimination statistic below is a smooth continuum with no
gap, so the fraction of data removed swings from 7% to 19% across equally
defensible cutoffs. The per-rater table this writes lets any consumer
apply their own rule in one line; baking one in here would not be
reversible.

Statistics computed per rater:

- **std** -- zero (or near-zero) std over many *different* images is
  straight-lining: a real judgment of dozens of different faces
  essentially never produces one identical score every time.
- **discrimination spread** -- the rater's mean score on images the *rest*
  of the pool scored in its top third, minus their mean on the bottom
  third (consensus computed leave-one-out, so a rater never contributes
  to the yardstick they're measured against). This is the informative
  statistic: it survives range restriction, so a genuinely harsh or
  generous rater compressed against one end of the scale still shows a
  clearly positive spread, while a rater whose scores are unrelated to
  the image sits near zero.
- **leave-one-out correlation** -- same idea, as a correlation. Reported
  for completeness; attenuated by range restriction, so `spread` is the
  more trustworthy of the two for extreme-mean raters.

Why not "extreme mean" (this script's previous heuristic): flagging
raters whose mean sits within 0.5 of the scale ends measures the wrong
construct. Verified against this dataset: it false-positived two raters
who discriminate perfectly well and merely score generously (spread +0.92
and +0.69), while entirely missing several high-volume raters with
strongly *negative* discrimination whose means look unremarkable (e.g.
310 ratings at spread -2.90). An extreme mean is a scale preference; a
flat spread is an absence of signal. Only the latter is a quality
problem, and it is not what the old test measured.

    uv run python scripts/data/check_rater_quality.py \\
        --ratings data/mebeauty_v3/ratings/by_rater/ratings_by_rater.parquet \\
        --output reports/legacy_audit/rater_quality_report.json \\
        --table-out data/mebeauty_v3/ratings/by_rater/rater_quality.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings", required=True, help="ratings_by_rater.parquet path"
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=10,
        help="Minimum ratings before reporting std-based flags for a rater",
    )
    parser.add_argument(
        "--min-ratings-discrimination",
        type=int,
        default=30,
        help="Minimum ratings before computing discrimination spread "
        "(needs enough per consensus tercile to be meaningful)",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.01,
        help="Max std to count as straight-lining",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--table-out",
        default=None,
        help="Optional: write the full per-rater quality table as parquet "
        "(intended to ship with the dataset so consumers can filter themselves)",
    )
    return parser.parse_args()


def build_quality_table(df: pd.DataFrame, min_discrimination: int) -> pd.DataFrame:
    """Per-rater stats, including leave-one-out discrimination measures."""
    # Consensus is computed within a rating task, never across them. The
    # generic and date tasks sit about a point apart (means 6.00 vs 5.01), so
    # a pooled consensus would score every date rating as "below consensus"
    # and every generic rating as "above" it for reasons that have nothing to
    # do with the rater. See docs/DATASET_AUDIT.md Finding 16.
    group_keys = ["image_id"]
    if "rating_type" in df.columns:
        group_keys.append("rating_type")

    totals = df.groupby(group_keys)["score"].agg(["sum", "count"])
    joined = df.join(totals, on=group_keys)

    # Leave-one-out consensus: the mean of everyone *else* on that image, so a
    # rater never helps define the yardstick they are measured against.
    # Images rated only once give no consensus and are dropped for this stat.
    joined = joined[joined["count"] > 1].copy()
    joined["loo"] = (joined["sum"] - joined["score"]) / (joined["count"] - 1)

    stats = df.groupby("rater_id")["score"].agg(["count", "mean", "std"]).reset_index()

    rows = []
    for rater_id, group in joined.groupby("rater_id"):
        spread = correlation = np.nan
        if len(group) >= min_discrimination:
            bands = pd.qcut(
                group["loo"], 3, labels=["low", "mid", "high"], duplicates="drop"
            )
            if bands.nunique() == 3:
                band_means = group.groupby(bands, observed=True)["score"].mean()
                spread = band_means["high"] - band_means["low"]
            if group["score"].std() > 0:
                correlation = np.corrcoef(group["score"], group["loo"])[0, 1]
        rows.append(
            {
                "rater_id": rater_id,
                "n_with_consensus": len(group),
                "discrimination_spread": spread,
                "loo_correlation": correlation,
            }
        )

    return stats.merge(pd.DataFrame(rows), on="rater_id", how="left").rename(
        columns={"count": "n_ratings", "mean": "mean_score", "std": "std_score"}
    )


def sensitivity_analysis(df: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    """What each candidate exclusion rule would cost, if one were applied.

    Deliberately reported rather than acted on: the point is to show that no
    threshold is privileged, and to let a consumer see the cost of theirs.
    """
    measured = table.dropna(subset=["discrimination_spread"])
    results = []
    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
        excluded = measured[measured["discrimination_spread"] <= threshold]
        n_ratings = int(excluded["n_ratings"].sum())
        results.append(
            {
                "spread_threshold": threshold,
                "raters_excluded": len(excluded),
                "ratings_excluded": n_ratings,
                "ratings_excluded_fraction": round(n_ratings / len(df), 4),
            }
        )
    return results


def aggregate_impact(df: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    """How far per-image means would move under each candidate rule."""
    baseline = df.groupby("image_id")["score"].mean()
    measured = table.dropna(subset=["discrimination_spread"])

    results = []
    for threshold in [0.0, 0.5, 1.0]:
        drop = set(
            measured.loc[measured["discrimination_spread"] <= threshold, "rater_id"]
        )
        kept = df[~df["rater_id"].isin(drop)]
        filtered = kept.groupby("image_id")["score"].mean()
        common = baseline.index.intersection(filtered.index)
        delta = (filtered[common] - baseline[common]).abs()
        results.append(
            {
                "spread_threshold": threshold,
                "images_losing_all_ratings": len(
                    baseline.index.difference(filtered.index)
                ),
                "mean_abs_shift": round(float(delta.mean()), 4),
                "max_abs_shift": round(float(delta.max()), 4),
                "images_shifting_over_0.25": int((delta > 0.25).sum()),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ratings)
    table = build_quality_table(df, args.min_ratings_discrimination)

    eligible = table[table["n_ratings"] >= args.min_ratings]
    straight_lining = eligible[eligible["std_score"].fillna(0) < args.std_threshold]

    measured = table.dropna(subset=["discrimination_spread"])
    non_discriminating = measured[measured["discrimination_spread"] <= 0]

    columns = [
        "rater_id",
        "n_ratings",
        "mean_score",
        "std_score",
        "discrimination_spread",
        "loo_correlation",
    ]
    report = {
        "policy": (
            "REPORT ONLY -- no rater is excluded from the canonical aggregate on "
            "the basis of these statistics. See docs/DATASET_AUDIT.md Finding 14."
        ),
        "total_raters": len(table),
        "total_ratings": len(df),
        "raters_with_discrimination_measured": len(measured),
        "min_ratings_for_discrimination": args.min_ratings_discrimination,
        "discrimination_spread_distribution": {
            "min": round(float(measured["discrimination_spread"].min()), 4),
            "p25": round(float(measured["discrimination_spread"].quantile(0.25)), 4),
            "median": round(float(measured["discrimination_spread"].median()), 4),
            "p75": round(float(measured["discrimination_spread"].quantile(0.75)), 4),
            "max": round(float(measured["discrimination_spread"].max()), 4),
            "note": (
                "Smooth continuum with no bimodal gap -- this is why no exclusion "
                "threshold is applied: every cutoff is arbitrary, and the amount of "
                "data it removes swings several-fold across equally defensible values."
            ),
        },
        "straight_lining_raters": straight_lining[columns].to_dict("records"),
        "lowest_discrimination_raters": non_discriminating.nsmallest(
            20, "discrimination_spread"
        )[columns].to_dict("records"),
        "non_discriminating_rater_count": len(non_discriminating),
        "non_discriminating_rating_count": int(non_discriminating["n_ratings"].sum()),
        "exclusion_rule_sensitivity": sensitivity_analysis(df, table),
        "aggregate_impact_if_excluded": aggregate_impact(df, table),
        "top_10_raters_by_volume_for_context": table.nlargest(10, "n_ratings")[
            columns
        ].to_dict("records"),
    }
    output_path.write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    print(f"{len(table)} raters, {len(df)} ratings")
    print(
        f"  discrimination measured for {len(measured)} (>= {args.min_ratings_discrimination} ratings)"
    )
    print(
        f"  spread range {measured['discrimination_spread'].min():.2f} .. "
        f"{measured['discrimination_spread'].max():.2f} (median "
        f"{measured['discrimination_spread'].median():.2f}) -- continuum, no gap"
    )
    print(f"  {len(straight_lining)} straight-lining (std < {args.std_threshold})")
    print(
        f"  {len(non_discriminating)} with spread <= 0 "
        f"({non_discriminating['n_ratings'].sum():,} ratings, "
        f"{non_discriminating['n_ratings'].sum() / len(df):.1%} of all)"
    )
    print("\nNo raters excluded -- reporting only. Sensitivity if one did filter:")
    for row in report["exclusion_rule_sensitivity"]:
        print(
            f"  spread <= {row['spread_threshold']:.2f}: "
            f"{row['raters_excluded']:3d} raters, "
            f"{row['ratings_excluded']:6,} ratings "
            f"({row['ratings_excluded_fraction']:.2%})"
        )

    if args.table_out:
        table_path = Path(args.table_out).expanduser().resolve()
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(table_path, index=False)
        print(f"\nWrote {table_path} ({len(table)} raters)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
