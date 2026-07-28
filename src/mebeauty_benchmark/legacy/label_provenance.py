"""Rater support behind each canonical attractiveness label.

The canonical scores in `ratings/aggregate/*.parquet` are inherited from the
legacy release. They are **not** a plain mean of the surviving rating files,
and were never meant to be: the 2021 pipeline
(`MEBeauty_creation_cleaning/*.ipynb`) applied rater cleaning before
averaging — per-image outlier masking and a consensus-correlation rater
drop, among steps that were written but never actually ran. See
`docs/DATASET_AUDIT.md`, Finding 20, for which steps took effect and how
each verdict was tested.

The inputs to those notebooks (`pers.xlsx`, `generic_all_path.xlsx`,
`generic_all_pure.xlsx`) lived on a machine that no longer exists, so the
canonical scores cannot be recomputed exactly.

This module attaches, per image, the rater support behind the label: how many
raters contributed, how much they disagreed, and `score_mean` -- the plain
unweighted mean of every `generic` rating this dataset ships, with no rater
excluded.

**`score_mean` is the SCUT-FBP5500-style label.** SCUT's `All_labels.txt`
mean is a plain mean of its released per-rater ratings, and that
reproducibility is the property that makes a benchmark auditable. Two labels
therefore ship side by side, and neither is "the recomputed one":

- `score` -- the legacy label. Consensus-filtered by the 2021 pipeline,
  comparable with the published paper, not reproducible.
- `score_mean` -- plain mean of the raw layer. Reproducible in one line,
  equal to the mean of `ratings/distributions.parquet`, not comparable with
  the paper.

They differ systematically, because one is filtered and the other is not.
That is expected, not an error, and `score_delta` measures it.

Nothing here modifies `score`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Columns in the legacy score workbooks that are not raters.
NON_RATER_COLUMNS = frozenset({"Unnamed: 0", "image", "mean", "path"})

#: |score - score_mean| above which the two labels are flagged as materially
#: different. Half a point on a 10-point scale: large enough to change how an
#: image ranks, and well beyond the routine gap (mean absolute delta is 0.20).
#:
#: This is deliberately NOT an anomaly flag. `score` is consensus-filtered and
#: `score_mean` is not, so a nonzero delta is the expected, systematic
#: consequence of that difference -- not evidence that either label is wrong.
#: It marks images where the choice between the two actually matters.
DIVERGENCE_THRESHOLD = 0.5

#: Columns this module adds to a canonical ratings table. Never includes
#: `score` -- the canonical label is passed through, never derived.
PROVENANCE_COLUMNS = (
    "n_ratings",
    "score_std",
    "score_mean",
    "score_delta",
    "diverges_from_score_mean",
)


@dataclass(frozen=True)
class LabelSupport:
    """Rater support behind one image's canonical label."""

    filename: str
    n_ratings: int
    score_mean: float
    score_std: float | None


def rater_columns(scores_df: pd.DataFrame) -> list[str]:
    """Return the rater-ID columns of a wide-format legacy score workbook."""
    return [column for column in scores_df.columns if column not in NON_RATER_COLUMNS]


def compute_label_support(
    scores_df: pd.DataFrame, image_column: str = "image"
) -> list[LabelSupport]:
    """Summarize the rater support behind each image in a score workbook.

    `scores_df` is wide-format: one row per image, one column per rater, with
    a blank cell where a rater did not score that image.

    A handful of images appear on more than one row of the legacy workbooks.
    Their ratings are pooled across those rows rather than dropped, so an
    image's support reflects every rating recorded for it.
    """
    raters = rater_columns(scores_df)
    if not raters:
        raise ValueError("No rater columns found; is this a wide-format workbook?")

    numeric = scores_df[raters].apply(pd.to_numeric, errors="coerce")
    numeric.insert(0, image_column, scores_df[image_column].astype(str))

    support = []
    for filename, group in numeric.groupby(image_column, sort=True):
        # Explicit dropna: `stack()` no longer drops nulls, and a blank cell
        # means "this rater did not score this image", not a rating of NaN.
        values = group[raters].stack().dropna()
        if values.empty:
            continue
        support.append(
            LabelSupport(
                filename=str(filename),
                n_ratings=int(values.size),
                score_mean=float(values.mean()),
                # A single rating has no spread; report it as missing rather
                # than as 0.0, which would read as perfect agreement.
                score_std=float(values.std()) if values.size > 1 else None,
            )
        )
    return support


def support_frame(support: list[LabelSupport]) -> pd.DataFrame:
    """Render `compute_label_support` output as a DataFrame keyed by filename."""
    return pd.DataFrame(
        [
            {
                "legacy_filename": item.filename,
                "n_ratings": item.n_ratings,
                "score_mean": item.score_mean,
                "score_std": item.score_std,
            }
            for item in support
        ]
    )


def attach_label_provenance(
    ratings_df: pd.DataFrame,
    support_df: pd.DataFrame,
    filename_by_image_id: dict[str, str],
    threshold: float = DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """Add rater-support columns to a canonical ratings table.

    `score` is passed through untouched -- it stays the canonical label. The
    added columns describe it; they never replace it. Images with no row in
    the surviving rater matrix keep null support rather than being dropped.
    """
    # Idempotent: re-running over an already-enriched table refreshes the
    # provenance columns instead of colliding with them into `_x`/`_y` pairs.
    out = ratings_df.drop(columns=list(PROVENANCE_COLUMNS), errors="ignore").copy()
    out["legacy_filename"] = out["image_id"].map(filename_by_image_id)
    out = out.merge(support_df, on="legacy_filename", how="left")
    out["score_delta"] = out["score"] - out["score_mean"]
    out["diverges_from_score_mean"] = out["score_delta"].abs() > threshold
    # Nullable integer, so a rating count reads as 24 rather than 24.0 while
    # still expressing "no surviving rater matrix row" for unmatched images.
    out["n_ratings"] = out["n_ratings"].astype("Int64")
    return out.drop(columns=["legacy_filename"])
