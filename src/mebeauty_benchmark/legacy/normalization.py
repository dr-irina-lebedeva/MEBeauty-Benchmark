"""Per-rater score normalisation before averaging.

Raters use the scale differently. Two people can rank a set of faces
identically while one averages 6.1 and the other 4.0 -- a gap this dataset
really contains. With an unbalanced design (nobody rated everything), a plain
mean lets that habit leak into the image's score: a face happens to be scored
by generous raters and comes out higher, for reasons that have nothing to do
with the face.

Normalising each rater to a common scale first removes that.

**The problem with plain z-scoring here.** The obvious method is
`(x - rater_mean) / rater_std`. It fails on this dataset because raters with
very few ratings are kept: a rater with one rating has no standard deviation
at all, and one with three has an estimate dominated by noise. Dividing by
that manufactures extreme values from nothing.

**The fix is shrinkage.** Each rater's mean and variance are pulled toward the
global values, weighted by how much evidence that rater actually provides:

    shrunk_mean = (n * rater_mean + k * global_mean) / (n + k)

A rater with 200 ratings keeps essentially their own statistics. A rater with
2 gets essentially the global ones, so they are barely adjusted rather than
wildly adjusted. `k` is the number of ratings at which a rater is trusted
halfway -- at the default of 5, a rater with 5 ratings sits midway between
their own mean and the global mean. See `DEFAULT_SHRINKAGE` for how that value
was measured, and for why the choice turns out not to matter much.

This is the standard empirical-Bayes treatment, and it is what makes "keep the
light raters" and "normalise per rater" compatible instead of contradictory.

**Scores return to the 1-10 scale.** Z-units are not interpretable and would
break every downstream consumer, so the normalised average is mapped back
through the global mean and standard deviation. The result is comparable to a
raw mean in magnitude while being free of rater-scale bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Ratings at which a rater is trusted halfway between their own statistics
#: and the global ones.
#:
#: Selected by held-out rating prediction, not by argument --
#: `scripts/data/validate_shrinkage.py`, 5 folds. Two things that measurement
#: showed, and the second matters more than the first:
#:
#: 1. k=5 minimises held-out RMSE. The previous value, 10, was chosen by
#:    analogy to the 10-rating image threshold and costs +0.002 RMSE.
#: 2. **The curve is almost flat.** Everything from k=0 to k=20 sits inside
#:    0.1% of the best. This is not a sensitive knob and should not be
#:    presented as a tuned one. What the measurement *does* rule out is heavy
#:    shrinkage: k>=50 is clearly worse (+0.03 and rising), because it drags
#:    every rater onto the global scale and erases the differences the
#:    normalisation exists to model.
#:
#: The comparison also understates shrinkage's value, and honestly so: to score
#: k=0 at all, raters whose z is undefined (a single rating, zero spread) must
#: be dropped from the evaluation -- which removes exactly the rows shrinkage
#: exists to handle. Shrinkage is in the pipeline because the policy keeps
#: light raters, not because this curve proves it.
DEFAULT_SHRINKAGE = 5.0

#: The rating scale. Normalised scores are clipped back into it.
SCORE_MIN, SCORE_MAX = 1.0, 10.0


@dataclass(frozen=True)
class RaterScale:
    """One rater's shrunken location and spread."""

    rater_id: str
    n_ratings: int
    raw_mean: float
    raw_std: float
    shrunk_mean: float
    shrunk_std: float


def rater_scales(
    ratings: pd.DataFrame, shrinkage: float = DEFAULT_SHRINKAGE
) -> pd.DataFrame:
    """Per-rater mean and standard deviation, shrunk toward the global values.

    Returns one row per rater with both the raw and shrunken statistics, so a
    consumer can see how much any given rater was adjusted rather than having
    to trust it.
    """
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")

    global_mean = float(ratings["score"].mean())
    global_var = float(ratings["score"].var(ddof=0))
    if global_var <= 0:
        raise ValueError("All ratings are identical; nothing to normalise")

    grouped = ratings.groupby("rater_id")["score"]
    frame = pd.DataFrame(
        {
            "n_ratings": grouped.size(),
            "raw_mean": grouped.mean(),
            # ddof=0 so a single rating gives 0 rather than NaN; the shrinkage
            # below is what stops that 0 becoming a division by zero.
            "raw_var": grouped.var(ddof=0).fillna(0.0),
        }
    ).reset_index()

    weight = frame["n_ratings"]
    frame["shrunk_mean"] = (weight * frame["raw_mean"] + shrinkage * global_mean) / (
        weight + shrinkage
    )
    frame["shrunk_std"] = np.sqrt(
        (weight * frame["raw_var"] + shrinkage * global_var) / (weight + shrinkage)
    )
    frame["raw_std"] = np.sqrt(frame["raw_var"])
    frame["shrinkage_weight"] = shrinkage / (weight + shrinkage)
    return frame.drop(columns=["raw_var"])


def normalise_ratings(
    ratings: pd.DataFrame, shrinkage: float = DEFAULT_SHRINKAGE
) -> pd.DataFrame:
    """Per-rating `z` (own rater's units) and `score_normalised` (back on 1-10).

    Clipping happens **per rating**, not after averaging. Two reasons, and the
    second is what makes the benchmark coherent:

    - A normalised value below 1 or above 10 is a single rating pushed off the
      scale by its rater's correction. Clamping it there is the same operation
      the scale itself performs; clamping an average instead lets an
      out-of-range rating drag the mean before anyone notices.
    - The image score is then the mean of values that all lie on 1-10, so the
      soft label built by binning those same values has *exactly* this mean.
      Clipping after averaging breaks that identity, and a label distribution
      whose expectation is not the label is a target no method can satisfy.
    """
    scales = rater_scales(ratings, shrinkage)
    merged = ratings.merge(
        scales[["rater_id", "shrunk_mean", "shrunk_std"]], on="rater_id", how="left"
    )
    merged["z"] = (merged["score"] - merged["shrunk_mean"]) / merged["shrunk_std"]

    global_mean = float(ratings["score"].mean())
    global_std = float(ratings["score"].std(ddof=0))
    merged["score_normalised"] = np.clip(
        global_mean + merged["z"] * global_std, SCORE_MIN, SCORE_MAX
    )
    return merged


def normalised_image_scores(
    ratings: pd.DataFrame,
    shrinkage: float = DEFAULT_SHRINKAGE,
    min_ratings: int = 1,
) -> pd.DataFrame:
    """Per-image score after removing each rater's scale habit.

    The average is taken in z-units, then mapped back onto 1-10 through the
    global mean and standard deviation so the result stays interpretable and
    directly comparable to a raw mean.

    `min_ratings` drops images with too little support to label; the caller
    decides the threshold, since it trades dataset size against label
    reliability.
    """
    normalised = normalise_ratings(ratings, shrinkage)

    grouped = normalised.groupby("image_id")
    frame = pd.DataFrame(
        {
            "n_ratings": grouped.size(),
            "raw_mean": grouped["score"].mean(),
            "z_mean": grouped["z"].mean(),
            "z_std": grouped["z"].std(ddof=0).fillna(0.0),
            # Mean of the per-rating clipped values -- see `normalise_ratings`
            # for why the clip is there and not here. This is what makes
            # `score` equal the mean of the normalised soft label exactly.
            "score_normalised": grouped["score_normalised"].mean(),
            "normalised_std": grouped["score_normalised"].std(ddof=0).fillna(0.0),
        }
    ).reset_index()
    return frame[frame["n_ratings"] >= min_ratings].reset_index(drop=True)
