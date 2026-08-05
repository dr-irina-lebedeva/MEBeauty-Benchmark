"""Which raters' judgements carry information, and which images can be labelled.

Two separate screens, and keeping them separate matters:

**Raters** are screened on whether their ratings can carry information at all.
One rule: a rater who used fewer than 3 distinct values is not discriminating
between faces, whatever their volume. Straight-lining (one value) and the near
case (alternating two) both fall under it.

**Images** are screened on support. An image needs at least 10 ratings before
it gets a label; below that the mean is too noisy to be a benchmark target.

Note what is *not* a rater rule any more: rating volume. A rater who scored
three faces is kept. Their three judgements are real, and dropping them
throws away information for no reason once the image-level support rule
guarantees every label rests on 10+ ratings. What light raters do break is
plain z-scoring -- you cannot estimate a standard deviation from one rating --
which is exactly why `legacy/normalization.py` shrinks each rater's statistics
toward the global ones instead.

The tempting third rule is to drop raters whose scores correlate poorly with
the consensus, and it is rejected: it defines a good rater as one who agrees
with the majority, inflates apparent inter-rater reliability, and on a
*multi-ethnic beauty* dataset deletes the minority aesthetic variation the
dataset exists to study. Those statistics still ship in
`ratings/by_rater/rater_quality.parquet`, so anyone who wants that filter can
apply it in one line -- as their choice, not baked into the labels.

**Nothing is deleted.** `ratings_by_rater.parquet` keeps every rating and
gains a `rater_valid` column, so the filter is auditable, reversible, and the
unfiltered mean stays recomputable from shipped data.
"""

from __future__ import annotations

import pandas as pd

#: A rater using fewer than this many distinct values is not discriminating
#: between faces. Covers exact straight-lining (one value) and the near case
#: (alternating two), which a zero-variance test misses. Applied at every
#: volume: a rater who gave "7, 7, 7" is as uninformative as one who gave it
#: two hundred times.
MIN_DISTINCT_SCORES = 3

#: Ratings an image needs before it can carry a label.
#:
#: **8, not 10, and the reason is measured.** The rating counts have a clean
#: gap: 435 images sit at exactly 9 valid ratings and 8 more at 8, while
#: nothing at all falls between 1 and 7. A threshold of 10 therefore does not
#: separate well-rated images from poorly-rated ones -- it slices through the
#: middle of a single collection batch.
#:
#: What it slices off is not random. The images with 8-9 ratings are 27.3%
#: asian, 17.8% indian and 17.2% black, against 10.9 / 10.4 / 10.5% among
#: images with 10 or more. Excluding them removes 443 images (18%) and pushes
#: the ethnic imbalance from 3.35x to 4.21x -- it makes a dataset built to be
#: multi-ethnic measurably less so, which is the opposite of its purpose.
#:
#: The reliability cost is real but small and, crucially, *disclosed*: median
#: standard error 0.71 for the 8-9 band against 0.41 for 10-29. `rating_count`,
#: `score_std` and `label_status` ship with every image, so a consumer who
#: wants only the tightly-rated subset can have it in one line -- while a
#: consumer who wants the balanced dataset is no longer forced to take an
#: unbalanced one.
MIN_RATINGS_PER_IMAGE = 8


def rater_validity(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-rater validity, with the reason attached.

    Returns one row per rater: `rater_id`, `n_ratings`, `n_distinct_scores`,
    `rater_valid`, and `invalid_reason` (empty when valid). The reason ships
    so a consumer can see *why* a rater was excluded rather than having to
    re-derive it.
    """
    grouped = ratings.groupby("rater_id")["score"]
    frame = pd.DataFrame(
        {
            "n_ratings": grouped.size(),
            "n_distinct_scores": grouped.nunique(),
        }
    ).reset_index()

    no_variation = frame["n_distinct_scores"] < MIN_DISTINCT_SCORES

    frame["rater_valid"] = ~no_variation
    frame["invalid_reason"] = ""
    frame.loc[no_variation, "invalid_reason"] = (
        f"fewer than {MIN_DISTINCT_SCORES} distinct scores"
    )
    return frame


def attach_validity(ratings: pd.DataFrame) -> pd.DataFrame:
    """Add `rater_valid` to a per-rater ratings table without dropping rows."""
    validity = rater_validity(ratings)[["rater_id", "rater_valid"]]
    return ratings.merge(validity, on="rater_id", how="left")


def labelled_image_ids(
    ratings: pd.DataFrame, min_ratings: int = MIN_RATINGS_PER_IMAGE
) -> set[str]:
    """Images with enough ratings *from valid raters* to carry a label.

    Order matters: the rater screen runs first, so an image kept alive only by
    straight-lining raters does not sneak past the support threshold.
    """
    valid = ratings[ratings["rater_valid"]] if "rater_valid" in ratings else ratings
    counts = valid.groupby("image_id")["score"].size()
    return set(counts.index[counts >= min_ratings])
