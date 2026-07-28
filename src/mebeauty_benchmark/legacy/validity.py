"""Which raters' judgements carry information, and which cannot.

The canonical `score` is the mean over **valid** raters. Validity here is
decided entirely from a rater's own behaviour -- how much they rated, and
whether they varied at all. It never consults whether they agreed with anyone.

That boundary is the whole design. The tempting third rule is to drop raters
whose scores correlate poorly with the consensus, and it is rejected: it
defines a good rater as one who agrees with the majority, inflates apparent
inter-rater reliability, and on a *multi-ethnic beauty* dataset deletes the
minority aesthetic variation the dataset exists to study. It is also the most
expensive rule measured on this corpus (1.4% of ratings against 0.9% for both
rules below combined). Those statistics still ship in
`ratings/by_rater/rater_quality.parquet`, so anyone who wants that filter can
apply it in one line -- as their choice, not baked into the labels.

**Nothing is deleted.** `ratings_by_rater.parquet` keeps every rating and
gains a `rater_valid` column, so the filter is auditable, reversible, and the
unfiltered mean stays recomputable from shipped data.

Two rules, tested against more elaborate alternatives (extra straight-lining
variants, floor/ceiling means) that between them caught exactly one additional
rater. The simpler pair is preferred: a filter that must be stated in six
clauses is harder to trust and harder to reproduce.
"""

from __future__ import annotations

import pandas as pd

#: Below this, a rater has not produced enough judgements for their behaviour
#: to be characterised at all -- their mean, spread and consistency are noise.
MIN_RATINGS = 10

#: A rater using this few distinct values across `MIN_RATINGS`+ judgements is
#: not discriminating between faces. Covers exact straight-lining (one value)
#: and the near case (alternating two), which a zero-variance test misses.
MAX_DISTINCT_SCORES = 2


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

    too_few = frame["n_ratings"] < MIN_RATINGS
    no_variation = (frame["n_distinct_scores"] <= MAX_DISTINCT_SCORES) & (
        frame["n_ratings"] >= MIN_RATINGS
    )

    frame["rater_valid"] = ~(too_few | no_variation)
    frame["invalid_reason"] = ""
    frame.loc[too_few, "invalid_reason"] = f"fewer than {MIN_RATINGS} ratings"
    frame.loc[no_variation, "invalid_reason"] = (
        f"{MAX_DISTINCT_SCORES} or fewer distinct scores"
    )
    return frame


def attach_validity(ratings: pd.DataFrame) -> pd.DataFrame:
    """Add `rater_valid` to a per-rater ratings table without dropping rows."""
    validity = rater_validity(ratings)[["rater_id", "rater_valid"]]
    return ratings.merge(validity, on="rater_id", how="left")
