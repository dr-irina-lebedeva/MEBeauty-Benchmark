"""Per-image rating distributions (soft labels) for label distribution learning.

The canonical `score` is a single number produced by the 2021 pipeline, which
dropped raters whose scores correlated below 0.10 with the pooled average
(`docs/DATASET_AUDIT.md`, Finding 20). Collapsing subjective judgments to one
consensus-filtered mean discards the disagreement itself -- and for a
multi-ethnic beauty dataset whose stated goals include studying rater
variation and subgroup bias, that disagreement is the object of study, not
noise around it.

This module builds the distribution instead: for each image, how many raters
chose each point on the 1-10 scale. Three properties are deliberate:

- **Unfiltered.** Every rater in `ratings_by_rater.parquet` contributes; none
  is excluded or down-weighted. This is the same policy as Finding 14 and the
  opposite of the canonical label's inherited 2021 filtering, so the two are
  genuinely different views of the same ratings rather than restatements.
- **Lossless.** Raw `counts` ship next to normalized `probabilities`, so a
  consumer can re-derive any statistic, re-weight, or re-bin without needing
  the per-rater table.
- **One task.** Only `generic` attractiveness ships, so there is exactly one
  distribution per image. The legacy `date` task was a different question
  (Finding 16) and is no longer released.

Nothing here replaces the canonical label. It is an additional view, keyed by
the same `image_id`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

#: The rating scale. Every rating in the legacy collection is an integer in
#: this range -- verified across all 123,177 rows, zero non-integer values.
SCORE_BINS: tuple[int, ...] = tuple(range(1, 11))

#: Maximum possible entropy, in bits, for `SCORE_BINS` -- log2(10). A label
#: at this value had its ratings spread perfectly evenly across the scale.
#:
#: WARNING: only reachable with at least 10 ratings. An image rated 9 times
#: can occupy at most 9 bins and so is capped at log2(9) = 3.170, whatever
#: the raters actually thought. See `entropy_bits` on `RatingDistribution`.
MAX_ENTROPY_BITS = math.log2(len(SCORE_BINS))


@dataclass(frozen=True)
class RatingDistribution:
    """One image's ratings for one task, as a distribution over `SCORE_BINS`.

    `entropy_bits` is **not comparable across images with different
    `n_ratings`**, and must not be used to rank images by how contested they
    are. Plug-in entropy is downward-biased at small samples, and the bins
    impose a hard ceiling on top of that: measured on this dataset it
    correlates +0.63 with `n_ratings` (images with 9-11 ratings average 2.27
    bits, those with 41+ average 2.91, a gap that is largely sample size
    rather than consensus). A Miller-Madow correction only reduces this to
    +0.50, because no estimator can recover bins the sample could never fill.

    Use `std` for cross-image comparison: it correlates just +0.11 with
    `n_ratings` and is flat across rating-count bands. `entropy_bits` remains
    an honest description of the *observed* distribution, which is what label
    distribution learning consumes, so it ships -- with this caveat attached.
    """

    counts: tuple[int, ...]
    mean: float
    median: float
    std: float | None
    entropy_bits: float

    @property
    def n_ratings(self) -> int:
        return sum(self.counts)

    @property
    def probabilities(self) -> tuple[float, ...]:
        total = self.n_ratings
        return tuple(count / total for count in self.counts)


def rating_distribution(scores: Sequence[float]) -> RatingDistribution:
    """Build the distribution for one image's ratings.

    Raises on anything outside `SCORE_BINS` rather than dropping it: a score
    of 0, 11, or 7.5 means the upstream data is not what this module assumes,
    and silently discarding it would corrupt the distribution invisibly.
    """
    if not scores:
        raise ValueError("Cannot build a distribution from zero ratings")

    counter: Counter[int] = Counter()
    for score in scores:
        if score != int(score) or int(score) not in SCORE_BINS:
            raise ValueError(
                f"Score {score!r} is not an integer in {SCORE_BINS[0]}-{SCORE_BINS[-1]}"
            )
        counter[int(score)] += 1

    counts = tuple(counter[bin_] for bin_ in SCORE_BINS)
    series = pd.Series(list(scores), dtype="float64")
    total = len(scores)

    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts if count
    )
    return RatingDistribution(
        counts=counts,
        mean=float(series.mean()),
        median=float(series.median()),
        # Sample std, matching `score_std` in label_provenance. Undefined for
        # a single rating -- reported as missing, not as 0.0, which would read
        # as agreement between raters that do not exist.
        std=float(series.std()) if total > 1 else None,
        entropy_bits=entropy,
    )


def build_distributions(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Build one distribution per image from a per-rater table.

    Expects the columns of `ratings/by_rater/ratings_by_rater.parquet`:
    `image_id` and `score`. Only the `generic` task ships, so there is one
    distribution per image and no task column to key on.
    """
    missing = {"image_id", "score"} - set(ratings_df.columns)
    if missing:
        raise ValueError(f"Per-rater table is missing columns: {sorted(missing)}")

    rows = []
    for image_id, group in ratings_df.groupby("image_id", sort=True):
        distribution = rating_distribution(group["score"].tolist())
        rows.append(
            {
                "image_id": image_id,
                "n_ratings": distribution.n_ratings,
                "counts": list(distribution.counts),
                "probabilities": list(distribution.probabilities),
                "mean": distribution.mean,
                "median": distribution.median,
                "std": distribution.std,
                "entropy_bits": distribution.entropy_bits,
            }
        )
    return pd.DataFrame(rows)
