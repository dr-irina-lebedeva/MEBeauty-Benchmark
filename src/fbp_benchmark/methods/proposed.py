"""RW-LDL: label distribution learning that knows how noisy each label is.

**The observation this is built on.** Every method in `benchmark-v1` treats
its training labels as equally trustworthy. On MEBeauty they are not: a label
rests on anywhere from 10 to 102 ratings, and the standard error of the mean
ranges over **4.3x** across the test split. An image rated 11 times with wide
disagreement is a far weaker target than one rated 102 times with narrow
agreement, and fitting both with the same loss spends capacity learning noise.

This is not a quirk of one dataset. It is what happens whenever subjective
labels are collected from an unbalanced rater pool -- which is most affective
computing datasets. It is invisible on SCUT-FBP5500 only because every image
there was rated by the same 60 people, so the variation does not arise.

**Two changes, both consequences of taking the sampling process seriously.**

*1. Fit the counts, not the histogram.* Standard LDL minimises KL divergence
between the predicted distribution and the *normalised* rating histogram. That
throws away the sample size: a histogram from 10 ratings and one from 100 are
treated identically, though the first is mostly sampling noise. If the model
predicts `p` and the raters are a sample from that population, the observed
counts `c` follow Multinomial(n, p), whose negative log-likelihood is

    -sum_k c_k log p_k

This is the *same expression* as cross-entropy against the histogram, except
weighted by `n` -- so reliability weighting is not an extra hyperparameter
bolted on, it falls out of writing down the correct likelihood. That is the
part worth arguing for: KL against a normalised histogram is the approximation,
and the multinomial likelihood is what it approximates.

*2. Weight the regression term by label precision.* The label's own standard
error is `se = std / sqrt(n)`. Inverse-variance weighting is the textbook
treatment for observations of differing precision (it is what meta-analysis
does), but it explodes when `se` is near zero, so the weights are regularised
with a floor at the median standard error:

    w = 1 / (se^2 + s0^2),  s0 = median(se)

then normalised to mean 1 so the loss keeps its scale and `--epochs` means the
same thing as for every other method.

**What is honestly novel here, and what is not.** Multinomial likelihood and
inverse-variance weighting are both old, standard statistics. Neither is
invented here. The contribution is the observation that facial-beauty
benchmarks systematically discard the label-reliability information they
already ship, plus the combination that exploits it -- and, importantly, a
measurement of whether it helps. If it does not beat `ldl-ren2017`, that is
the result and it will be reported as such.

**Deliberately not used at test time.** `n_ratings` and `std` are properties
of the *labels*, so using them to predict would be reading the answer. They
enter the loss only; prediction sees pixels alone, and the harness's leak
check confirms it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data import Protocol, Split
from ..registry import register
from .training import SCORE_BINS, TrainConfig, _DeepMethod

#: Relative weight of the regression term against the distribution likelihood.
DEFAULT_REGRESSION_WEIGHT = 1.0


@register(
    "rw-ldl",
    era="deep",
    reference="proposed in this benchmark",
    requires=("distributions",),
    trainable=True,
    notes="proposed",
)
class ReliabilityWeightedLDL(_DeepMethod):
    """Multinomial-likelihood LDL with precision-weighted regression.

    Predicts a distribution over the 1-10 scale; the reported score is its
    expectation, exactly as `ldl-ren2017` does, so the two differ *only* in
    the loss and the comparison isolates it.

    **How the counts reach the loss.** Rather than widen the dataset's return
    signature for every other method, `fit` hands the training and validation
    splits a distribution tensor holding raw counts instead of probabilities.
    Everything the loss needs follows from those: `n` is their sum, and the
    rating mean and variance come from the histogram itself. Prediction is
    untouched and never sees them.
    """

    name = "rw-ldl"
    predicts_distribution = True

    def __init__(
        self,
        config: TrainConfig | None = None,
        seed: int = 0,
        regression_weight: float = DEFAULT_REGRESSION_WEIGHT,
        use_precision_weights: bool = True,
        use_multinomial: bool = True,
    ) -> None:
        super().__init__(config, seed)
        self.regression_weight = regression_weight
        # Both switchable, so the ablations are constructor arguments rather
        # than copies of this class that can drift from it.
        self.use_precision_weights = use_precision_weights
        self.use_multinomial = use_multinomial
        self._mean_n = 1.0
        self._floor_variance = 1e-6

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, len(SCORE_BINS)))

    @staticmethod
    def _with_counts(split: Split) -> Split:
        """A copy whose `distributions` hold counts rather than probabilities."""
        if split.distributions is None or split.n_ratings is None:
            return split
        counts = split.distributions * np.asarray(split.n_ratings, float)[:, None]
        return dataclasses.replace(split, distributions=counts)

    def fit(self, protocol: Protocol) -> None:
        train = protocol.train
        if train.n_ratings is None:
            raise ValueError(
                "rw-ldl needs per-image rating counts; this protocol has none"
            )

        # Constants from the training split only. The floor is what stops a
        # single tightly-agreed image from dominating: unregularised inverse
        # variance spans ~20x on this dataset.
        self._mean_n = float(np.mean(train.n_ratings))
        error = train.standard_error
        self._floor_variance = float(np.median(error) ** 2)
        print(
            f"  rw-ldl: mean n={self._mean_n:.1f}, "
            f"se {error.min():.3f}-{error.max():.3f}, "
            f"weight floor {self._floor_variance:.4f}",
            flush=True,
        )

        super().fit(
            dataclasses.replace(
                protocol,
                train=self._with_counts(train),
                val=self._with_counts(protocol.val),
            )
        )

    def loss(self, output, labels, counts):
        log_p = F.log_softmax(output, dim=-1)
        probabilities = log_p.exp()
        bins = SCORE_BINS.to(output.device)
        expectation = (probabilities * bins).sum(-1)

        n = counts.sum(-1).clamp(min=1.0)

        if self.use_multinomial:
            # -sum_k c_k log p_k. Identical in form to cross-entropy against
            # the histogram, but weighted by n -- which is the whole point:
            # the reliability weighting is the likelihood, not an add-on.
            # Divided by the mean rating count so the loss magnitude does not
            # depend on how heavily this particular dataset was rated.
            distribution_term = (-(counts * log_p).sum(-1) / self._mean_n).mean()
        else:
            empirical = counts / n.unsqueeze(-1)
            distribution_term = F.kl_div(log_p, empirical, reduction="batchmean")

        if self.use_precision_weights:
            empirical = counts / n.unsqueeze(-1)
            rating_mean = (empirical * bins).sum(-1)
            rating_var = (empirical * (bins - rating_mean.unsqueeze(-1)) ** 2).sum(-1)
            # Standard error of the label, squared, then regularised.
            weights = 1.0 / (rating_var / n + self._floor_variance)
            weights = weights / weights.mean()
        else:
            weights = torch.ones_like(labels)

        regression = (weights * (expectation - labels).abs()).mean()
        return distribution_term + self.regression_weight * regression

    def to_scores(self, output) -> torch.Tensor:
        return (F.softmax(output, dim=-1) * SCORE_BINS.to(output.device)).sum(-1)


@register(
    "rw-ldl-noweight",
    era="deep",
    reference="ablation of rw-ldl",
    requires=("distributions",),
    trainable=True,
    notes="ablation: no reliability weighting",
)
class RWLDLNoWeighting(ReliabilityWeightedLDL):
    """Ablation: multinomial likelihood, every image weighted equally."""

    name = "rw-ldl-noweight"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed, use_precision_weights=False)


@register(
    "rw-ldl-kl",
    era="deep",
    reference="ablation of rw-ldl",
    requires=("distributions",),
    trainable=True,
    notes="ablation: KL instead of multinomial",
)
class RWLDLKLOnly(ReliabilityWeightedLDL):
    """Ablation: precision weighting, but plain KL instead of the likelihood.

    With `RWLDLNoWeighting` this isolates which of the two changes does the
    work -- or shows that neither does, which is equally worth reporting.
    """

    name = "rw-ldl-kl"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed, use_multinomial=False)
