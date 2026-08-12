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
from torch.utils.data import DataLoader

from ..data import Protocol, Split
from ..registry import register
from ..reproducibility import set_seed
from .foundation import DINOv2Partial
from .training import SCORE_BINS, FaceDataset, TrainConfig, _DeepMethod

#: Relative weight of the regression term against the distribution likelihood.
DEFAULT_REGRESSION_WEIGHT = 1.0


@register(
    "rw-ldl",
    era="deep",
    reference="This work (reliability-weighted LDL)",
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
        counts, error = train.n_ratings, train.standard_error
        if error is None:
            raise ValueError("rw-ldl needs per-image rating counts")
        self._mean_n = float(np.mean(counts))
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
    reference="This work (ablation: no precision weighting)",
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
    reference="This work (ablation: no multinomial likelihood)",
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


@register(
    "rater-dinov2",
    era="foundation",
    reference="proposed in this benchmark",
    requires=("ratings",),
    trainable=True,
    notes="proposed: rater effects on a foundation backbone",
)
class RaterAwareFoundation(DINOv2Partial):
    """Predict the *rater-marginal*, by learning the model the label is defined by.

    `beauty_score` is not an arbitrary summary. It is the fitted `quality(i)`
    in `rating(r, i) = quality(i) + offset(r)`. Every other method in this
    benchmark ignores that: they regress the summary statistic, discarding
    55,542 individual ratings to fit 1,962 means.

    This one fits the generative model instead. The network predicts
    `quality(i)`; a learned scalar per rater supplies `offset(r)`; and the loss
    is taken against every individual rating rather than against their mean.
    At test time only `quality(i)` is used, which is exactly the published
    label -- so the training objective and the evaluation target are the same
    quantity, not a proxy for it.

    **Three things fall out for free.**

    *No extra compute.* One forward pass per image, as before. The loss sums
    over that image's 8-92 ratings, so supervision multiplies ~28x while the
    backbone cost is unchanged.

    *Reliability weighting, unparameterised.* An image rated 92 times
    contributes 92 loss terms and one rated 8 times contributes 8. `rw-ldl`
    buys the same effect with an explicit inverse-variance scheme and a tuned
    floor; here it is a consequence of counting observations.

    *Rater bias cannot leak into the image score.* A harsh rater's ratings are
    explained by their own offset rather than by the faces they happened to
    see -- the same correction the label itself was built with.

    **Offsets are re-centred every step, weighted by rating count.** Without an
    anchor the model is degenerate: any constant can move between `quality` and
    `offset`. Count-weighting matches how `beauty_score` was defined, and keeps
    the predictions on the scale the ratings were given on.

    Not novel in general machine learning -- annotator embeddings are
    established in subjective NLP, speech MOS and medical segmentation. Novel
    in facial beauty prediction, where the per-rater data needed for it has
    only just been published.
    """

    name = "rater-dinov2"
    predicts_distribution = False

    def __init__(
        self,
        config: TrainConfig | None = None,
        seed: int = 0,
        consensus_weight: float = 0.2,
        offset_decay: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(config, seed, **kwargs)
        #: Weight on the auxiliary term that regresses the consensus directly.
        #: The rater term alone is the principled objective, but the consensus
        #: is what is scored, and a small direct signal stabilises early epochs.
        self.consensus_weight = consensus_weight
        self.offset_decay = offset_decay

    def _prepare_ratings(self, split: Split) -> None:
        """Ragged per-image ratings -> padded tensors indexed by row."""
        if (
            split.rating_value is None
            or split.rating_rater is None
            or split.rating_image is None
        ):
            raise ValueError(
                "rater-dinov2 needs individual ratings; run it against the "
                "`personalized_fbp` config"
            )
        # Bound once, so the rest of the method works with plain arrays.
        values, who, which = split.rating_value, split.rating_rater, split.rating_image
        raters, rater_index = np.unique(who, return_inverse=True)
        n_images = len(split)
        counts = np.bincount(which, minlength=n_images)
        width = int(counts.max())

        pad_rater = np.zeros((n_images, width), dtype=np.int64)
        pad_value = np.zeros((n_images, width), dtype=np.float32)
        pad_mask = np.zeros((n_images, width), dtype=np.float32)
        slot = np.zeros(n_images, dtype=np.int64)
        for image, rater, value in zip(which, rater_index, values, strict=True):
            position = slot[image]
            pad_rater[image, position] = rater
            pad_value[image, position] = value
            pad_mask[image, position] = 1.0
            slot[image] += 1

        self._pad_rater = torch.from_numpy(pad_rater).to(self.device)
        self._pad_value = torch.from_numpy(pad_value).to(self.device)
        self._pad_mask = torch.from_numpy(pad_mask).to(self.device)
        # Count-weighted centring, as used to define the label itself.
        self._rater_counts = torch.from_numpy(
            np.bincount(rater_index, minlength=len(raters)).astype(np.float32)
        ).to(self.device)
        self.offsets = nn.Embedding(len(raters), 1).to(self.device)
        nn.init.zeros_(self.offsets.weight)
        print(
            f"  rater-dinov2: {len(values):,} ratings from "
            f"{len(raters)} raters over {n_images} images "
            f"({len(split.rating_value) / n_images:.1f} per image)",
            flush=True,
        )

    @torch.no_grad()
    def _recentre(self) -> None:
        """Anchor the offsets so quality carries the level, not the raters."""
        weights = self.offsets.weight
        centre = (weights.squeeze(-1) * self._rater_counts).sum() / (
            self._rater_counts.sum()
        )
        weights -= centre

    def _rater_loss(self, quality: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        rater = self._pad_rater[rows]
        value = self._pad_value[rows]
        mask = self._pad_mask[rows]
        predicted = quality.unsqueeze(1) + self.offsets(rater).squeeze(-1)
        per_rating = F.smooth_l1_loss(predicted, value, reduction="none", beta=1.0)
        return (per_rating * mask).sum() / mask.sum().clamp(min=1.0)

    def fit(self, protocol: Protocol) -> None:
        set_seed(self.seed)
        self._prepare_ratings(protocol.train)
        super().fit(protocol)

    def _modules(self) -> dict[str, nn.Module]:
        modules = super()._modules()
        modules["offsets"] = self.offsets
        return modules

    def _fit_loop(self, protocol: Protocol) -> None:
        """As the parent's loop, but supervised by individual ratings.

        `attribute_codes` carries the row index rather than a demographic code,
        which is what lets the loss find each image's ratings without the
        DataLoader having to collate a ragged structure.
        """
        train_loader = DataLoader(
            FaceDataset(
                protocol.train,
                self.config.image_size,
                train=True,
                attribute_codes=np.arange(len(protocol.train)),
                augmentation=self.config.augmentation,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        modules = self._modules()
        for module in modules.values():
            module.to(self.device)
        parameters = [
            p for m in modules.values() for p in m.parameters() if p.requires_grad
        ]
        optimiser = self._make_optimiser(parameters)
        schedule = self._make_schedule(optimiser, len(train_loader))

        best_mae, best_state, waited, epoch = float("inf"), None, 0, 0
        for epoch in range(self.config.epochs):
            for module in modules.values():
                module.train()
            for images, labels, _distributions, rows in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                rows = rows.to(self.device)
                optimiser.zero_grad()
                quality = self._forward(images).squeeze(-1)
                loss = self._rater_loss(quality, rows)
                if self.consensus_weight:
                    loss = loss + self.consensus_weight * F.l1_loss(quality, labels)
                if self.offset_decay:
                    loss = loss + self.offset_decay * self.offsets.weight.pow(2).mean()
                loss.backward()
                optimiser.step()
                self._recentre()
            schedule.step()

            mae = float(
                np.abs(self.predict(protocol.val).scores - protocol.val.labels).mean()
            )
            if mae < best_mae - 1e-4:
                best_mae, waited, best_state = mae, 0, self._snapshot()
            else:
                waited += 1
                if (
                    waited >= self.config.patience
                    and epoch + 1 >= self.config.min_epochs
                ):
                    break

        self.epochs_trained = epoch + 1
        self.best_val_mae = best_mae
        if best_state is not None:
            self._restore(best_state)
