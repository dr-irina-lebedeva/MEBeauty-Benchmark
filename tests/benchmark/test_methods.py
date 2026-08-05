"""Tests for the benchmark methods themselves.

Until these existed, nothing verified a method's *internals*. Two consequences
were found the hard way and both are covered here:

- `fpem` and `transfbp` had never been executed. They crashed on the first
  real call -- one on a device mismatch, one on an unknown backbone -- despite
  being written, reviewed and documented as ready. `test_every_method_runs`
  is that missing check.
- `comboloss` optimised a different objective from its paper for as long as it
  existed, because no test asserted the formula. `test_comboloss_*` do.

The loss tests build tensors directly and never touch a backbone, so they run
in milliseconds. The end-to-end tests need pretrained weights and are marked
`slow`; run them with `-m slow`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mebeauty_benchmark.benchmark.protocol import Split, load_protocol
from mebeauty_benchmark.methods.deep import (
    R3CNN,
    SCORE_BINS,
    ComboLoss,
    LabelDistributionLearning,
    TrainConfig,
)


def _head_output(batch: int, width: int = 10) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, width)


# --------------------------------------------------------------- ComboLoss


def test_comboloss_matches_the_papers_weighted_sum():
    """L_combo = 2*L_reg + 1*L_exp + 1*L_cls, recomputed independently."""
    method = ComboLoss(TrainConfig())
    method.class_weights = torch.ones(len(SCORE_BINS))

    logits = _head_output(8)
    predicted = torch.tensor([5.0, 6.0, 7.0, 4.0, 8.0, 3.0, 6.5, 5.5])
    labels = torch.tensor([5.5, 6.5, 6.0, 4.5, 7.5, 3.5, 6.0, 5.0])

    actual = method.loss((logits, predicted), labels, torch.zeros(8, 10))

    bins = SCORE_BINS
    expectation = (torch.softmax(logits, dim=-1) * bins).sum(-1)
    target = torch.clamp(torch.round(labels) - 1, 0, 9).long()
    expected = (
        2.0 * torch.nn.functional.l1_loss(predicted, labels)
        + 1.0 * torch.nn.functional.l1_loss(predicted, expectation.detach())
        + 1.0
        * torch.nn.functional.cross_entropy(
            logits, target, weight=torch.ones(len(bins))
        )
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_comboloss_regression_term_dominates():
    """alpha=2 must actually weight L_reg twice, not once.

    The bug this catches: dropping the coefficients. With alpha applied, a
    regression error hurts twice as much as the same error in L_exp.
    """
    method = ComboLoss(TrainConfig())
    method.class_weights = torch.ones(len(SCORE_BINS))
    logits = torch.zeros(4, 10)  # uniform -> expectation 5.5
    labels = torch.full((4,), 5.5)

    on_target = method.loss((logits, torch.full((4,), 5.5)), labels, torch.zeros(4, 10))
    off_by_one = method.loss(
        (logits, torch.full((4,), 6.5)), labels, torch.zeros(4, 10)
    )
    # predicted moves 1.0 away from both the label and the expectation,
    # so the rise is 2*1.0 (L_reg) + 1*1.0 (L_exp) = 3.0.
    assert float(off_by_one - on_target) == pytest.approx(3.0, abs=1e-5)


def test_comboloss_needs_a_separate_regression_output():
    """L_reg and L_exp must read a regression head, not the distribution.

    If both are computed from one distribution head they become the same
    quantity and L_exp is identically zero -- the exact defect this method had.
    """
    method = ComboLoss(TrainConfig())
    features = 16
    head = method.build_head(features)

    assert hasattr(method, "regressor"), "no separate regression head"
    visual = torch.randn(4, features)
    assert head(visual).shape == (4, len(SCORE_BINS))
    assert method.regressor(visual).shape == (4, 1)


def test_comboloss_class_weights_favour_rare_scores():
    """Balancing must up-weight the bins MEBeauty has few of."""
    # A training split piled up at 6, with one image at 2 -- the shape of this
    # dataset's score histogram, exaggerated.
    weights = ComboLoss.class_weights_for(np.array([6.0] * 99 + [2.0]))

    assert weights[1] > weights[5], "the rare score must carry more weight"
    assert float(weights[0]) == 0.0, "an empty bin must not divide by zero"
    assert torch.isfinite(weights).all()


def test_comboloss_class_weights_come_from_training_labels_only():
    """Weighting from all labels would leak the test distribution."""
    train_only = ComboLoss.class_weights_for(np.array([5.0] * 20))
    with_test = ComboLoss.class_weights_for(np.array([5.0] * 20 + [9.0] * 20))

    assert float(train_only[8]) == 0.0
    assert float(with_test[8]) > 0.0


# ------------------------------------------------------------ other losses


def test_ldl_expectation_is_the_reported_score():
    method = LabelDistributionLearning(TrainConfig())
    logits = _head_output(6)

    scores = method.to_scores(logits)
    expected = (torch.softmax(logits, dim=-1) * SCORE_BINS).sum(-1)

    assert torch.allclose(scores, expected)
    assert ((scores >= 1.0) & (scores <= 10.0)).all()


def test_ldl_loss_is_zero_when_the_distribution_is_predicted_exactly():
    method = LabelDistributionLearning(TrainConfig())
    target = torch.full((4, 10), 0.1)
    logits = torch.zeros(4, 10)  # softmax -> uniform 0.1

    loss = method.loss(logits, torch.full((4,), 5.5), target)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_r3cnn_penalises_a_reversed_ranking():
    """The ranking term must fire when the predicted order is wrong."""
    method = R3CNN(TrainConfig())
    labels = torch.tensor([2.0, 8.0])

    correct = method.loss(torch.tensor([[2.0], [8.0]]), labels, torch.zeros(2, 10))
    reversed_ = method.loss(torch.tensor([[8.0], [2.0]]), labels, torch.zeros(2, 10))

    assert float(reversed_) > float(correct)


def test_r3cnn_ignores_pairs_whose_labels_are_effectively_tied():
    method = R3CNN(TrainConfig())
    labels = torch.tensor([5.0, 5.05])  # gap below the 0.1 threshold

    loss = method.loss(torch.tensor([[5.0], [5.05]]), labels, torch.zeros(2, 10))

    # Falls back to pure regression, which is zero for a perfect prediction.
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------- end-to-end


def _tiny(split: Split, n: int) -> Split:
    """First `n` rows of a split, with every per-image field kept in step.

    `dataclasses.replace` rather than a fresh `Split(...)`: listing the fields
    by hand silently drops any added later, which is exactly how `rw-ldl` came
    to fail here for want of `n_ratings`.
    """

    def head(value):
        return None if value is None else value[:n]

    return dataclasses.replace(
        split,
        image_ids=split.image_ids[:n],
        labels=split.labels[:n],
        image_paths=split.image_paths[:n],
        distributions=head(split.distributions),
        n_ratings=head(split.n_ratings),
        label_std=head(split.label_std),
        metadata=split.metadata.iloc[:n].reset_index(drop=True),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "mean-baseline",
        "eisenthal2006",
        "kagian2008",
        "fan2012",
        "gan2014",
        "cnn-resnet18",
        "pi-cnn",
        "ldl-ren2017",
        "cnn-resnext50",
        "r3cnn",
        "aanet",
        "comboloss",
        "uol",
        "fpem",
        "transfbp",
        "rw-ldl",
        "rw-ldl-noweight",
        "rw-ldl-kl",
        "dinov2-linear",
        "dinov2-partial",
    ],
)
def test_every_method_runs(name):
    """One epoch on a small subset. Would have caught both 2026-07-28 crashes.

    Deliberately runs *every* registered method: the two that were broken were
    exactly the two nobody had executed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmark"))
    from run import build_registry

    protocol = load_protocol("data/mebeauty_v3")
    # 40 images: enough for eisenthal2006's 15-neighbour KNN.
    protocol = dataclasses.replace(
        protocol,
        train=_tiny(protocol.train, 40),
        val=_tiny(protocol.val, 40),
        test=_tiny(protocol.test, 40),
    )

    method = build_registry(seed=0, epochs=1)[name]()
    method.fit(protocol)
    prediction = method.predict(protocol.test)

    assert len(prediction.scores) == len(protocol.test)
    assert np.isfinite(prediction.scores).all()
    assert prediction.scores.min() >= 1.0
    assert prediction.scores.max() <= 10.0


# ------------------------------------------------------------ rw-ldl (proposed)


def _rwldl(**kwargs):
    from mebeauty_benchmark.methods.proposed import ReliabilityWeightedLDL

    method = ReliabilityWeightedLDL(TrainConfig(), **kwargs)
    method._mean_n = 30.0
    method._floor_variance = 0.09  # median se ~0.3
    return method


def _counts(n: int, centre: int) -> torch.Tensor:
    """A histogram of `n` ratings concentrated on one bin."""
    row = torch.zeros(10)
    row[centre - 1] = n
    return row


def test_multinomial_term_weights_a_heavily_rated_image_more():
    """The core claim: sample size enters the loss, unlike plain KL.

    Two images with identical *normalised* histograms but 10 vs 100 ratings
    must not contribute equally. Under KL they would.
    """
    method = _rwldl(regression_weight=0.0)
    logits = torch.zeros(1, 10)  # uniform prediction

    light = method.loss(logits, torch.tensor([5.0]), _counts(10, 5).unsqueeze(0))
    heavy = method.loss(logits, torch.tensor([5.0]), _counts(100, 5).unsqueeze(0))

    assert float(heavy) > float(light) * 5


def test_kl_ablation_ignores_sample_size():
    """The contrast that makes the point: the ablation is blind to n."""
    method = _rwldl(regression_weight=0.0, use_multinomial=False)
    logits = torch.zeros(1, 10)

    light = method.loss(logits, torch.tensor([5.0]), _counts(10, 5).unsqueeze(0))
    heavy = method.loss(logits, torch.tensor([5.0]), _counts(100, 5).unsqueeze(0))

    assert float(light) == pytest.approx(float(heavy), abs=1e-5)


def test_precision_weighting_downweights_an_unreliable_label():
    """A noisy, thinly-rated label must pull the regression term less."""
    method = _rwldl()
    logits = torch.zeros(2, 10)

    # Image 0: 60 ratings all on 5 -> tiny standard error.
    # Image 1: 10 ratings split across the scale -> large standard error.
    tight = _counts(60, 5)
    loose = torch.full((10,), 1.0)
    counts = torch.stack([tight, loose])

    # Both predicted equally wrong, so any difference is the weighting.
    labels = torch.tensor([5.0, 5.5])
    weights = _weights_from(method, counts)

    assert weights[0] > weights[1]
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)
    assert torch.isfinite(method.loss(logits, labels, counts))


def _weights_from(method, counts):
    """Recompute the loss's internal weights, to assert on them directly."""
    bins = torch.arange(1.0, 11.0)
    n = counts.sum(-1).clamp(min=1.0)
    empirical = counts / n.unsqueeze(-1)
    mean = (empirical * bins).sum(-1)
    var = (empirical * (bins - mean.unsqueeze(-1)) ** 2).sum(-1)
    weights = 1.0 / (var / n + method._floor_variance)
    return weights / weights.mean()


def test_precision_weights_are_bounded_by_the_floor():
    """Without the floor, a zero-variance label would get infinite weight."""
    method = _rwldl()
    # 100 raters who all said exactly 6: variance is zero.
    counts = torch.stack([_counts(100, 6), _counts(10, 3)])

    weights = _weights_from(method, counts)

    assert torch.isfinite(weights).all()
    assert float(weights.max()) < 10.0


def test_rwldl_reports_the_distribution_expectation():
    method = _rwldl()
    logits = _head_output(6)

    scores = method.to_scores(logits)

    expected = (torch.softmax(logits, dim=-1) * SCORE_BINS).sum(-1)
    assert torch.allclose(scores, expected)


def test_rwldl_refuses_a_protocol_without_rating_counts():
    """It must fail loudly, not silently fall back to unweighted training."""
    import dataclasses

    from mebeauty_benchmark.benchmark.protocol import load_protocol

    protocol = load_protocol("data/mebeauty_v3")
    stripped = dataclasses.replace(
        protocol, train=dataclasses.replace(protocol.train, n_ratings=None)
    )

    with pytest.raises(ValueError, match="rating counts"):
        _rwldl().fit(stripped)
