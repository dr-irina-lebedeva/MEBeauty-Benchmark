"""What a benchmark method must provide.

Deliberately small. A method sees the training split, may see validation for
model selection, and returns one number per test image -- optionally also a
rating distribution. Everything else (metrics, splits, seeding, reporting) is
the harness's job, so no method can accidentally define its own evaluation.

**Methods must never look at test labels.** `Split.labels` exists on the test
split because the harness needs it for scoring, and nothing technically stops a
method reading it. `assert_no_test_leak` makes that an explicit, checked
contract rather than a convention: it runs each method twice with the test
labels shuffled and fails if predictions move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol as TypingProtocol
from typing import runtime_checkable

import numpy as np

from ..data import Protocol, Split
from ..registry import register


@dataclass(frozen=True)
class Prediction:
    """A method's output for one split."""

    scores: np.ndarray
    distributions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scores).all():
            raise ValueError("Predicted scores contain NaN or infinity")


@runtime_checkable
class Method(TypingProtocol):
    """The interface every benchmark entry implements."""

    name: str

    def fit(self, protocol: Protocol) -> None:
        """Train on `protocol.train`, tune on `protocol.val`."""

    def predict(self, split: Split) -> Prediction:
        """Predict for every image in `split`, in the order given."""


def assert_no_test_leak(method: Method, protocol: Protocol, seed: int = 0) -> None:
    """Fail if a method's predictions depend on the test labels.

    Shuffles the test labels and re-predicts. A method that only reads pixels,
    landmarks and training data cannot notice; one that peeks at the answers
    will produce different output. This is cheap insurance against the single
    mistake that would invalidate an entire results table.
    """
    rng = np.random.default_rng(seed)
    baseline = method.predict(protocol.test).scores

    shuffled = protocol.test.with_labels(rng.permutation(protocol.test.labels))
    after = method.predict(shuffled).scores

    if not np.allclose(baseline, after, equal_nan=True):
        raise AssertionError(
            f"{method.name}: predictions changed when test labels were shuffled -- "
            "the method is reading the labels it is meant to predict"
        )


@register(
    "mean-baseline",
    era="baseline",
    reference="--",
    notes="floor: predicts the training mean",
)
class MeanBaseline:
    """Predict the training mean for every image.

    The floor any real method must clear. Its Pearson correlation is 0 by
    construction, which is exactly why it belongs in the table: it shows what
    'no signal' scores on these metrics, and its MAE is often closer to a weak
    model's than readers expect.
    """

    name = "mean-baseline"

    def __init__(self, seed: int = 0) -> None:
        del seed  # deterministic; accepted so every method builds alike
        self._mean = 0.0

    def fit(self, protocol: Protocol) -> None:
        self._mean = float(protocol.train.labels.mean())

    def predict(self, split: Split) -> Prediction:
        return Prediction(scores=np.full(len(split), self._mean))
