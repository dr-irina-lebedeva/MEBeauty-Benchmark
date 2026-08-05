"""Run one method against one protocol and write a result.

The harness owns the split, the seed, the metrics and the leak check, so no
method can define its own evaluation. A method sees training data, may look at
validation for model selection, and returns one number per test image. That is
the whole contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Protocol
from .methods.base import Prediction, assert_no_test_leak
from .metrics import evaluate
from .registry import Entry, create, get
from .reproducibility import set_seed, write_result


@dataclass(frozen=True)
class Result:
    """One method's scores on one protocol."""

    method: str
    era: str
    metrics: dict[str, float]
    seconds: float
    predictions: np.ndarray
    protocol: dict

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "era": self.era,
            "metrics": self.metrics,
            "seconds": round(self.seconds, 1),
            **self.protocol,
        }


def check_requirements(entry: Entry, protocol: Protocol) -> None:
    """Fail early and readably when the dataset lacks what a method needs.

    Without this, a distribution method handed a dataset with no histogram
    trains on zeros and reports a plausible-looking bad score, which reads as
    'the method is weak' rather than 'the run was misconfigured'.
    """
    missing = []
    if "distributions" in entry.requires and protocol.train.distributions is None:
        missing.append(
            f"a rating distribution column "
            f"(spec.distribution_column={protocol.spec.distribution_column!r})"
        )
    if "attributes" in entry.requires:
        columns = set(protocol.train.metadata.columns)
        needed = {"gender", "ethnicity"} - columns
        if needed:
            missing.append(
                f"demographic columns {sorted(needed)} (spec.metadata_columns="
                f"{list(protocol.spec.metadata_columns)}); the minimal `fbp` "
                "config does not carry them -- use `fbp_extended`"
            )
    if "landmarks" in entry.requires and not protocol.train.landmarks:
        missing.append(
            f"facial landmarks (spec.landmark_column={protocol.spec.landmark_column!r})"
        )
    if missing:
        raise ValueError(
            f"{entry.name} needs {' and '.join(missing)}, which "
            f"{protocol.spec.repo_id}/{protocol.spec.config} does not provide."
        )


def run(
    name: str,
    protocol: Protocol,
    seed: int = 0,
    check_leak: bool = True,
    **overrides,
) -> Result:
    """Train `name` on the protocol's training split and score it on test."""
    entry = get(name)
    check_requirements(entry, protocol)

    set_seed(seed)
    method = create(name, seed=seed, **overrides)

    started = time.perf_counter()
    method.fit(protocol)
    prediction: Prediction = method.predict(protocol.test)
    seconds = time.perf_counter() - started

    if check_leak:
        # Cheap insurance against the one mistake that would invalidate the
        # entire table. Runs the method again with the test labels shuffled.
        assert_no_test_leak(method, protocol, seed=seed)

    metrics = evaluate(
        protocol.test.labels,
        prediction.scores,
        predicted_distributions=prediction.distributions,
        true_distributions=protocol.test.distributions,
    )
    return Result(
        method=name,
        era=entry.era,
        metrics=metrics,
        seconds=seconds,
        predictions=prediction.scores,
        protocol=protocol.describe(),
    )


def save(result: Result, directory: str | Path) -> Path:
    """Write the result JSON and its per-image predictions."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.method}.json"
    write_result(path, result.as_dict())
    np.savez_compressed(
        directory / f"{result.method}_predictions.npz",
        predictions=result.predictions,
    )
    return path
