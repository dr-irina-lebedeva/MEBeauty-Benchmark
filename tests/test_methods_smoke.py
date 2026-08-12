"""Every method fits and predicts. Slow, and the only test that catches a
method that cannot run at all.

Marked `slow` and `network`: these download the dataset and pretrained weights,
and train for a single epoch. `make check` skips them; `make test-slow` runs
them. "Skipped by default" must not become "never run" -- a method that throws
on the first batch passes every other test in this suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from fbp_benchmark import load_protocol, registry
from fbp_benchmark.runner import run

pytestmark = [pytest.mark.slow, pytest.mark.network]


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


@pytest.mark.parametrize("name", [e.name for e in registry.available()])
def test_method_runs_end_to_end(name, protocol):
    result = run(name, protocol, epochs=1, patience=1)

    assert len(result.predictions) == len(protocol.test)
    assert np.isfinite(result.predictions).all()
    assert "PC" in result.metrics and "MAE" in result.metrics
    # Anything outside the rating scale is a bug in the head, not a weak model.
    low, high = protocol.score_range
    assert result.predictions.min() >= low - 2
    assert result.predictions.max() <= high + 2


def test_the_mean_baseline_is_the_floor(protocol):
    # Its correlation is 0 by construction. If it is not, the metric is wrong.
    result = run("mean-baseline", protocol)
    assert abs(result.metrics["PC"]) < 1e-6
    assert len(set(result.predictions.tolist())) == 1
