"""The harness: requirement checks, leak detection, and what gets written."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fbp_benchmark.data import DatasetSpec, Protocol, Split
from fbp_benchmark.methods.base import Prediction, assert_no_test_leak
from fbp_benchmark.registry import Entry
from fbp_benchmark.runner import Result, check_requirements, save


def make_split(n=8, **kwargs):
    return Split(
        name="train",
        image_ids=np.array([f"i{k}" for k in range(n)]),
        labels=np.linspace(2.0, 9.0, n),
        images=None,
        **kwargs,
    )


def make_protocol(**kwargs):
    split = make_split(**kwargs)
    return Protocol(
        train=split,
        val=split,
        test=split,
        spec=DatasetSpec(),
        protocol="holdout",
        seed=0,
    )


def entry(name="m", **kwargs):
    return Entry(name=name, factory=object, era="deep", reference="r", **kwargs)


def test_requirements_pass_when_nothing_is_needed():
    check_requirements(entry(), make_protocol())


def test_a_distribution_method_without_histograms_fails_before_training():
    # Training on zeros would report a plausible bad score, which reads as
    # "weak method" rather than "misconfigured run".
    with pytest.raises(ValueError, match="rating distribution column"):
        check_requirements(entry(requires=("distributions",)), make_protocol())


def test_a_landmark_method_without_landmarks_fails_before_training():
    with pytest.raises(ValueError, match="facial landmarks"):
        check_requirements(entry(requires=("landmarks",)), make_protocol())


def test_an_attribute_method_names_the_config_that_would_work():
    with pytest.raises(ValueError, match="fbp_extended"):
        check_requirements(entry(requires=("attributes",)), make_protocol())


def test_a_rater_effects_method_without_ratings_says_which_config_has_them():
    with pytest.raises(ValueError, match="personalized_fbp"):
        check_requirements(entry(requires=("ratings",)), make_protocol())


def test_requirements_are_satisfied_when_the_columns_are_present():
    protocol = make_protocol(
        distributions=np.ones((8, 10)) / 10,
        landmarks={"i0": np.zeros((68, 2))},
        rating_value=np.ones(8),
        rating_image=np.arange(8),
        rating_rater=np.array(["r"] * 8),
    )
    check_requirements(
        entry(requires=("distributions", "landmarks", "ratings")), protocol
    )


def test_a_method_that_ignores_labels_passes_the_leak_check():
    class Honest:
        name = "honest"

        def predict(self, split):
            return Prediction(scores=np.zeros(len(split)))

    assert_no_test_leak(Honest(), make_protocol())


def test_a_method_that_reads_test_labels_is_caught():
    class Cheater:
        name = "cheater"

        def predict(self, split):
            return Prediction(scores=np.asarray(split.labels, dtype=float))

    with pytest.raises(AssertionError, match="reading the labels"):
        assert_no_test_leak(Cheater(), make_protocol())


def test_predictions_containing_nan_are_rejected_at_construction():
    # A NaN reaching the metrics silently produces NaN scores in the table.
    with pytest.raises(ValueError, match="NaN"):
        Prediction(scores=np.array([1.0, np.nan]))


def test_saving_writes_the_result_and_its_predictions(tmp_path):
    result = Result(
        method="m",
        era="deep",
        metrics={"PC": 0.5},
        seconds=1.0,
        predictions=np.array([1.0, 2.0]),
        protocol={"seed": 0},
    )
    path = save(result, tmp_path)
    payload = json.loads(path.read_text())

    assert payload["method"] == "m"
    assert payload["metrics"]["PC"] == 0.5
    # Provenance is what lets a number be reproduced or discounted later.
    assert "git_commit" in payload["environment"]
    assert "git_tree_dirty" in payload["environment"]
    stored = np.load(tmp_path / "m_predictions.npz")["predictions"]
    assert stored.tolist() == [1.0, 2.0]
