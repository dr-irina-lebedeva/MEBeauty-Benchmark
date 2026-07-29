import numpy as np
import pandas as pd
import pytest

from mebeauty_benchmark.benchmark.base import (
    MeanBaseline,
    Prediction,
    assert_no_test_leak,
)
from mebeauty_benchmark.benchmark.protocol import Protocol, Split, load_protocol


def _split(name: str, n: int, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    return Split(
        name=name,
        image_ids=np.array([f"{name}{i}" for i in range(n)]),
        labels=rng.uniform(2, 9, n),
        image_paths=[],
        metadata=pd.DataFrame({"gender": ["female"] * n, "ethnicity": ["asian"] * n}),
    )


def _protocol() -> Protocol:
    return Protocol(
        train=_split("tr", 40),
        val=_split("va", 10, seed=1),
        test=_split("te", 20, seed=2),
        label="score",
        images="cropped_256",
        seed=0,
        root=".",
    )


def test_prediction_rejects_non_finite_scores():
    # A NaN escaping into a results table would silently poison every metric.
    with pytest.raises(ValueError, match="NaN or infinity"):
        Prediction(scores=np.array([1.0, np.nan]))


def test_mean_baseline_predicts_the_training_mean():
    protocol = _protocol()
    method = MeanBaseline()

    method.fit(protocol)
    predicted = method.predict(protocol.test).scores

    assert np.allclose(predicted, protocol.train.labels.mean())
    assert len(predicted) == len(protocol.test)


def test_leak_check_passes_for_an_honest_method():
    protocol = _protocol()
    method = MeanBaseline()
    method.fit(protocol)

    assert_no_test_leak(method, protocol)  # must not raise


def test_leak_check_catches_a_method_that_reads_test_labels():
    class Cheater:
        name = "cheater"

        def fit(self, protocol):
            pass

        def predict(self, split):
            # The exact bug the check exists for: returning the answers.
            return Prediction(scores=split.labels.copy())

    with pytest.raises(AssertionError, match="reading the labels"):
        assert_no_test_leak(Cheater(), _protocol())


def test_unknown_label_is_rejected_before_any_work():
    with pytest.raises(ValueError, match="Unknown label"):
        load_protocol("data/mebeauty_v3", label="not_a_column")


def test_unknown_image_config_is_rejected():
    with pytest.raises(ValueError, match="Unknown images"):
        load_protocol("data/mebeauty_v3", images="jpeg2000")


def test_real_protocol_is_deterministic_and_aligned():
    a = load_protocol("data/mebeauty_v3")
    b = load_protocol("data/mebeauty_v3")

    # Two loads must present identical order, or per-image predictions from
    # different runs cannot be compared.
    assert (a.test.image_ids == b.test.image_ids).all()
    assert np.array_equal(a.test.labels, b.test.labels)
    assert len(a.test.image_paths) == len(a.test)
    assert all(p.exists() for p in a.test.image_paths[:20])


def test_splits_do_not_overlap():
    p = load_protocol("data/mebeauty_v3")
    train, val, test = (set(s.image_ids) for s in (p.train, p.val, p.test))

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_labels_are_finite_and_in_range():
    p = load_protocol("data/mebeauty_v3")
    for split in (p.train, p.val, p.test):
        assert np.isfinite(split.labels).all()
        assert split.labels.min() >= 1.0
        assert split.labels.max() <= 10.0


def test_distribution_rows_align_with_the_raw_mean():
    p = load_protocol("data/mebeauty_v3", label="score_raw_mean")
    assert p.test.distributions is not None
    assert p.test.distributions.shape[0] == len(p.test)
    # The distribution counts the integer scores raters gave, so its
    # expectation must reproduce the *unnormalised* mean exactly. If these
    # disagree the two artifacts were built from different rows.
    bins = np.arange(1, 11)
    expectation = (p.test.distributions * bins).sum(axis=1)
    assert np.abs(expectation - p.test.labels).max() < 1e-9


def test_the_canonical_label_is_normalised_away_from_the_raw_mean():
    # `score` corrects for rater scale, so it must *not* equal the
    # distribution mean -- while still ranking images nearly the same way.
    # A test asserting equality would silently pass if normalisation broke.
    raw = load_protocol("data/mebeauty_v3", label="score_raw_mean").test.labels
    normalised = load_protocol("data/mebeauty_v3").test.labels

    assert np.abs(normalised - raw).max() > 0.1
    assert np.corrcoef(normalised, raw)[0, 1] > 0.95
