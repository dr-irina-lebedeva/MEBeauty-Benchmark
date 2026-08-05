"""The dataset adapter, exercised without touching the network."""

from __future__ import annotations

import numpy as np
import pytest

from fbp_benchmark.data import DatasetSpec, _build_split


class FakeDataset:
    """The slice of the `datasets` API that `_build_split` actually uses."""

    def __init__(self, columns: dict) -> None:
        self._columns = columns
        self._length = len(next(iter(columns.values())))

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        return {name: values[key] for name, values in self._columns.items()}


def make(**overrides) -> FakeDataset:
    columns = {
        "image": ["<img0>", "<img1>", "<img2>"],
        "image_id": ["a", "b", "c"],
        "beauty_score": [4.0, 6.0, 8.0],
        "rating_distribution": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "landmarks": [[1.0, 2.0, 3.0, 4.0]] * 3,
    }
    columns.update(overrides)
    return FakeDataset(columns)


def test_labels_and_ids_are_read_through():
    split = _build_split("test", make(), DatasetSpec())
    assert list(split.image_ids) == ["a", "b", "c"]
    assert split.labels.tolist() == [4.0, 6.0, 8.0]
    assert len(split) == 3


def test_distributions_are_normalised_from_counts():
    # The dataset ships integer counts; methods need probabilities.
    split = _build_split("test", make(), DatasetSpec())
    assert np.allclose(split.distributions.sum(axis=1), 1.0)


def test_an_all_zero_histogram_becomes_uniform_rather_than_nan():
    dataset = make(rating_distribution=[[0, 0, 0], [0, 2, 0], [0, 0, 2]])
    split = _build_split("test", dataset, DatasetSpec())
    assert np.allclose(split.distributions[0], 1 / 3)
    assert np.isfinite(split.distributions).all()


def test_landmarks_are_reshaped_to_points():
    split = _build_split("test", make(), DatasetSpec())
    assert split.landmarks["a"].shape == (2, 2)


def test_optional_columns_may_be_absent():
    dataset = FakeDataset({"image": ["<i>"], "image_id": ["a"], "beauty_score": [5.0]})
    split = _build_split("test", dataset, DatasetSpec())
    assert split.distributions is None
    assert split.landmarks == {}


def test_a_missing_label_column_names_what_is_available():
    dataset = FakeDataset({"image": ["<i>"], "score": [5.0]})
    with pytest.raises(ValueError, match="no column 'beauty_score'"):
        _build_split("test", dataset, DatasetSpec())


def test_a_missing_label_value_is_refused():
    # Scoring against NaN silently produces NaN metrics.
    with pytest.raises(ValueError, match="cannot score against a missing label"):
        _build_split("test", make(beauty_score=[4.0, float("nan"), 8.0]), DatasetSpec())


def test_n_bins_follows_the_score_range():
    assert DatasetSpec().n_bins == 10
    assert DatasetSpec(score_range=(1.0, 5.0)).n_bins == 5


def test_spec_round_trips_through_yaml(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("repo_id: org/data\nscore_range: [1.0, 5.0]\n")
    spec = DatasetSpec.from_yaml(path)
    assert spec.repo_id == "org/data"
    assert spec.score_range == (1.0, 5.0)


def test_an_unknown_yaml_field_is_refused_rather_than_ignored():
    # A typo that silently does nothing is how a run ends up on the wrong label.
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "spec.yaml"
        path.write_text("labl_column: beauty_score\n")
        with pytest.raises(ValueError, match="unknown field"):
            DatasetSpec.from_yaml(path)
