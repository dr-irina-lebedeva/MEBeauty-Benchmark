"""Where the data comes from, and how to point the benchmark at your own.

Everything loads from the Hugging Face Hub. There is no download script, no
preprocessing step and no `data/` directory to populate: `load_protocol()`
fetches the dataset, caches it, and hands the methods images and labels.

**Using a different dataset.** Nothing in this file is specific to MEBeauty
except the defaults. A dataset works here if one row is one face and it has,
at minimum, an image column, a numeric label column and a split. Describe it
with a `DatasetSpec` -- in Python or in a YAML file under `configs/` -- and
every method runs against it unchanged:

    spec = DatasetSpec(
        repo_id="your-org/your-dataset",
        label_column="beauty_score",
        score_range=(1.0, 5.0),
    )

Columns the benchmark can use but does not require:

* `distribution_column` -- a histogram of the raw ratings. Label-distribution
  methods need it; without it they are skipped rather than silently trained on
  a distribution reconstructed from the mean, which is not the same thing.
* `landmark_column` -- facial landmarks. The classical methods are built on
  geometry and cannot run without them; the deep and foundation methods never
  look at them.

**Splits.** Two protocols ship, and a run must say which it used. `holdout`
uses the dataset's own train/validation/test split. `cv` uses the integer
`fold_column`, holding out one fold at a time. Neither is re-derived here: a
benchmark that reshuffles splits itself would undo the grouping that keeps
photographs of the same person out of two splits at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

#: The dataset this benchmark was built for.
MEBEAUTY_REPO = "dr-irina-lebedeva/MEBeauty"


@dataclass(frozen=True)
class DatasetSpec:
    """How to read one dataset. The only place a dataset is described."""

    repo_id: str = MEBEAUTY_REPO
    #: Hugging Face config name. `fbp_extended` is the default rather than the
    #: smaller `fbp` because it is the smallest config every registered method
    #: can run against: attribute-aware methods need `gender`/`ethnicity`, and
    #: a default that leaves one method unrunnable is a bad default.
    config: str = "fbp_extended"
    image_column: str = "image"
    label_column: str = "beauty_score"
    id_column: str = "image_id"
    distribution_column: str | None = "rating_distribution"
    landmark_column: str | None = "landmarks"
    fold_column: str | None = "cv_fold"
    #: Nested per-rater ratings: a list of `{rater_id, score}` per row. Off by
    #: default because it is only in the personalized configs and costs memory.
    #: Methods that model rater effects declare `requires=("ratings",)`.
    ratings_column: str | None = None
    #: Hub split names, in train/val/test order.
    split_names: tuple[str, str, str] = ("train", "validation", "test")
    #: The rating scale. Metrics and the soft-label binning need it, and it is
    #: not inferrable from the data -- a dataset whose observed scores run
    #: 2.1-8.7 may still be on a 1-10 scale.
    score_range: tuple[float, float] = (1.0, 10.0)
    #: Extra columns carried through to `Split.metadata`, for subgroup
    #: analysis. Missing ones are ignored rather than raising.
    metadata_columns: tuple[str, ...] = ("gender", "ethnicity")
    #: Passed to `load_dataset`; set for private or gated repositories.
    token: str | None = None

    @property
    def n_bins(self) -> int:
        low, high = self.score_range
        return round(high - low) + 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetSpec:
        """Read a spec from `configs/*.yaml`."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"{path}: unknown field(s) {unknown}. Known fields: {sorted(known)}"
            )
        for key in ("split_names", "score_range", "metadata_columns"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)


class ImageSource:
    """Lazy access to a split's images.

    Methods receive this rather than file paths. Hugging Face decodes an image
    only when the row is touched, so holding a whole split costs a handle, not
    2,462 decoded bitmaps -- and the same interface works whether the bytes
    came from the Hub or from a local folder.
    """

    def __init__(self, dataset: Any, column: str) -> None:
        self._dataset = dataset
        self._column = column

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Image:
        image = self._dataset[int(index)][self._column]
        return image.convert("RGB") if image.mode != "RGB" else image


@dataclass(frozen=True)
class Split:
    """One split's images, labels and whatever else the dataset carries."""

    name: str
    image_ids: np.ndarray
    labels: np.ndarray
    images: ImageSource
    distributions: np.ndarray | None = None
    #: How many raters produced each label, and how much they disagreed.
    #: Both are derived from the rating histogram rather than read from
    #: separate columns, so they are available in every config that ships a
    #: distribution -- not only the heaviest one. Label reliability is *known*
    #: here and varies ~10x across a split; a method may legitimately use it
    #: on train and val to weight its loss.
    n_ratings: np.ndarray | None = None
    label_std: np.ndarray | None = None
    #: Individual ratings in flat form: for observation k, `rating_image[k]` is
    #: the row it belongs to, `rating_rater[k]` who gave it and
    #: `rating_value[k]` what they said. Flat rather than nested so a loss can
    #: index it without a Python loop per batch.
    rating_image: np.ndarray | None = None
    rating_rater: np.ndarray | None = None
    rating_value: np.ndarray | None = None
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def standard_error(self) -> np.ndarray | None:
        """Standard error of each label's mean: sd / sqrt(n)."""
        if self.n_ratings is None or self.label_std is None:
            return None
        return np.asarray(self.label_std) / np.sqrt(np.asarray(self.n_ratings))

    def with_labels(self, labels: np.ndarray) -> Split:
        """A copy carrying different labels. Used by the leak check."""
        return replace(self, labels=np.asarray(labels, dtype=float))


@dataclass(frozen=True)
class Protocol:
    """Everything a method is given, plus the provenance of it."""

    train: Split
    val: Split
    test: Split
    spec: DatasetSpec
    protocol: str
    seed: int
    fold: int | None = None

    @property
    def n_bins(self) -> int:
        return self.spec.n_bins

    @property
    def score_range(self) -> tuple[float, float]:
        return self.spec.score_range

    def describe(self) -> dict[str, object]:
        return {
            "dataset": self.spec.repo_id,
            "config": self.spec.config,
            "label": self.spec.label_column,
            "protocol": self.protocol,
            "fold": self.fold,
            "seed": self.seed,
            "score_range": list(self.spec.score_range),
            "sizes": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
            },
        }


def _column(dataset: Any, name: str | None) -> list | None:
    if not name or name not in dataset.column_names:
        return None
    return dataset[name]


def _build_split(name: str, dataset: Any, spec: DatasetSpec) -> Split:
    if spec.label_column not in dataset.column_names:
        raise ValueError(
            f"{spec.repo_id}/{spec.config}: no column {spec.label_column!r}. "
            f"Available: {sorted(dataset.column_names)}"
        )
    labels = np.asarray(dataset[spec.label_column], dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError(
            f"{name}: {int((~np.isfinite(labels)).sum())} rows have no "
            f"{spec.label_column!r}. A benchmark cannot score against a "
            "missing label."
        )

    ids = _column(dataset, spec.id_column)
    image_ids = np.asarray(
        ids if ids is not None else [f"{name}_{i}" for i in range(len(dataset))]
    )

    raw_distributions = _column(dataset, spec.distribution_column)
    distributions = n_ratings = label_std = None
    if raw_distributions is not None:
        counts = np.asarray(raw_distributions, dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        # Counts, not probabilities: normalise, and leave any all-zero row as
        # uniform rather than dividing by zero.
        distributions = np.divide(
            counts,
            totals,
            out=np.full_like(counts, 1.0 / counts.shape[1]),
            where=totals > 0,
        )
        # The histogram is a complete record of the raw ratings, so the count
        # and their spread come out of it exactly -- no extra column needed.
        low, _ = spec.score_range
        bins = low + np.arange(counts.shape[1], dtype=float)
        n_ratings = totals.ravel()
        mean = (counts * bins).sum(axis=1) / np.where(n_ratings > 0, n_ratings, 1)
        variance = (counts * (bins - mean[:, None]) ** 2).sum(axis=1)
        # ddof=1, matching the sample standard deviation a groupby would give.
        label_std = np.sqrt(
            np.divide(
                variance,
                n_ratings - 1,
                out=np.zeros_like(variance),
                where=n_ratings > 1,
            )
        )

    raw_landmarks = _column(dataset, spec.landmark_column)
    landmarks: dict[str, np.ndarray] = {}
    if raw_landmarks is not None:
        for image_id, points in zip(image_ids, raw_landmarks):
            landmarks[image_id] = np.asarray(points, dtype=float).reshape(-1, 2)

    rating_image = rating_rater = rating_value = None
    nested = _column(dataset, spec.ratings_column)
    if nested is not None:
        rows, raters, values = [], [], []
        for index, records in enumerate(nested):
            for record in records:
                rows.append(index)
                raters.append(record["rater_id"])
                values.append(float(record["score"]))
        rating_image = np.asarray(rows, dtype=np.int64)
        rating_rater = np.asarray(raters)
        rating_value = np.asarray(values, dtype=float)

    present = [c for c in spec.metadata_columns if c in dataset.column_names]
    metadata = (
        pd.DataFrame({c: dataset[c] for c in present}) if present else pd.DataFrame()
    )

    return Split(
        name=name,
        image_ids=image_ids,
        labels=labels,
        images=ImageSource(dataset, spec.image_column),
        distributions=distributions,
        n_ratings=n_ratings,
        label_std=label_std,
        rating_image=rating_image,
        rating_rater=rating_rater,
        rating_value=rating_value,
        landmarks=landmarks,
        metadata=metadata,
    )


def load_protocol(
    spec: DatasetSpec | None = None,
    protocol: str = "holdout",
    fold: int | None = None,
    seed: int = 0,
) -> Protocol:
    """Fetch a dataset from the Hub and assemble the three splits.

    `holdout` uses the dataset's own split. `cv` needs `fold`, and builds the
    test set from that fold, the validation set from the next one round, and
    trains on the rest -- so every image is tested exactly once across a full
    sweep and no image is ever in two roles at once.
    """
    from datasets import load_dataset

    spec = spec or DatasetSpec()
    if protocol not in ("holdout", "cv"):
        raise ValueError(f"Unknown protocol {protocol!r}; use 'holdout' or 'cv'")

    if protocol == "holdout":
        parts = {
            role: load_dataset(
                spec.repo_id, spec.config, split=hub_split, token=spec.token
            )
            for role, hub_split in zip(("train", "val", "test"), spec.split_names)
        }
        return Protocol(
            train=_build_split("train", parts["train"], spec),
            val=_build_split("val", parts["val"], spec),
            test=_build_split("test", parts["test"], spec),
            spec=spec,
            protocol=protocol,
            seed=seed,
        )

    if fold is None:
        raise ValueError("protocol='cv' requires a fold index")
    if not spec.fold_column:
        raise ValueError("protocol='cv' requires `fold_column` on the spec")

    from datasets import concatenate_datasets

    everything = concatenate_datasets(
        [
            load_dataset(spec.repo_id, spec.config, split=s, token=spec.token)
            for s in spec.split_names
        ]
    )
    folds = np.asarray(everything[spec.fold_column])
    n_folds = int(folds.max()) + 1
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}, got {fold}")

    test_mask = folds == fold
    val_mask = folds == (fold + 1) % n_folds
    train_mask = ~(test_mask | val_mask)
    parts = {
        "train": everything.select(np.flatnonzero(train_mask)),
        "val": everything.select(np.flatnonzero(val_mask)),
        "test": everything.select(np.flatnonzero(test_mask)),
    }
    return Protocol(
        train=_build_split("train", parts["train"], spec),
        val=_build_split("val", parts["val"], spec),
        test=_build_split("test", parts["test"], spec),
        spec=spec,
        protocol=protocol,
        seed=seed,
        fold=fold,
    )
