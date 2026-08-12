"""Landmark-geometry methods: Eisenthal 2006, Kagian 2008, Fan 2012.

The first decade of facial-beauty prediction shared one pipeline -- landmarks,
hand-built geometric features, a shallow regressor -- and differed in how the
features were formed and which regressor followed. All three run on CPU in
seconds, which makes them the natural floor for a benchmark whose upper rows
are transformers.

**These are reimplementations in the papers' spirit, not exact reproductions,
and the results table must say so.** Each original used a different landmark
set to the 68 points MEBeauty ships -- Eisenthal's were placed by hand, Kagian
used 84 -- so the feature spaces cannot be identical. What is preserved is the
part that defines each method:

| Method | Feature construction | Regressor |
|---|---|---|
| Eisenthal 2006 | normalised pairwise distances + symmetry + appearance summary | KNN + ridge ensemble |
| Kagian 2008 | all normalised pairwise distances, then feature selection | SVR (RBF) |
| Fan 2012 | ratios of distances -- proportions, not lengths | polynomial ridge |

**Why normalisation is not a detail.** Raw pixel distances encode how large the
face is in frame, which is a property of the crop, not the person. Every method
here divides by inter-ocular distance first. Without that, a model can score
respectably by learning the photographer's zoom -- the geometric equivalent of
the shortcut this project has already had to remove twice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data import Protocol, Split
from ..registry import register
from .base import Prediction

#: 68-point (ibug/300-W) indices used for normalisation and symmetry.
LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)

#: A compact, well-spread subset of the 68 points. Using all 68 gives 2,278
#: pairwise distances for 1,962 training images -- more features than samples,
#: which is how the classical methods overfit rather than how they worked.
#: These 19 cover jaw, brows, eyes, nose and mouth.
KEY_POINTS = (0, 4, 8, 12, 16, 19, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 62)

#: Mirror pairs for the symmetry features, as (left, right) indices.
SYMMETRY_PAIRS = ((0, 16), (4, 12), (19, 24), (36, 45), (39, 42), (48, 54), (31, 35))


def landmarks_for(protocol: Protocol) -> dict[str, np.ndarray]:
    """image_id -> (68, 2) landmarks, pooled from every split.

    Landmarks travel with the rows rather than in a side file, so this only
    merges the three splits' dictionaries. Raises rather than degrading
    silently: these methods *are* their geometry, and one that quietly
    predicted the training mean for every image because a column was missing
    would look like a weak method rather than a misconfigured run.
    """
    pooled: dict[str, np.ndarray] = {}
    for split in (protocol.train, protocol.val, protocol.test):
        pooled.update(split.landmarks)
    if not pooled:
        raise ValueError(
            f"No landmarks in {protocol.spec.repo_id}/{protocol.spec.config}. "
            "The classical methods are built on facial geometry and cannot "
            "run without them -- set `landmark_column` on the DatasetSpec, or "
            "choose a config that carries landmarks."
        )
    return pooled


def _normalised(points: np.ndarray) -> np.ndarray:
    """Centre on the face and scale by inter-ocular distance.

    Removes translation and scale, so what remains is shape. Falls back to
    overall spread if the eyes coincide (extreme profile views, which this
    dataset does contain).
    """
    left = points[LEFT_EYE].mean(axis=0)
    right = points[RIGHT_EYE].mean(axis=0)
    scale = float(np.linalg.norm(right - left))
    if scale < 1e-6:
        scale = float(points.std()) or 1.0
    return (points - points.mean(axis=0)) / scale


def pairwise_distances(points: np.ndarray, indices=KEY_POINTS) -> np.ndarray:
    """Normalised distances between every pair of `indices`."""
    selected = _normalised(points)[list(indices)]
    rows, cols = np.triu_indices(len(indices), k=1)
    return np.linalg.norm(selected[rows] - selected[cols], axis=1)


def symmetry_features(points: np.ndarray) -> np.ndarray:
    """How far each mirror pair sits from the face's own midline.

    Symmetry is the one geometric property the psychology literature is most
    consistent about, and it is not recoverable from unsigned pairwise
    distances -- hence a separate block.
    """
    normalised = _normalised(points)
    midline = normalised[27:31, 0].mean()
    return np.array(
        [
            abs(abs(normalised[a, 0] - midline) - abs(normalised[b, 0] - midline))
            for a, b in SYMMETRY_PAIRS
        ]
    )


def proportion_features(points: np.ndarray, indices=KEY_POINTS) -> np.ndarray:
    """Ratios between distances -- Fan's proportions rather than lengths.

    Scale-invariant by construction, so this block survives even if the
    inter-ocular normalisation is imperfect.
    """
    distances = pairwise_distances(points, indices)
    # Ratios against a fixed reference set, rather than all N^2 pairs, which
    # would explode the feature count past the sample size.
    reference = distances[:: max(1, len(distances) // 12)][:12]
    return (distances[:, None] / (reference[None, :] + 1e-8)).ravel()


class _GeometricMethod:
    """Shared plumbing: build features, standardise, fit, predict."""

    name = "geometric"
    #: How many features survive univariate selection. Declared here rather
    #: than left to each subclass: it is read by `fit`, so a subclass that
    #: forgot it would fail only at training time.
    n_features: int = 30

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._landmarks: dict[str, np.ndarray] = {}
        self._model: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._selected: np.ndarray | None = None
        self._fallback = 0.0

    def features(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def make_model(self):  # pragma: no cover
        raise NotImplementedError

    def _matrix(self, split: Split) -> np.ndarray:
        rows: list[np.ndarray | None] = []
        for image_id in split.image_ids:
            points = self._landmarks.get(image_id)
            if points is None:
                # No landmarks: emit zeros, which standardise to the training
                # mean and so predict the mean rather than crashing a whole run.
                rows.append(np.zeros_like(rows[0]) if rows else None)
                continue
            rows.append(self.features(points))
        width = next(len(r) for r in rows if r is not None)
        return np.vstack([r if r is not None else np.zeros(width) for r in rows])

    def fit(self, protocol: Protocol) -> None:
        self._landmarks = landmarks_for(protocol)
        self._fallback = float(protocol.train.labels.mean())

        features = self._matrix(protocol.train)
        mean = features.mean(axis=0)
        spread = features.std(axis=0)
        # A constant feature has zero spread; dividing by it yields inf.
        spread[spread < 1e-8] = 1.0
        self._mean, self._std = mean, spread
        standardised = (features - mean) / spread

        # Univariate selection, as all three papers used in some form: with
        # more features than images, keeping everything fits noise.
        target = protocol.train.labels
        centred = target - target.mean()
        correlation = np.abs(
            (standardised * centred[:, None]).sum(axis=0)
            / (np.sqrt((standardised**2).sum(axis=0) * (centred**2).sum()) + 1e-12)
        )
        keep = min(self.n_features, standardised.shape[1])
        self._selected = np.argsort(correlation)[::-1][:keep]

        self._model = self.make_model()
        self._model.fit(standardised[:, self._selected], target)

    def predict(self, split: Split) -> Prediction:
        if self._model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        features = (self._matrix(split) - self._mean) / self._std
        scores = self._model.predict(features[:, self._selected])
        return Prediction(scores=np.clip(scores, 1.0, 10.0))


@register(
    "eisenthal2006",
    paper="https://doi.org/10.1162/089976606774841602",
    era="classical",
    reference="Eisenthal, Dror & Ruppin 2006, Neural Computation 18(1)",
    requires=("landmarks",),
)
class Eisenthal2006(_GeometricMethod):
    """Geometry + symmetry + appearance summary, KNN and ridge averaged.

    The 2006 paper combined geometric measurements with appearance and
    averaged several predictors. The appearance half used eigenfaces over raw
    pixels; here it is a coarse intensity summary of the aligned crop, which
    preserves the idea (texture carries signal beyond shape) without pretending
    to be the same feature.
    """

    name = "eisenthal2006"
    n_features = 60

    def features(self, points: np.ndarray) -> np.ndarray:
        return np.concatenate([pairwise_distances(points), symmetry_features(points)])

    def make_model(self):
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor

        return VotingRegressor(
            [
                ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance")),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )


@register(
    "kagian2008",
    paper="https://www.sciencedirect.com/science/article/pii/S0042698907005032",
    era="classical",
    reference="Kagian et al. 2008, Vision Research 48(2)",
    requires=("landmarks",),
)
class Kagian2008(_GeometricMethod):
    """All normalised pairwise distances, selected, then SVR with an RBF kernel."""

    name = "kagian2008"
    n_features = 100

    def features(self, points: np.ndarray) -> np.ndarray:
        return pairwise_distances(points)

    def make_model(self):
        from sklearn.svm import SVR

        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)


@register(
    "fan2012",
    era="classical",
    reference="Fan et al. 2012, Pattern Recognition 45(6)",
    requires=("landmarks",),
)
class Fan2012(_GeometricMethod):
    """Facial proportions with a polynomial ridge.

    The paper's claim is that *ratios* between measurements predict
    attractiveness better than the measurements themselves, fitted
    nonlinearly. Both halves are preserved: ratio features, quadratic model.
    """

    name = "fan2012"
    n_features = 80

    def features(self, points: np.ndarray) -> np.ndarray:
        return proportion_features(points)

    def make_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=100.0),
        )
