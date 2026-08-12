"""Xu et al. 2018: transfer rich deep features, then regress with Bayesian ridge.

The method predates end-to-end fine-tuning for this task. Nothing is trained
by gradient descent: a network pretrained on *face verification* is frozen,
features are pooled from several depths and concatenated, and a Bayesian ridge
model maps them to a score. The claim is that face-identity features already
encode most of what predicts attractiveness.

The face-verification backbone matters and is not interchangeable with an
ImageNet one -- that is the paper's whole premise -- so this uses
InceptionResnetV1 trained on VGGFace2 via `facenet-pytorch`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data import Protocol, Split
from ..registry import register
from ..reproducibility import set_seed
from .base import Prediction

#: Depths pooled and concatenated. The paper fuses several stacked layers
#: rather than taking only the embedding, so mid-level texture survives
#: alongside the identity code.
FUSION_LAYERS = ("repeat_2", "repeat_3", "block8")


@register(
    "transfbp",
    paper="https://arxiv.org/abs/1803.07253",
    era="deep",
    reference="Xu, Jinhai & Yuan 2018, arXiv:1803.07253 (TransFBP)",
    notes="frozen face-verification features + Bayesian ridge; no fine-tuning",
)
class TransFBP:
    """Frozen VGGFace2 features from several depths, fused, then Bayesian ridge."""

    name = "transfbp"

    def __init__(self, seed: int = 0, image_size: int = 160) -> None:
        self.seed = seed
        self.image_size = image_size
        self._model: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _backbone(self):
        import torch

        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "transfbp needs a face-verification backbone: "
                "pip install facenet-pytorch (add --no-deps if it tries to "
                "downgrade torchvision)"
            ) from exc

        from .training import best_device

        if getattr(self, "_net", None) is None:
            self._device = best_device()
            self._net = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
            for parameter in self._net.parameters():
                parameter.requires_grad_(False)
            self._torch = torch
        return self._net

    def _features(self, split: Split) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        from .training import FaceDataset

        net = self._backbone()
        pooled: dict[str, list] = {name: [] for name in FUSION_LAYERS}
        handles = []

        def capture(name):
            def hook(_module, _inputs, output):
                # Global average pool: the paper fuses layer *descriptors*, not
                # feature maps, and the maps differ in spatial size by depth.
                pooled[name].append(output.mean(dim=(2, 3)).cpu())

            return hook

        for name in FUSION_LAYERS:
            handles.append(getattr(net, name).register_forward_hook(capture(name)))

        loader = DataLoader(
            FaceDataset(split, self.image_size, train=False, augmentation=()),
            batch_size=32,
            shuffle=False,
        )
        with torch.no_grad():
            for images, *_ in loader:
                net(images.to(self._device))
        for handle in handles:
            handle.remove()

        blocks = [torch.cat(pooled[name]).numpy() for name in FUSION_LAYERS]
        return np.concatenate(blocks, axis=1)

    def fit(self, protocol: Protocol) -> None:
        from sklearn.linear_model import BayesianRidge

        set_seed(self.seed)
        features = self._features(protocol.train)
        mean = features.mean(axis=0)
        spread = features.std(axis=0)
        # A constant feature has zero spread; dividing by it yields inf.
        spread[spread < 1e-8] = 1.0
        self._mean, self._std = mean, spread
        self._model = BayesianRidge()
        self._model.fit((features - mean) / spread, protocol.train.labels)

    def predict(self, split: Split) -> Prediction:
        if self._model is None:
            raise RuntimeError("transfbp.predict called before fit")
        features = (self._features(split) - self._mean) / self._std
        low, high = 1.0, 10.0
        return Prediction(scores=np.clip(self._model.predict(features), low, high))
