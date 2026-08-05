"""Post-2025 additions: foundation-model features, TTA, and ensembling.

Everything in `deep.py` and `modern.py` reimplements a published method. This
module is different: it is what the literature since 2025 says should work on
*this* dataset, adapted rather than reproduced, and it exists to answer "can
we do better than the survey table?"

**The reasoning, which matters more than the code.** benchmark-v1 has 1,399
training images. Every published method here fine-tunes an ImageNet backbone,
and at this scale that is the binding constraint: a ResNet-18 has 11M
parameters chasing 1,399 labels, so most of the training run is spent
memorising. The 2025-2026 results reflect this -- the gains come from *better
representations*, not better heads:

| Reported on SCUT-FBP5500 | PC |
|---|---|
| R3CNN (2022) | 0.9142 |
| FairViT-GAN (2025) | 0.9230 |
| MD-Net / SynergyNet (2025) | 0.9235 |
| Hybrid VMamba-ViT (2025) | 0.9261 |

MD-Net's own ablation is the clearest evidence: removing its diffusion prior
costs 0.021 PC, removing Mamba costs 0.015, and replacing cross-attention
fusion with concatenation costs 0.011. The *pretrained prior* carries more
than the architecture.

So this module takes the cheapest version of that lesson: a frozen
self-supervised backbone that has seen far more faces than this dataset
contains, with a small head on top. Nothing is fine-tuned by default, so there
is almost nothing to overfit.

**Why DINOv2 and not DINOv3.** DINOv3 is stronger and would be the better
choice, but its weights are gated behind a licence requiring personal details.
This project does not automate its way around a licence -- the same reason
`validate_on_scut.py` will not download SCUT-FBP5500. DINOv2 is Apache-2.0 and
ungated, so it is what ships. Anyone who has accepted the DINOv3 licence can
set `--backbone` to a DINOv3 checkpoint and this code runs unchanged.

**What is deliberately not implemented.** MD-Net needs a Stable Diffusion
U-Net encoder and Vision Mamba. `mamba-ssm` requires CUDA kernels and does not
run on Apple Silicon, so MD-Net cannot be reproduced on this machine at all --
recorded here rather than quietly omitted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..benchmark.base import Prediction
from ..benchmark.protocol import Protocol, Split, set_seed
from .deep import FaceDataset, TrainConfig, _DeepMethod

#: Ungated, Apache-2.0. Swap for a DINOv3 checkpoint if you hold that licence.
DEFAULT_BACKBONE = "facebook/dinov2-base"


class FoundationBackbone(nn.Module):
    """A frozen self-supervised transformer, exposing pooled + patch tokens.

    Wrapped rather than used directly so the rest of the harness sees the same
    interface as a torchvision backbone.
    """

    def __init__(self, name: str = DEFAULT_BACKBONE, trainable_blocks: int = 0):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        self.width = self.model.config.hidden_size

        # Frozen by default. With 1,399 images, fine-tuning 86M parameters is
        # the failure mode this module exists to avoid; `trainable_blocks`
        # unfreezes only the last few layers when that is wanted.
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if trainable_blocks > 0:
            layers = self.model.encoder.layer
            for layer in layers[-trainable_blocks:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images).last_hidden_state
        cls, patches = output[:, 0], output[:, 1:]
        # CLS summarises, mean-pooled patches keep what pooling discards.
        # Concatenating both is consistently better than either alone and
        # costs one extra linear layer.
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)


class DINOv2Regression(_DeepMethod):
    """Frozen DINOv2 features, small MLP head, L1 regression.

    The simplest form of the 2025 lesson, and the one most likely to hold up:
    no architecture novelty at all, just a representation trained on far more
    images than this dataset has labels for.

    Test-time augmentation is on by default -- a horizontal flip and its
    average. Faces are near-symmetric, the training augmentation already
    includes flips, and it costs one extra forward pass.
    """

    name = "dinov2-linear"
    predicts_distribution = False

    def __init__(
        self,
        config: TrainConfig | None = None,
        seed: int = 0,
        backbone_name: str = DEFAULT_BACKBONE,
        trainable_blocks: int = 0,
        test_time_flip: bool = True,
    ) -> None:
        super().__init__(config or TrainConfig(), seed)
        self.backbone_name = backbone_name
        self.trainable_blocks = trainable_blocks
        self.test_time_flip = test_time_flip

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(features),
            nn.Dropout(0.2),
            nn.Linear(features, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def fit(self, protocol: Protocol) -> None:
        set_seed(self.seed)
        self.backbone = FoundationBackbone(self.backbone_name, self.trainable_blocks)
        self.head = self.build_head(self.backbone.width * 2)
        self.backbone.to(self.device)
        self.head.to(self.device)
        self._fit_loop(protocol)

    def _fit_loop(self, protocol: Protocol) -> None:
        """The shared loop, minus the torchvision backbone construction."""
        train_loader = DataLoader(
            FaceDataset(
                protocol.train,
                self.config.image_size,
                train=True,
                augmentation=self.config.augmentation,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        modules = self._modules()
        for module in modules.values():
            module.to(self.device)
        # Only what is actually trainable: handing frozen parameters to an
        # optimiser with weight decay would decay them toward zero even though
        # they receive no gradient.
        parameters = [
            p for m in modules.values() for p in m.parameters() if p.requires_grad
        ]
        optimiser = self._make_optimiser(parameters)
        schedule = self._make_schedule(optimiser, len(train_loader))

        best_mae, best_state, waited, epoch = float("inf"), None, 0, 0
        for epoch in range(self.config.epochs):
            for module in modules.values():
                module.train()
            for images, labels, _distributions, _attributes in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimiser.zero_grad()
                loss = self.loss(self._forward(images), labels, None)
                loss.backward()
                optimiser.step()
            schedule.step()

            mae = float(
                np.abs(self.predict(protocol.val).scores - protocol.val.labels).mean()
            )
            if mae < best_mae - 1e-4:
                best_mae, waited, best_state = mae, 0, self._snapshot()
            else:
                waited += 1
                if (
                    waited >= self.config.patience
                    and epoch + 1 >= self.config.min_epochs
                ):
                    break

        self.epochs_trained = epoch + 1
        self.best_val_mae = best_mae
        if best_state is not None:
            self._restore(best_state)

    def loss(self, output, labels, distributions):
        return F.l1_loss(output.squeeze(-1), labels)

    @torch.no_grad()
    def predict(self, split: Split) -> Prediction:
        for module in self._modules().values():
            module.eval()
        loader = DataLoader(
            FaceDataset(
                split,
                self.config.image_size,
                train=False,
                augmentation=self.config.augmentation,
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        scores = []
        for images, _labels, _distributions, _attributes in loader:
            images = images.to(self.device)
            predicted = self.to_scores(self._forward(images))
            if self.test_time_flip:
                flipped = self.to_scores(self._forward(torch.flip(images, dims=[3])))
                predicted = 0.5 * (predicted + flipped)
            scores.append(predicted.cpu().numpy())
        return Prediction(scores=np.clip(np.concatenate(scores), 1.0, 10.0))


class DINOv2Partial(DINOv2Regression):
    """The same, with the last transformer blocks unfrozen.

    Included so the frozen-vs-tuned question is answered by measurement rather
    than by the argument in this module's docstring. If unfreezing helps, the
    argument was wrong and the table will say so.
    """

    name = "dinov2-partial"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed, trainable_blocks=4)


class Ensemble:
    """Average the predictions of several already-run methods.

    Not a method in the usual sense -- it never sees an image. It reads the
    per-image predictions the harness already saves and combines them, which
    is part of why those files are written.

    **Combined by rank, not by raw score.** The members disagree about scale:
    a distribution's expectation is pulled toward the centre of the range,
    while a plain regressor is not, so averaging raw scores lets the
    widest-spread member dominate. Ranks are averaged instead, then mapped
    back onto the training labels' own distribution by quantile, so MAE and
    RMSE stay meaningful -- a pure rank average would score terribly on both
    while correlating fine.
    """

    name = "ensemble"

    def __init__(self, members: list[str], train_labels: np.ndarray) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = members
        self.reference = np.sort(np.asarray(train_labels, dtype=float))

    @staticmethod
    def _ranks(values: np.ndarray) -> np.ndarray:
        """Ordinal ranks, 0-based. Ties broken by position, which is fine at
        this scale and avoids a scipy dependency for one function."""
        return np.argsort(np.argsort(np.asarray(values, dtype=float)))

    def predict_from(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in self.members if name not in predictions]
        if missing:
            raise KeyError(f"No predictions for ensemble member(s): {missing}")

        mean_rank = np.mean([self._ranks(predictions[m]) for m in self.members], axis=0)
        # Rank -> quantile in (0, 1), then through the empirical quantile
        # function of the training labels.
        quantile = (self._ranks(mean_rank) + 0.5) / len(mean_rank)
        return np.quantile(self.reference, quantile)
