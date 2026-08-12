"""Foundation-model era: large pretrained backbones, frozen or lightly adapted.

Adapted rather than reproduced. With 1,962 training images the binding
constraint is representation, not architecture, so these entries replace the
ImageNet backbone used elsewhere with a self-supervised one and change little
else. `dinov2-partial` is the strongest method in the benchmark.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data import Protocol, Split
from ..registry import register
from ..reproducibility import set_seed
from .base import Prediction
from .training import FaceDataset, TrainConfig, _DeepMethod

#: Ungated, Apache-2.0. Swap for a DINOv3 checkpoint if you hold that licence.
DEFAULT_BACKBONE = "facebook/dinov2-base"


class FoundationBackbone(nn.Module):
    """A frozen self-supervised transformer, exposing pooled + patch tokens."""

    def __init__(self, name: str = DEFAULT_BACKBONE, trainable_blocks: int = 0):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        self.width = self.model.config.hidden_size

        # Frozen by default. With 1,962 images, fine-tuning 86M parameters is
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


@register(
    "dinov2-linear",
    paper="https://arxiv.org/abs/2304.07193",
    era="foundation",
    reference="Oquab et al. 2024 (DINOv2) + this benchmark",
    trainable=True,
    notes="frozen backbone, linear head",
)
class DINOv2Regression(_DeepMethod):
    """Frozen DINOv2 features, small MLP head, L1 regression."""

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


@register(
    "dinov2-partial",
    paper="https://arxiv.org/abs/2304.07193",
    era="foundation",
    reference="Oquab et al. 2024 (DINOv2) + this benchmark",
    trainable=True,
    notes="last blocks unfrozen",
)
class DINOv2Partial(DINOv2Regression):
    """The same, with the last transformer blocks unfrozen."""

    name = "dinov2-partial"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed, trainable_blocks=4)


class Ensemble:
    """Average the predictions of several already-run methods."""

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


@register(
    "xattn-vit",
    era="foundation",
    reference="Boukhari & Dornaika 2026 (cross-attention ViT)",
    trainable=True,
    notes="ViT-B/16 backbone",
)
class CrossAttentionViT(_DeepMethod):
    """Cross-attention between the CLS token and the patch tokens of a ViT."""

    name = "xattn-vit"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config or TrainConfig(), seed)
        self.config.backbone = "vit_b_16"
        self.config.image_size = 224  # ViT-B/16 is fixed at 224

    def build_head(self, features: int) -> nn.Module:
        self.cross_attention = nn.MultiheadAttention(
            features, num_heads=8, batch_first=True
        )
        return nn.Sequential(
            nn.LayerNorm(features), nn.Dropout(0.2), nn.Linear(features, 1)
        )

    def _tokens(self, images):
        """Patch and CLS tokens from the ViT encoder."""
        x = self.backbone._process_input(images)
        cls = self.backbone.class_token.expand(x.shape[0], -1, -1)
        x = self.backbone.encoder(torch.cat([cls, x], dim=1))
        return x[:, :1], x[:, 1:]

    def _forward(self, images, attributes=None):
        cls, patches = self._tokens(images)
        attended, _ = self.cross_attention(cls, patches, patches)
        # Residual: keep the pooled summary, add what attention found.
        return self.head((cls + attended).squeeze(1))

    def loss(self, output, labels, distributions):
        return F.smooth_l1_loss(output.squeeze(-1), labels)


@register(
    "vit-fbp",
    paper="https://www.ijeetc.com/vol13/IJEETC-V13N3-252.pdf",
    era="foundation",
    reference="Boukhari 2023, IJEETC 13(3) (ViT-FBP)",
    trainable=True,
    notes="plain ViT-B/16, fine-tuned end to end",
)
class ViTFBP(_DeepMethod):
    """A plain ViT-B/16 fine-tuned for regression, with no added structure.

    The paper's claim is that a transformer beats the CNN baselines once its
    schedule is tuned for the task rather than inherited from ImageNet. It
    ships here as the control for `xattn-vit`: the two share a backbone and
    differ only in whether cross-attention is added, so any gap is
    attributable to that.
    """

    name = "vit-fbp"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config or TrainConfig(), seed)
        self.config.backbone = "vit_b_16"
        self.config.image_size = 224

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(features), nn.Dropout(0.1), nn.Linear(features, 1)
        )

    def loss(self, output, labels, distributions):
        return F.smooth_l1_loss(output.squeeze(-1), labels)
