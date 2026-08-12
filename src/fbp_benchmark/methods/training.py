"""Shared training machinery for every torch-based method.

Split out from the era modules deliberately. `deep.py` and `foundation.py`
should read as a list of *methods*; the DataLoader plumbing, the early-stopping
loop and the backbone factory are not part of any one paper's contribution and
repeating them per era would invite them to drift apart -- at which point two
methods would no longer be trained comparably, which is the one thing a
benchmark must guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data import Protocol, Split
from ..reproducibility import set_seed
from .base import Prediction

#: The 1-10 rating scale, as bin centres for the distribution methods.
SCORE_BINS = torch.arange(1.0, 11.0)

#: ImageNet statistics -- the backbones are pretrained with these.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """One method's training configuration."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    image_size: int = 224
    backbone: str = "resnet18"
    augmentation: tuple[str, ...] = ("hflip",)
    patience: int = 5
    #: Epochs that must pass before early stopping may fire. SGD runs at the
    #: published learning rates (0.01-0.1) have noisy validation curves for
    #: their first several epochs, and patience alone can halt inside that
    #: noise -- measured: `cnn-resnext50` stopped at epoch 9 of 40 and moved
    #: 0.145 PC between two otherwise identical runs.
    min_epochs: int = 0
    num_workers: int = 0

    @classmethod
    def from_setup(cls, method: str, **overrides) -> TrainConfig:
        """Build from the published setup recorded for `method`."""
        from ..setups import setup_for

        setup = setup_for(method)
        config = cls(
            epochs=setup.epochs,
            batch_size=setup.batch_size,
            learning_rate=setup.learning_rate,
            weight_decay=setup.weight_decay,
            momentum=setup.momentum,
            optimizer=setup.optimizer,
            scheduler=setup.scheduler,
            image_size=setup.image_size,
            backbone=setup.backbone,
            augmentation=setup.augmentation,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config


class FaceDataset(Dataset):
    """Aligned crops with labels, and light augmentation for training only."""

    def __init__(
        self,
        split: Split,
        size: int,
        train: bool,
        attribute_codes: np.ndarray | None = None,
        augmentation: tuple[str, ...] = ("hflip",),
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
    ) -> None:
        from torchvision import transforms

        # Images arrive from the Hub as decoded PIL objects rather than as
        # paths on disk. `ImageSource` decodes lazily, so a DataLoader worker
        # touches one row at a time exactly as it would with `Image.open`.
        self.images = split.images
        # Carried by the dataset, not the method, so a shuffled batch cannot
        # be paired with the wrong attributes. Zeros when unused.
        self.attributes = torch.tensor(
            attribute_codes if attribute_codes is not None else np.zeros(len(split)),
            dtype=torch.long,
        )
        self.labels = torch.tensor(split.labels, dtype=torch.float32)
        self.distributions = (
            torch.tensor(split.distributions, dtype=torch.float32)
            if split.distributions is not None
            else None
        )
        # Augmentation follows each paper's recorded recipe (setups.py), so a
        # method that published a colour-jitter pipeline gets one and a method
        # that did not is not silently given an advantage.
        steps = []
        if "resize_256" in augmentation and "random_crop_224" in augmentation:
            # Resize then crop -- random while training, centred at eval.
            # A plain resize at eval would be a self-inflicted domain shift.
            steps.append(transforms.Resize((256, 256)))
            steps.append(
                transforms.RandomCrop(size) if train else transforms.CenterCrop(size)
            )
        else:
            steps += [transforms.Resize((size, size))]
        if train:
            if "hflip" in augmentation:
                steps.append(transforms.RandomHorizontalFlip())
            if "rotation" in augmentation:
                steps.append(transforms.RandomRotation(10))
            if "color_jitter" in augmentation:
                steps.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        # Normalisation belongs to the backbone, not to the harness: a
        # face-verification net trained on (x-127.5)/128 receives garbage if
        # it is handed ImageNet statistics. Measured on transfbp, getting this
        # wrong cost 0.037 correlation.
        steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        tensor = self.transform(self.images[index])
        distribution = (
            self.distributions[index]
            if self.distributions is not None
            else torch.zeros(len(SCORE_BINS))
        )
        return tensor, self.labels[index], distribution, self.attributes[index]


def make_backbone(name: str) -> tuple[nn.Module, int]:
    """A pretrained backbone with its classifier removed."""
    from torchvision import models

    factories = {
        "resnet18": (models.resnet18, models.ResNet18_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
        "resnext50": (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights),
        "vgg16": (models.vgg16, models.VGG16_Weights),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights),
    }
    if name not in factories:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(factories)}")
    factory, weights = factories[name]
    model = factory(weights=weights.DEFAULT)

    if name == "vgg16":
        # Keep the two 4096-d fully-connected layers -- they are most of what
        # VGG learned -- and drop only the 1000-way ImageNet classifier.
        features = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()
    elif name == "vit_b_16":
        features = model.hidden_dim
        model.heads = nn.Identity()
    else:
        features = model.fc.in_features
        model.fc = nn.Identity()
    return model, features


class _DeepMethod:
    """Shared fine-tuning loop. Subclasses define the head and the loss."""

    name = "deep"
    predicts_distribution = False

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        self.config = config or TrainConfig()
        self.seed = seed
        self.device = best_device()
        self.model: nn.Module | None = None

    # --- to be supplied by each method -----------------------------------
    def build_head(self, features: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def loss(self, output, labels, distributions):  # pragma: no cover
        raise NotImplementedError

    def to_scores(self, output) -> torch.Tensor:
        """Turn raw head output into a 1-10 score."""
        return output.squeeze(-1)

    def distribution_of(self, output) -> torch.Tensor:
        """The predicted rating distribution, for methods that emit one."""
        return F.softmax(output, dim=-1)

    def attribute_codes(self, split: Split) -> np.ndarray | None:
        """Per-image attribute code, for methods that condition on one."""
        return None

    # --- shared ------------------------------------------------------------
    def _make_optimiser(self, parameters):
        """SGD or AdamW, whichever the method's paper used."""
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _make_schedule(self, optimiser, steps_per_epoch: int):
        """The paper's schedule shape, rescaled to this dataset's length."""
        total = max(1, self.config.epochs)
        if self.config.scheduler == "step_50":
            # ComboLoss: divide by 10 periodically. The paper's period is 50 of
            # its 200 epochs, so a quarter of training, preserved as a ratio.
            return torch.optim.lr_scheduler.StepLR(
                optimiser, step_size=max(1, total // 4), gamma=0.1
            )
        if self.config.scheduler == "warmup_cosine":
            # FPEM: "linear warm-up and cosine annealing scheme".
            warmup = max(1, int(0.1 * total))

            def factor(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)
        if self.config.scheduler == "warmup_linear":
            # AaNet: linear warm-up then linear decay to zero. The paper's
            # 2K-of-20K warm-up is 10% of training; the fraction is what
            # transfers, not the iteration count.
            warmup = max(1, int(0.1 * total))

            def factor(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                return max(0.0, (total - epoch) / max(1, total - warmup))

            return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total)

    def _forward(self, images, attributes=None):
        return self.head(self.backbone(images))

    def _modules(self) -> dict[str, nn.Module]:
        """Every trainable module this method owns, keyed by attribute name."""
        found = {"backbone": self.backbone, "head": self.head}
        for name, value in vars(self).items():
            if isinstance(value, nn.Module) and name not in found:
                found[name] = value
        return found

    def _snapshot(self) -> dict[str, dict]:
        return {
            name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in self._modules().items()
        }

    def _restore(self, snapshot: dict[str, dict]) -> None:
        for name, state in snapshot.items():
            module = self._modules()[name]
            module.load_state_dict(state)
            module.to(self.device)

    def fit(self, protocol: Protocol) -> None:
        set_seed(self.seed)
        self.backbone, features = make_backbone(self.config.backbone)
        self.head = self.build_head(features)
        self.backbone.to(self.device)
        self.head.to(self.device)

        train_loader = DataLoader(
            FaceDataset(
                protocol.train,
                self.config.image_size,
                train=True,
                attribute_codes=self.attribute_codes(protocol.train),
                augmentation=self.config.augmentation,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        modules = self._modules()
        parameters = []
        for module in modules.values():
            module.to(self.device)
            parameters += list(module.parameters())
        optimiser = self._make_optimiser(parameters)
        schedule = self._make_schedule(optimiser, len(train_loader))

        best_mae, best_state, waited = float("inf"), None, 0
        for epoch in range(self.config.epochs):
            for module in modules.values():
                module.train()
            for images, labels, distributions, attributes in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                distributions = distributions.to(self.device)
                optimiser.zero_grad()
                loss = self.loss(
                    self._forward(images, attributes.to(self.device)),
                    labels,
                    distributions,
                )
                loss.backward()
                optimiser.step()
            schedule.step()

            # Selection on validation only -- see the module docstring.
            predicted = self.predict(protocol.val).scores
            mae = float(np.abs(predicted - protocol.val.labels).mean())
            if mae < best_mae - 1e-4:
                best_mae, waited = mae, 0
                best_state = self._snapshot()
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

    @torch.no_grad()
    def predict(self, split: Split) -> Prediction:
        for module in self._modules().values():
            module.eval()
        loader = DataLoader(
            FaceDataset(
                split,
                self.config.image_size,
                train=False,
                attribute_codes=self.attribute_codes(split),
                augmentation=self.config.augmentation,
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        scores, distributions = [], []
        for images, _, _, attributes in loader:
            output = self._forward(images.to(self.device), attributes.to(self.device))
            scores.append(self.to_scores(output).cpu().numpy())
            if self.predicts_distribution:
                distributions.append(self.distribution_of(output).cpu().numpy())
        return Prediction(
            scores=np.clip(np.concatenate(scores), 1.0, 10.0),
            distributions=np.concatenate(distributions) if distributions else None,
        )
