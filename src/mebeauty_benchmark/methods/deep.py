"""CNN methods: the shared training harness and the losses that distinguish them.

From 2015 onward the field converged on one recipe -- fine-tune an ImageNet
backbone on aligned face crops -- and the interesting differences moved into
the *objective*. So the training loop lives here once, and each method supplies
only its head and its loss:

| Method | Objective |
|---|---|
| `cnn-resnet18` / `cnn-resnext50` | plain L1 regression (Xie 2015, Liang 2018, MEBeauty 2022 baselines) |
| `ldl-ren2017` | predict the rating *distribution*, KL against the true one |
| `comboloss` | regression + classification + expectation, summed |
| `r3cnn` | regression plus a pairwise relative-ranking term |
| `aanet` | attribute-aware: demographic embedding modulates the features |

**Nobody trains from scratch at 1,399 images.** Every method fine-tunes a
pretrained backbone, which is what the papers do and what makes the comparison
about the objective rather than about who had more data.

**Early stopping is on validation, never test.** The val split exists for this;
using test would turn every reported number into a best-of-N and inflate the
whole table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..benchmark.base import Prediction
from ..benchmark.protocol import Protocol, Split, set_seed

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
    """One method's training configuration.

    Defaults are conservative; real values come from `setups.py`, which
    records each paper's published setup and what had to change. Build one
    with `TrainConfig.from_setup(...)` rather than filling it in by hand, so a
    run's settings always trace back to a citation.
    """

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
        from .setups import setup_for

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
    ) -> None:
        from torchvision import transforms

        self.paths = split.image_paths
        # Attribute-conditioned methods need these per sample; everything else
        # gets zeros and ignores them. Carried by the dataset rather than
        # stashed on the method, so a shuffled batch can never be paired with
        # the wrong attributes.
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
            # The paper's recipe: resize to 256, then take a `size` crop --
            # random while training, centred at evaluation. Evaluating with a
            # plain resize instead (as this did) shows the model a face at a
            # different scale from the one it trained on, which is a
            # self-inflicted domain shift, not a property of the method.
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
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        distribution = (
            self.distributions[index]
            if self.distributions is not None
            else torch.zeros(len(SCORE_BINS))
        )
        return tensor, self.labels[index], distribution, self.attributes[index]


def make_backbone(name: str) -> tuple[nn.Module, int]:
    """A pretrained backbone with its classifier removed.

    Returns the module and its output width. The three families expose their
    classifier under different attributes (`fc`, `classifier`, `heads`), which
    is why this cannot be one line.
    """
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
        """Every trainable module this method owns, keyed by attribute name.

        Methods build extra submodules in `build_head` -- AaNet's attribute
        embedding and gate, FPEM's projections and fusion, TransFBP's
        cross-attention -- and assign them to `self` rather than folding them
        into the returned head. Those are discovered here instead of being
        listed by name, because a hardcoded list silently omits whatever is
        added next: an omitted module is never moved to the device (a crash),
        never given to the optimiser (it trains at its random initialisation),
        and never restored with the best epoch (the "best" weights are a
        mixture of two epochs). All three of those were real.
        """
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


class CNNRegression(_DeepMethod):
    """Plain L1 regression on a fine-tuned backbone.

    The baseline shared by Xie 2015, the SCUT-FBP5500 paper and the 2022
    MEBeauty release. L1 rather than L2 because beauty labels are means over a
    handful of raters and a squared penalty lets the noisiest labels dominate.
    """

    name = "cnn-resnet18"

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))

    def loss(self, output, labels, distributions):
        return F.l1_loss(output.squeeze(-1), labels)


class CNNResNeXt(CNNRegression):
    """The same objective on the heavier backbone the SCUT-FBP5500 paper used."""

    name = "cnn-resnext50"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config or TrainConfig(backbone="resnext50"), seed)
        self.config.backbone = "resnext50"


class LabelDistributionLearning(_DeepMethod):
    """Ren & Geng 2017: predict the rating distribution, not just its mean.

    Trained with KL against the true distribution; the reported score is the
    distribution's expectation. This is the method MEBeauty's soft labels exist
    for -- it is the only entry here that uses information a mean throws away,
    and it is scored on the distribution metrics as well as the point ones.
    """

    name = "ldl-ren2017"
    predicts_distribution = True

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, len(SCORE_BINS)))

    def loss(self, output, labels, distributions):
        predicted = F.log_softmax(output, dim=-1)
        # If an image has no distribution (all zeros), fall back to regressing
        # its mean rather than training against an impossible target.
        has_distribution = distributions.sum(dim=-1) > 0
        if not has_distribution.any():
            expectation = (F.softmax(output, -1) * SCORE_BINS.to(output.device)).sum(-1)
            return F.l1_loss(expectation, labels)
        return F.kl_div(
            predicted[has_distribution],
            distributions[has_distribution],
            reduction="batchmean",
        )

    def to_scores(self, output) -> torch.Tensor:
        return (F.softmax(output, dim=-1) * SCORE_BINS.to(output.device)).sum(-1)


class ComboLoss(_DeepMethod):
    """Xu & Xiang 2020: L_combo = 2*L_reg + L_exp + L_cls.

    Three complementary terms, quoted from the paper: `L_reg` is L1 between a
    **regression output** and the label, `L_exp` is L1 between that same
    regression output and the expectation of the classification distribution,
    and `L_cls` is a class-balanced cross-entropy. Coefficients alpha=2,
    beta=1, gamma=1 are the paper's.

    The regression output and the classification distribution must be
    *separate heads*: `L_exp` is what ties them together, so computing both
    from one distribution head makes `L_reg` and `L_exp` the same quantity and
    deletes the term. An earlier version of this class did exactly that, and
    also used MSE where the paper uses L1.

    `L_cls` is weighted by inverse class frequency in the training split.
    MEBeauty's scores pile up around 6, so an unweighted cross-entropy would
    let the middle bins dominate -- which is the imbalance the paper's
    "category balancing weights" exist to correct.
    """

    name = "comboloss"
    predicts_distribution = True

    #: The paper's loss coefficients, in its own notation.
    ALPHA, BETA, GAMMA = 2.0, 1.0, 1.0

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.class_weights: torch.Tensor | None = None

    def build_head(self, features: int) -> nn.Module:
        # Classification logits; the regression scalar is a sibling head.
        self.regressor = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, len(SCORE_BINS)))

    @staticmethod
    def class_weights_for(labels: np.ndarray) -> torch.Tensor:
        """Inverse-frequency weights over the score bins.

        MEBeauty's scores pile up around 6, so an unweighted cross-entropy
        would be dominated by the middle bins. Normalised to mean 1 over the
        occupied bins so the loss keeps its scale; empty bins get 0 rather
        than dividing by zero, and contribute nothing anyway.
        """
        bins = np.clip(np.round(labels) - 1, 0, len(SCORE_BINS) - 1).astype(int)
        counts = np.bincount(bins, minlength=len(SCORE_BINS))
        weights = np.where(counts > 0, len(bins) / np.maximum(counts, 1), 0.0)
        weights = weights / weights[counts > 0].mean()
        return torch.tensor(weights, dtype=torch.float32)

    def fit(self, protocol: Protocol) -> None:
        # Computed from the training split only -- deriving them from all
        # labels would leak the test distribution into the loss.
        self.class_weights = self.class_weights_for(protocol.train.labels)
        super().fit(protocol)

    def _forward(self, images, attributes=None):
        visual = self.backbone(images)
        return self.head(visual), self.regressor(visual).squeeze(-1)

    def loss(self, output, labels, distributions):
        logits, predicted = output
        bins = SCORE_BINS.to(logits.device)
        expectation = (F.softmax(logits, dim=-1) * bins).sum(-1)
        target = torch.clamp(torch.round(labels) - 1, 0, len(bins) - 1).long()

        regression = F.l1_loss(predicted, labels)
        # Detached: this term should pull the regressor toward the
        # distribution's expectation, not drag the distribution toward the
        # regressor and collapse both onto each other.
        expectation_term = F.l1_loss(predicted, expectation.detach())
        classification = F.cross_entropy(
            logits, target, weight=self.class_weights.to(logits.device)
        )
        return (
            self.ALPHA * regression
            + self.BETA * expectation_term
            + self.GAMMA * classification
        )

    def to_scores(self, output) -> torch.Tensor:
        _, predicted = output
        return predicted

    def distribution_of(self, output) -> torch.Tensor:
        logits, _ = output
        return F.softmax(logits, dim=-1)


class R3CNN(_DeepMethod):
    """Lin et al.: absolute regression guided by relative ranking.

    Each batch contributes both terms -- regression on individual scores, and a
    margin ranking loss over every pair within the batch. Pairs come from the
    batch rather than a separately-built pair set, which keeps memory flat; at
    1,399 images an explicit pair list is unnecessary.
    """

    name = "r3cnn"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.rank_weight = 0.5

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))

    def loss(self, output, labels, distributions):
        predicted = output.squeeze(-1)
        regression = F.l1_loss(predicted, labels)

        # All ordered pairs in the batch where the true scores differ enough
        # for the ordering to be meaningful.
        left, right = torch.triu_indices(len(labels), len(labels), offset=1)
        gap = labels[left] - labels[right]
        meaningful = gap.abs() > 0.1
        if not meaningful.any():
            return regression
        sign = torch.sign(gap[meaningful])
        ranking = F.margin_ranking_loss(
            predicted[left][meaningful],
            predicted[right][meaningful],
            sign,
            margin=0.1,
        )
        return regression + self.rank_weight * ranking


class AttributeAware(_DeepMethod):
    """Lin et al. AaNet: demographic attributes modulate the visual features.

    The original learns filter modulation from attributes. Here the shipped
    gender and ethnicity labels are embedded and used to gate the backbone's
    features, which preserves the mechanism -- attributes change *how* the
    image is read, not just what is added to the final score.

    Worth stating plainly in any write-up: this method conditions on ethnicity
    and gender by design. On a multi-ethnic beauty dataset that is exactly the
    thing a fairness analysis should scrutinise, not a neutral architectural
    choice.
    """

    name = "aanet"

    #: Room for every gender x ethnicity combination, plus headroom.
    MAX_ATTRIBUTES = 32

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self._lookup: dict[str, int] = {}

    def build_head(self, features: int) -> nn.Module:
        self.embedding = nn.Embedding(self.MAX_ATTRIBUTES, 32)
        self.gate = nn.Sequential(nn.Linear(32, features), nn.Sigmoid())
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))

    def attribute_codes(self, split: Split) -> np.ndarray:
        """gender/ethnicity -> a stable integer code.

        Codes are assigned in first-seen order and reused, so val and test map
        to the same embedding rows as train. A combination never seen in
        training falls back to 0 rather than indexing out of bounds.
        """
        keys = (split.metadata["gender"] + "/" + split.metadata["ethnicity"]).tolist()
        for key in keys:
            if key not in self._lookup and len(self._lookup) < self.MAX_ATTRIBUTES:
                self._lookup[key] = len(self._lookup)
        return np.array([self._lookup.get(k, 0) for k in keys])

    def fit(self, protocol: Protocol) -> None:
        # Populate the lookup from training before any split is encoded.
        self.attribute_codes(protocol.train)
        super().fit(protocol)
        self.embedding.to(self.device)
        self.gate.to(self.device)

    def _forward(self, images, attributes=None):
        visual = self.backbone(images)
        if attributes is not None:
            visual = visual * self.gate(self.embedding(attributes))
        return self.head(visual)

    def loss(self, output, labels, distributions):
        return F.l1_loss(output.squeeze(-1), labels)
