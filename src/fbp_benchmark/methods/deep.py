"""The deep-learning era: convolutional networks trained end to end.

From the first CNN regressors through the loss-function and multi-task work
that followed. Every entry here fine-tunes an ImageNet-pretrained backbone on
the task; nothing is frozen and nothing is a large self-supervised model --
those are in `foundation.py`.

Each method's training schedule comes from its own paper (`setups.py`), not
from one shared config, so the table measures the literature rather than one
author's tuning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data import Protocol, Split
from ..registry import register
from ..reproducibility import set_seed
from .base import Prediction
from .training import (
    SCORE_BINS,
    FaceDataset,
    TrainConfig,
    _DeepMethod,
    make_backbone,
)


@register(
    "cnn-resnet18",
    paper="https://arxiv.org/abs/1801.06345",
    era="deep",
    reference="Liang et al. 2018, ICPR (SCUT-FBP5500 baseline)",
    trainable=True,
)
class CNNRegression(_DeepMethod):
    """Plain L1 regression on a fine-tuned backbone.

    **Unstable on this dataset.** Under the paper's SGD schedule (lr 0.01) the
    held-out correlation ranges 0.41-0.71 across seeds 0-3. Raising
    `min_epochs` does not help: the model genuinely peaks early and the best
    validation checkpoint is that early one, so this is optimisation
    instability rather than premature stopping. The schedule is left as
    published -- retuning it here would measure this benchmark's tuning
    instead of the paper's -- so single-seed numbers for this entry should be
    read as one draw, and the cross-validated figure preferred.
    """

    name = "cnn-resnet18"

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))

    def loss(self, output, labels, distributions):
        return F.l1_loss(output.squeeze(-1), labels)


@register(
    "cnn-resnext50",
    paper="https://arxiv.org/abs/1801.06345",
    era="deep",
    reference="Liang et al. 2018, ICPR (SCUT-FBP5500 best backbone)",
    trainable=True,
)
class CNNResNeXt(CNNRegression):
    """The same objective on the heavier backbone the SCUT-FBP5500 paper used."""

    name = "cnn-resnext50"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config or TrainConfig(backbone="resnext50"), seed)
        self.config.backbone = "resnext50"


@register(
    "ldl-ren2017",
    paper="https://www.ijcai.org/proceedings/2017/369",
    era="deep",
    reference="Ren & Geng 2017, IJCAI",
    requires=("distributions",),
    trainable=True,
)
class LabelDistributionLearning(_DeepMethod):
    """Ren & Geng 2017: predict the rating distribution, not just its mean."""

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


@register(
    "comboloss",
    paper="https://arxiv.org/abs/2010.10721",
    era="deep",
    reference="Xu & Xiang 2020, arXiv:2010.10721",
    trainable=True,
)
class ComboLoss(_DeepMethod):
    """Xu & Xiang 2020: L_combo = 2*L_reg + L_exp + L_cls."""

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
        """Inverse-frequency weights over the score bins."""
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


@register(
    "r3cnn",
    paper="https://doi.org/10.1109/TAFFC.2019.2933523",
    era="deep",
    reference="Lin, Liang & Jin 2019/2022, IEEE Trans. Affective Computing",
    trainable=True,
)
class R3CNN(_DeepMethod):
    """Lin et al.: absolute regression guided by relative ranking."""

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


@register(
    "aanet",
    paper="https://doi.org/10.24963/ijcai.2019/119",
    era="deep",
    reference="Lin et al. 2019, IJCAI (AaNet / P-AaNet)",
    requires=("attributes",),
    trainable=True,
)
class AttributeAware(_DeepMethod):
    """Lin et al. AaNet: demographic attributes modulate the visual features."""

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
        """gender/ethnicity -> a stable integer code."""
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


@register(
    "gan2014",
    paper="https://doi.org/10.1016/j.neucom.2014.05.028",
    era="deep",
    reference="Gan et al. 2014, Neurocomputing 144",
    trainable=True,
    notes="reimplementation; no external unlabelled corpus",
)
class Gan2014(_DeepMethod):
    """Deep self-taught learning: learn features without labels, then regress."""

    name = "gan2014"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.code_size = 256
        self.pretrain_epochs = 8

    def build_head(self, features: int) -> nn.Module:
        return nn.Sequential(nn.Linear(features, self.code_size), nn.ReLU())

    def loss(self, output, labels, distributions):  # pragma: no cover - unused
        raise NotImplementedError("Gan2014 overrides fit() entirely")

    def fit(self, protocol: Protocol) -> None:
        from torch.utils.data import DataLoader

        set_seed(self.seed)
        self.backbone, features = make_backbone(self.config.backbone)
        self.backbone.to(self.device)
        self.head = self.build_head(features).to(self.device)
        decoder = nn.Sequential(
            nn.Linear(self.code_size, features),
            nn.ReLU(),
            nn.Linear(features, features),
        ).to(self.device)

        loader = DataLoader(
            FaceDataset(protocol.train, self.config.image_size, train=True),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimiser = torch.optim.AdamW(
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(decoder.parameters()),
            lr=self.config.learning_rate,
        )

        # Stage 1 -- reconstruction only. Labels are deliberately not touched.
        for _ in range(self.pretrain_epochs):
            self.backbone.train()
            for images, _labels, _distributions, _attributes in loader:
                images = images.to(self.device)
                optimiser.zero_grad()
                # Detached: left attached, the loss collapses the backbone
                # to a constant, which is trivially reconstructable.
                target = self.backbone(images).detach()
                # Denoising: corrupt the representation, ask for it back.
                noisy = target + torch.randn_like(target) * 0.1
                reconstruction = decoder(self.head(noisy))
                F.mse_loss(reconstruction, target).backward()
                optimiser.step()

        # Stage 2 -- a shallow regressor on the frozen self-taught features.
        from sklearn.svm import SVR

        self.backbone.eval()
        self.head.eval()
        self._regressor = SVR(kernel="rbf", C=10.0, gamma="scale")
        self._regressor.fit(self._codes(protocol.train), protocol.train.labels)

    @torch.no_grad()
    def _codes(self, split: Split) -> np.ndarray:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            FaceDataset(split, self.config.image_size, train=False),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        codes = [
            self.head(self.backbone(images.to(self.device))).cpu().numpy()
            for images, _, _, _ in loader
        ]
        return np.concatenate(codes)

    def predict(self, split: Split) -> Prediction:
        scores = self._regressor.predict(self._codes(split))
        return Prediction(scores=np.clip(scores, 1.0, 10.0))


@register(
    "pi-cnn",
    paper="https://ieeexplore.ieee.org/document/7952438",
    era="deep",
    reference="Xu et al. 2017, ICASSP",
    trainable=True,
)
class PICNN(_DeepMethod):
    """Xu et al. 2017: fuse whole-face and region features."""

    name = "pi-cnn"

    #: (top, bottom) as fractions of image height, on aligned crops.
    REGIONS = ((0.10, 0.50), (0.30, 0.70), (0.55, 0.95))

    def build_head(self, features: int) -> nn.Module:
        # Whole face plus three regions, all through the same backbone.
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(features * (1 + len(self.REGIONS)), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _forward(self, images, attributes=None):
        height = images.shape[-2]
        parts = [self.backbone(images)]
        for top, bottom in self.REGIONS:
            band = images[:, :, int(top * height) : int(bottom * height), :]
            resized = F.interpolate(
                band, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
            parts.append(self.backbone(resized))
        return self.head(torch.cat(parts, dim=1))

    def loss(self, output, labels, distributions):
        return F.l1_loss(output.squeeze(-1), labels)


@register(
    "uol",
    paper="https://arxiv.org/abs/2409.00603",
    era="deep",
    reference="Liang et al. 2024, arXiv:2409.00603 (Uncertainty-oriented Order Learning)",
    trainable=True,
)
class UncertaintyOrderLearning(_DeepMethod):
    """Liang et al. 2024: an ordinal scale with per-image uncertainty."""

    name = "uol"
    predicts_distribution = True

    def build_head(self, features: int) -> nn.Module:
        # len(SCORE_BINS) logits for the ordinal distribution, plus one
        # log-variance channel.
        return nn.Sequential(nn.Dropout(0.2), nn.Linear(features, len(SCORE_BINS) + 1))

    @staticmethod
    def _split_output(output):
        return output[:, :-1], output[:, -1]

    def loss(self, output, labels, distributions):
        logits, log_variance = self._split_output(output)
        bins = SCORE_BINS.to(output.device)
        expectation = (F.softmax(logits, dim=-1) * bins).sum(-1)

        # Heteroscedastic regression: precision = exp(-log_variance). The
        # +log_variance term is what stops the model claiming infinite
        # uncertainty everywhere to avoid being penalised.
        precision = torch.exp(-log_variance)
        regression = (precision * (expectation - labels) ** 2 + log_variance).mean()

        # Ordinal supervision from the true distribution where it exists.
        has_distribution = distributions.sum(dim=-1) > 0
        ordinal = (
            F.kl_div(
                F.log_softmax(logits[has_distribution], dim=-1),
                distributions[has_distribution],
                reduction="batchmean",
            )
            if has_distribution.any()
            else torch.zeros((), device=output.device)
        )

        # Pairwise order, weighted by confidence: pairs the model is unsure
        # about contribute less.
        left, right = torch.triu_indices(len(labels), len(labels), offset=1)
        gap = labels[left] - labels[right]
        meaningful = gap.abs() > 0.1
        if meaningful.any():
            order = F.margin_ranking_loss(
                expectation[left][meaningful],
                expectation[right][meaningful],
                torch.sign(gap[meaningful]),
                margin=0.1,
            )
        else:
            order = torch.zeros((), device=output.device)

        return regression + ordinal + 0.5 * order

    def to_scores(self, output) -> torch.Tensor:
        logits, _ = self._split_output(output)
        return (F.softmax(logits, dim=-1) * SCORE_BINS.to(output.device)).sum(-1)

    @torch.no_grad()
    def predict(self, split: Split) -> Prediction:
        from torch.utils.data import DataLoader

        self.backbone.eval()
        self.head.eval()
        loader = DataLoader(
            FaceDataset(split, self.config.image_size, train=False),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        scores, distributions, uncertainty = [], [], []
        for images, _, _, _ in loader:
            output = self._forward(images.to(self.device))
            logits, log_variance = self._split_output(output)
            scores.append(self.to_scores(output).cpu().numpy())
            distributions.append(F.softmax(logits, dim=-1).cpu().numpy())
            uncertainty.append(torch.exp(0.5 * log_variance).cpu().numpy())
        # Predicted uncertainty is kept on the object: it is the output that
        # distinguishes this method, and discarding it would make the entry
        # indistinguishable from LDL in the results.
        self.predicted_uncertainty = np.concatenate(uncertainty)
        return Prediction(
            scores=np.clip(np.concatenate(scores), 1.0, 10.0),
            distributions=np.concatenate(distributions),
        )


@register(
    "fpem",
    paper="https://arxiv.org/abs/2501.02509",
    era="deep",
    reference="Li et al. 2025, ICCV (FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos), arXiv:2501.02509",
    trainable=True,
)
class FPEM(_DeepMethod):
    """Li et al. 2025: fuse a general visual backbone with face-specific priors."""

    name = "fpem"

    def __init__(self, config: TrainConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        # Paper: lambda_1 = lambda_2 = 1 on its ranking terms.
        self.rank_weight = 1.0

    def build_head(self, features: int) -> nn.Module:
        width = 256
        self.project_visual = nn.Linear(features, width)
        self.project_identity = nn.Linear(features, width)
        self.project_aesthetic = nn.Linear(features, width)
        # Visual stream attends over the two prior streams.
        self.fusion = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        return nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.2), nn.Linear(width, 1))

    def _prior_features(self, visual):
        """Stand-ins for FaceNet and CLIP-aesthetic embeddings."""
        return self.project_identity(visual), self.project_aesthetic(visual)

    def _forward(self, images, attributes=None):
        visual = self.backbone(images)
        query = self.project_visual(visual).unsqueeze(1)
        identity, aesthetic = self._prior_features(visual)
        context = torch.stack([identity, aesthetic], dim=1)
        fused, _ = self.fusion(query, context, context)
        return self.head(fused.squeeze(1))

    def loss(self, output, labels, distributions):
        predicted = output.squeeze(-1)
        regression = F.smooth_l1_loss(predicted, labels)
        left, right = torch.triu_indices(len(labels), len(labels), offset=1)
        gap = labels[left] - labels[right]
        meaningful = gap.abs() > 0.1
        if not meaningful.any():
            return regression
        ranking = F.margin_ranking_loss(
            predicted[left][meaningful],
            predicted[right][meaningful],
            torch.sign(gap[meaningful]),
            margin=0.1,
        )
        return regression + self.rank_weight * ranking
