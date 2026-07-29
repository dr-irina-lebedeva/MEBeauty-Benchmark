"""The remaining benchmark entries: self-taught, region-based, ordinal, fusion.

These share the fine-tuning harness in `deep.py` and differ in architecture
rather than only in loss, so each builds its own forward path.

| Method | Year | The idea it contributes |
|---|---|---|
| `gan2014` | 2014 | features learned from *unlabelled* faces, then a shallow regressor |
| `pi-cnn` | 2017 | psychologically-motivated regions (eyes, nose, mouth) fused |
| `uol` | 2024 | scores as a distribution over an ordinal scale, with uncertainty |
| `fpem` | 2025 | fuse a general backbone with face-identity and aesthetic priors |
| `transfbp` | 2026 | ViT patch/CLS cross-attention instead of pooled features |

**All five are reimplementations in each paper's spirit, and the results table
must label them as such.** Where a paper depends on a resource this benchmark
does not have -- Gan's external unlabelled corpus, FPEM's CLIP aesthetic
predictor, published pretrained weights -- the substitution is named in that
class's docstring rather than buried. A reimplementation scoring below its
published number is evidence about this implementation and this dataset, not
about the original work.

**Why no paper's reported figure appears here.** Published numbers come from
SCUT-FBP5500 or the 2022 MEBeauty release under different splits and labels.
Putting them in the same table as these results would invite exactly the
comparison that is not valid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark.base import Prediction
from ..benchmark.protocol import Protocol, Split, set_seed
from .deep import SCORE_BINS, FaceDataset, TrainConfig, _DeepMethod, make_backbone


class Gan2014(_DeepMethod):
    """Deep self-taught learning: learn features without labels, then regress.

    The 2014 method pretrains a feature extractor on unlabelled faces and fits
    a shallow regressor on top, the appeal being that unlabelled faces are
    plentiful while rated ones are not.

    **Substitution, stated plainly.** The original's external unlabelled corpus
    is not available here, so the self-taught stage is a denoising
    autoencoder trained on this dataset's *training images only, without their
    labels*. That preserves the mechanism -- representation learned from pixels
    alone, labels used only by the final regressor -- while staying inside the
    protocol. It does not reproduce the original's advantage, which came from
    seeing far more faces than the labelled set contains.
    """

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
                # Detached: the target must be a fixed thing to reconstruct.
                # Left attached, stage 1 can drive the loss to zero by
                # collapsing the backbone toward a constant -- reconstructing
                # a constant is trivial -- which destroys the representation
                # the whole method is about.
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


class PICNN(_DeepMethod):
    """Xu et al. 2017: fuse whole-face and region features.

    The psychological premise is that raters attend to specific regions rather
    than the face as a whole, so the network is given those regions explicitly.
    Three crops -- upper (eyes/brows), middle (nose), lower (mouth/jaw) -- are
    taken from the aligned image and encoded by a shared backbone alongside the
    full face, then concatenated.

    Regions are fixed horizontal bands rather than landmark-driven boxes. That
    is defensible only because every input is already landmark-aligned: the
    eyes sit at the same height in every crop, which is precisely what the
    alignment step guarantees. On unaligned images this would be wrong.
    """

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


class UncertaintyOrderLearning(_DeepMethod):
    """Liang et al. 2024: an ordinal scale with per-image uncertainty.

    Two departures from plain regression, both of which suit this dataset:

    - **Ordinal, not continuous.** The head emits a distribution over the 1-10
      scale, so the model can express "somewhere between 6 and 7" rather than
      being forced to a point.
    - **Uncertainty is predicted, not assumed.** A second output is a
      per-image log-variance, used in a heteroscedastic loss: the model may
      down-weight its own errors on images it finds ambiguous, provided it says
      so in advance.

    That matters here specifically. MEBeauty's labels rest on between 7 and 103
    ratings, so their reliability genuinely varies -- a model that must be
    equally confident everywhere is being asked for something the data does not
    support.

    The paper recovers scores through a Bradley-Terry treatment of pairwise
    comparisons. This implementation keeps the ordinal-plus-uncertainty core
    and uses batch-internal pairwise ordering, as in `R3CNN`, rather than
    building an explicit comparison graph -- at 1,399 images the graph adds
    memory without adding information.
    """

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


class FPEM(_DeepMethod):
    """Li et al. 2025: fuse a general visual backbone with face-specific priors.

    The paper's claim is that facial beauty needs three complementary signals --
    general visual features, face-identity structure, and a learned aesthetic
    prior -- combined by cross-attention rather than concatenation.

    **Substitutions, stated plainly.** The original uses Swin, FaceNet and a
    CLIP aesthetic predictor. Here the general branch is the configured
    torchvision backbone and the prior branches are two independently
    initialised projections of it. That preserves the *architecture* -- three
    streams, cross-attention fusion, joint regression and ranking -- but not the
    external knowledge, which is where much of the paper's benefit lives. This
    entry should be read as "the fusion architecture on equal footing", not as
    a reproduction of FPEM's reported performance.

    Wiring real encoders is a matter of replacing `_prior_features`; the fusion
    and heads need no change.
    """

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
        """Stand-ins for FaceNet and CLIP-aesthetic embeddings.

        Separate projections of the shared backbone. Genuinely weaker than the
        paper's independent encoders -- these cannot contribute knowledge the
        backbone does not already hold -- and named here so the results table
        can say so.
        """
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


class TransFBP(_DeepMethod):
    """Boukhari & Dornaika 2026: cross-attention over ViT tokens.

    A ViT's CLS token is a pooled summary; the patch tokens hold where the
    information actually is. This method keeps both and lets the CLS token
    attend over the patches before regression, rather than discarding the
    spatial detail that pooling throws away.

    The paper's attention-guided TransMix augmentation is **not** implemented:
    it mixes two images and their labels in proportion to attention mass, which
    needs the attention map during augmentation and would make this entry's
    training loop differ structurally from every other. Omitting it is recorded
    here because it is part of the paper's contribution, so this entry
    under-represents the method.

    Requires a ViT backbone; `TrainConfig.backbone` is overridden accordingly.
    """

    name = "transfbp"

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
        """Patch and CLS tokens from the ViT encoder.

        `self.backbone` is torchvision's ViT with its classification head
        removed. Its `forward` returns only the pooled CLS token, which is
        exactly the information this method exists to avoid discarding, so the
        encoder is driven directly instead.
        """
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
