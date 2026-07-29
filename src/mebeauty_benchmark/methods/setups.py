"""Per-method training setups, sourced from the papers where they are stated.

Hyperparameters are not neutral. A method trained under someone else's
schedule is not that method, and a benchmark whose settings were invented by
its author measures the author's tuning rather than the literature. So each
entry here records:

- `source`: `paper` if the value is quoted from the publication, `adapted` if
  the paper's value could not transfer and a substitute was reasoned, or
  `default` if the paper does not state one.
- `quote`: the paper's own words, verbatim, where available.
- `deviation`: what was changed for this benchmark and why.

**Why anything deviates at all.** Every published setup here was written for
SCUT-FBP5500 (5,500 images, 5-fold cross-validation, labels on 1-5) or SCUT-FBP
(500 images). MEBeauty's benchmark-v1 is 1,399 training images, a fixed split,
and labels on 1-10. Three consequences run through the table below:

1. **Epoch counts are ceilings, not targets.** An earlier version of this file
   rescaled epochs to match each paper's *optimiser step* count. That was
   wrong: ComboLoss's 200 epochs over 4,400 images shows each image 200 times.
   Matching its step count instead -- 200 x ceil(4400/64) = 13,800 -- on 1,399
   images at batch 64 (22 steps per epoch) would take ~627 epochs, showing
   each image 627 times: more overfitting, not less. Overfitting tracks
   epochs, not steps.
   The paper's epoch count is kept as an upper bound and early stopping
   decides the real length.
2. **Iteration-based schedules must be rescaled.** AaNet's warm-up and decay
   are defined in iterations, so they are converted through this dataset's
   steps-per-epoch rather than copied.
3. **Label scale differs.** Losses that are scale-sensitive (L2, smooth-L1's
   beta) behave differently on 1-10 than on 1-5. Noted per method where it
   bites.

**Early stopping on validation is imposed on every method**, including those
whose papers used a fixed schedule with no validation split. Published setups
could afford fixed schedules because 5-fold cross-validation averaged the
variance away; a single fixed split cannot, and a fixed schedule here would
report whatever the last epoch happened to give. This is a deliberate,
uniform deviation and is recorded as such in every entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["paper", "adapted", "default"]


@dataclass(frozen=True)
class Setup:
    """One method's training configuration, with its provenance."""

    method: str
    reference: str
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 30
    backbone: str = "resnet18"
    image_size: int = 224
    scheduler: str = "cosine"
    augmentation: tuple[str, ...] = ("hflip",)
    source: Source = "default"
    quote: str = ""
    deviation: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "reference": self.reference,
            "source": self.source,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "backbone": self.backbone,
            "image_size": self.image_size,
            "scheduler": self.scheduler,
            "augmentation": list(self.augmentation),
            "paper_quote": self.quote,
            "deviation_from_paper": self.deviation,
            "notes": self.notes,
        }


#: Imposed on every method regardless of its paper. See the module docstring.
UNIFORM_DEVIATION = (
    "Early stopping on the validation split (patience 5) replaces the paper's "
    "fixed schedule: benchmark-v1 is a single fixed split, so a fixed epoch "
    "count would report whatever the last epoch produced rather than the "
    "method's best honest result."
)

MEBEAUTY_TRAIN_IMAGES = 1399

#: No method may run longer than this regardless of its paper. With 1,399
#: training images and early stopping active, anything beyond this is spent
#: memorising: the published schedules were written for 3-4x more data.
MAX_EPOCHS = 60


def capped_epochs(paper_epochs: int) -> int:
    """The paper's epoch count, bounded by what this dataset can support.

    Returns the paper's value when it is already modest, and `MAX_EPOCHS`
    otherwise. Early stopping on validation almost always halts first; this
    only bounds the pathological case.
    """
    return min(paper_epochs, MAX_EPOCHS)


SETUPS: dict[str, Setup] = {
    # ------------------------------------------------------------------ 2006-2012
    "eisenthal2006": Setup(
        method="eisenthal2006",
        reference="Eisenthal, Dror & Ruppin 2006, Neural Computation 18(1)",
        source="adapted",
        optimizer="none",
        notes=(
            "No gradient training: geometric features, feature selection, then "
            "a KNN/ridge ensemble. The paper's landmarks were placed by hand "
            "and its appearance features were eigenfaces over a 92-image "
            "corpus; neither transfers, so the reimplementation keeps the "
            "structure (geometry + symmetry, ensembled shallow regressors) and "
            "substitutes the 68-point landmarks this dataset ships."
        ),
    ),
    "kagian2008": Setup(
        method="kagian2008",
        reference="Kagian et al. 2008, Vision Research 48(2)",
        source="adapted",
        optimizer="none",
        notes=(
            "Paper used 84 manually-placed landmarks, all pairwise distances "
            "normalised by face size, feature selection, then SVR. Structure "
            "preserved; landmark set is the 68-point one available here, so "
            "the feature space is smaller than the original's."
        ),
    ),
    "fan2012": Setup(
        method="fan2012",
        reference="Fan et al. 2012, Pattern Recognition 45(6)",
        source="adapted",
        optimizer="none",
        notes=(
            "Ratios between distances rather than distances themselves, fitted "
            "nonlinearly. Both properties preserved (proportion features, "
            "polynomial ridge)."
        ),
    ),
    # ------------------------------------------------------------------ 2014-2018
    "gan2014": Setup(
        method="gan2014",
        reference="Gan et al. 2014, Neurocomputing 133",
        source="adapted",
        optimizer="adamw",
        learning_rate=1e-4,
        batch_size=32,
        epochs=20,
        notes=(
            "Self-taught: representation from unlabelled faces, then a shallow "
            "regressor. The paper's external unlabelled corpus is unavailable, "
            "so the unsupervised stage runs on this dataset's training images "
            "without their labels. This removes the method's main advantage -- "
            "seeing far more faces than are labelled -- and the result should "
            "be read as a floor for the approach, not a reproduction."
        ),
    ),
    "cnn-resnet18": Setup(
        method="cnn-resnet18",
        reference="Liang et al. 2018, ICPR (SCUT-FBP5500 baseline)",
        source="paper",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        epochs=40,
        backbone="resnet18",
        image_size=224,
        augmentation=("resize_256", "random_crop_224", "hflip"),
        quote=(
            "Each raw RGB image was resized as 256x256, and a 224x224 random "
            "crop was sent to ResNeXt. ... Model parameters were initialized "
            "by pretrained CNN models of ImageNet and updated by mini-batch "
            "Stochastic Gradient Descent (SGD)."
        ),
        deviation=(
            "The paper reports 5-fold cross-validation and a 60/40 split on "
            "5,500 images with labels on 1-5; benchmark-v1 is a fixed split on "
            "1,399 training images with labels on 1-10. L1 is used rather "
            "than the paper's L2 because the doubled label scale puts about "
            "four times the weight on the same relative error, letting the "
            "noisiest labels dominate. " + UNIFORM_DEVIATION
        ),
    ),
    "cnn-resnext50": Setup(
        method="cnn-resnext50",
        reference="Liang et al. 2018, ICPR (SCUT-FBP5500 best backbone)",
        source="paper",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        epochs=40,
        backbone="resnext50",
        image_size=224,
        augmentation=("resize_256", "random_crop_224", "hflip"),
        quote=(
            "ResNeXt-50 obtains the best performance, PC 0.8997 by 5-fold "
            "cross validation and 0.8777 by 60% training / 40% testing."
        ),
        deviation=(
            "Same protocol difference as cnn-resnet18. The paper's 0.8997 is "
            "on SCUT-FBP5500 and must not be compared with any number produced "
            "here. " + UNIFORM_DEVIATION
        ),
    ),
    "pi-cnn": Setup(
        method="pi-cnn",
        reference="Xu et al. 2017, ICASSP",
        source="adapted",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        epochs=40,
        notes=(
            "Paper's cascaded fine-tuning over psychology-motivated regions is "
            "reduced to a single stage with a shared backbone over the whole "
            "face plus three horizontal bands. Bands rather than "
            "landmark-driven boxes is defensible only because every input is "
            "already landmark-aligned."
        ),
        deviation=UNIFORM_DEVIATION,
    ),
    "ldl-ren2017": Setup(
        method="ldl-ren2017",
        reference="Ren & Geng 2017, IJCAI",
        source="adapted",
        optimizer="adamw",
        learning_rate=1e-4,
        batch_size=32,
        epochs=30,
        quote=(
            "Ten-fold cross validation on SCUT-FBP. Six measures: Chebyshev, "
            "Clark, Sorensen, Topsoe, Cosine, Intersection."
        ),
        notes=(
            "The paper's SLDL is a structural SVM over label distributions, "
            "not a CNN. This entry keeps the *objective* -- predict the rating "
            "distribution, evaluate with distribution measures -- on the same "
            "backbone as every other deep entry, so the comparison isolates "
            "the objective rather than confounding it with the model class. "
            "The paper's Sorensen and Topsoe are replaced by KL and Canberra "
            "in this benchmark's metric set."
        ),
        deviation=(
            "500-image SCUT-FBP with ten-fold CV, versus 1,399 images on a "
            "fixed split. " + UNIFORM_DEVIATION
        ),
    ),
    # ------------------------------------------------------------------ 2019-2020
    "r3cnn": Setup(
        method="r3cnn",
        reference="Lin, Liang & Jin 2019/2022, IEEE Trans. Affective Computing",
        source="adapted",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        epochs=40,
        notes=(
            "The paper's implementation section is behind IEEE and was not "
            "retrievable, so the optimiser settings follow its sibling paper "
            "(AaNet, same first author and year) rather than being invented. "
            "Pairs for the ranking term are formed within each batch instead "
            "of from a precomputed pair set: at 1,399 images an explicit list "
            "adds memory without adding information."
        ),
        deviation=(
            "Optimiser settings inherited from the same authors' AaNet paper, "
            "not quoted from R3CNN itself. " + UNIFORM_DEVIATION
        ),
    ),
    "aanet": Setup(
        method="aanet",
        reference="Lin et al. 2019, IJCAI (AaNet / P-AaNet)",
        source="paper",
        optimizer="sgd",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        epochs=40,
        backbone="resnet18",
        scheduler="warmup_linear",
        quote=(
            "trained by using mini-batch Stochastic Gradient Descent (SGD) "
            "with a batch size of 32, a momentum of 0.9, and a weight decay of "
            "5e-4 ... the learning rate is increased from 0 to a peak value of "
            "0.01 in a warm-up schedule of 2K iterations, and then decreased "
            "to 0, linearly, in 18K iterations. ... For ResNet-18 and its "
            "extension networks, we set the peak value of learning rate and "
            "weight decay as 0.1 and 1e-4."
        ),
        deviation=(
            "The paper's 20K-iteration schedule is defined in iterations, not "
            "epochs. At batch 32 on 1,399 images (44 steps/epoch) that would "
            "be ~455 epochs -- far past overfitting on a quarter of the data. "
            "The warm-up *fraction* (10% of training) is preserved and the "
            "total rescaled. ResNet-18 values (peak lr 0.1, weight decay 1e-4) "
            "are used since that is this benchmark's backbone; the AlexNet "
            "values quoted above (0.01, 5e-4) do not apply. " + UNIFORM_DEVIATION
        ),
        notes=(
            "This method conditions on gender and ethnicity by construction. "
            "On a multi-ethnic beauty dataset that belongs in the fairness "
            "analysis, not in a footnote."
        ),
    ),
    "comboloss": Setup(
        method="comboloss",
        reference="Xu & Xiang 2020, arXiv:2010.10721",
        source="paper",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-3,
        batch_size=64,
        # Paper: 200 epochs. Capped -- see MAX_EPOCHS and the module docstring.
        epochs=capped_epochs(200),
        backbone="resnext50",
        image_size=224,
        scheduler="step_50",
        augmentation=(
            "resize_256",
            "random_crop_224",
            "hflip",
            "color_jitter",
            "rotation",
        ),
        quote=(
            "The learning rate starts from 0.01 and is divided by 10 per 50 "
            "epochs. Weight decay and batch size are set as 0.001 and 64, "
            "respectively. The model is trained via SGD with 0.9 momentum for "
            "200 epochs. The images are resized and randomly cropped to "
            "224x224 patches, color jittering and random rotation are applied "
            "for data augmentation. The network is initialized with ImageNet "
            "pretrained weights."
        ),
        deviation=(
            "Backbone is ResNeXt-50, not the paper's SE-ResNeXt-50, which "
            "torchvision does not provide; the squeeze-excitation blocks are "
            "therefore absent. The paper's 200 epochs are capped at "
            f"{MAX_EPOCHS}: 200 passes over 1,399 images is well past the "
            "point where this dataset is being memorised rather than learned. "
            + UNIFORM_DEVIATION
        ),
    ),
    # ------------------------------------------------------------------ 2024-2026
    "uol": Setup(
        method="uol",
        reference="Liang et al. 2024, arXiv:2409.00603 (Uncertainty-oriented "
        "Order Learning)",
        source="paper",
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        # Paper: 100 epochs. Capped -- see MAX_EPOCHS and the module docstring.
        epochs=capped_epochs(100),
        backbone="vgg16",
        image_size=224,
        scheduler="cosine",
        augmentation=("resize_256", "random_crop_224", "hflip"),
        quote=(
            "Adam optimizer with a batch size of 32 ... learning rate is 1e-4 "
            "at the beginning ... Cosine Annealing scheduler with the minimal "
            "learning rate 1e-6 ... 100 epochs ... pretrained VGG16 on "
            "ImageNet as the backbone ... resized to 256x256 ... 224x224 "
            "center cropping and random horizontal flipping."
        ),
        notes=(
            "An earlier version of this file recorded 'no published "
            "hyperparameters to quote' and ran a ResNet-18. That was wrong: "
            "the paper states all of them, and the backbone is VGG16. The "
            "values above are now the paper's. AdamW is used rather than the "
            "paper's Adam -- the only remaining optimiser difference -- "
            "because every other entry decouples weight decay and mixing the "
            "two would confound the comparison. The method's distinguishing "
            "parts (ordinal distribution head, predicted per-image "
            "uncertainty, pairwise order) are implemented; the paper's "
            "Gaussian-embedding comparator with Monte-Carlo sampling and its "
            "Wasserstein hinge are replaced by batch-internal pairwise "
            "ordering, so this is an adaptation of the method under the "
            "paper's training setup."
        ),
        deviation=(
            "Adam -> AdamW; the comparator is batch-internal rather than "
            "Monte-Carlo over sampled Gaussian embeddings. " + UNIFORM_DEVIATION
        ),
    ),
    "fpem": Setup(
        method="fpem",
        reference="Li et al. 2025, ICCV (FPEM: Face Prior Enhanced Facial "
        "Attractiveness Prediction for Live Videos), arXiv:2501.02509",
        source="paper",
        optimizer="adamw",
        learning_rate=5e-5,
        weight_decay=1e-4,
        batch_size=32,
        # Paper: 50 epochs per training phase. This entry has one phase.
        epochs=capped_epochs(50),
        scheduler="warmup_cosine",
        quote=(
            "AdamW is selected as the optimizer, and the learning rate "
            "scheduler is set with linear warm-up and cosine annealing "
            "scheme. ... Each training phase lasts 50 epochs ... batch size "
            "is set as 32."
        ),
        notes=(
            "ARCHITECTURE ONLY, and this is the largest gap in the table. The "
            "paper's PAPM fuses a Swin-T (pretrained on face tasks) with a "
            "frozen FaceNet, and its MAEM adds CLIP ViT-B/16 + GPT-2 as image "
            "and text encoders. Here all of them are replaced by separate "
            "projections of one shared torchvision backbone, which cannot "
            "contribute knowledge the backbone does not already hold. The "
            "optimiser, schedule, batch size and epoch count are the paper's; "
            "the encoders are not. Expect this entry to underperform its "
            "published figure substantially -- that is a fact about this "
            "implementation, not about FPEM. The paper is also built for live "
            "video with face retouching, a setting this dataset does not "
            "have. Its base learning rate 'varies with different training "
            "phase' and is not stated, so 5e-5 is this benchmark's choice for "
            "an attention-fusion head."
        ),
        deviation=(
            "All pretrained prior encoders (Swin-T, FaceNet, CLIP, GPT-2) "
            "missing; two-phase training reduced to one; base learning rate "
            "not stated in the paper. " + UNIFORM_DEVIATION
        ),
    ),
    "transfbp": Setup(
        method="transfbp",
        reference="Boukhari & Dornaika 2026, Cognitive Computation",
        source="adapted",
        optimizer="adamw",
        learning_rate=3e-5,
        weight_decay=0.05,
        batch_size=16,
        epochs=25,
        backbone="vit_b_16",
        image_size=224,
        notes=(
            "Published May 2026 and not retrievable in full at the time of "
            "writing, so hyperparameters follow standard ViT fine-tuning "
            "practice (low lr, high weight decay) rather than a quoted value. "
            "The paper's attention-guided TransMix augmentation is NOT "
            "implemented -- it mixes images and labels by attention mass and "
            "would make this entry's training loop structurally different from "
            "every other. This entry therefore under-represents the method."
        ),
        deviation=(
            "TransMix augmentation omitted; hyperparameters not quoted. "
            + UNIFORM_DEVIATION
        ),
    ),
    "mean-baseline": Setup(
        method="mean-baseline",
        reference="--",
        source="default",
        optimizer="none",
        notes=(
            "Predicts the training mean. Its Pearson correlation is 0 by "
            "construction, which is the point: it shows what no signal scores, "
            "and its MAE is usually closer to a weak model's than expected."
        ),
    ),
}


def setup_for(method: str) -> Setup:
    if method not in SETUPS:
        raise KeyError(f"No setup recorded for {method!r}; add one to SETUPS")
    return SETUPS[method]


def provenance_table() -> str:
    """Markdown table of every method's setup and where it came from."""
    lines = [
        "| Method | Reference | Source | Optimizer | LR | Batch | Epochs | Backbone |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for setup in SETUPS.values():
        lines.append(
            f"| `{setup.method}` | {setup.reference} | **{setup.source}** | "
            f"{setup.optimizer} | {setup.learning_rate} | {setup.batch_size} | "
            f"{setup.epochs} | {setup.backbone} |"
        )
    return "\n".join(lines)
