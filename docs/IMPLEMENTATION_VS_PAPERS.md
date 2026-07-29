# Implementation vs. papers — audit of 2026-07-28

Every benchmark entry checked against its source publication before any result
was produced. This document records what matched, what did not, and what was
changed as a result. It is written so a reviewer can find the gaps without
reading the code.

**Six defects were found and fixed. Two of them made a method impossible to
run at all; three made a method silently wrong; one was a false claim in the
documentation.** Numbers produced before this audit are void.

---

## Defects found

### 1. `fpem` and `transfbp` could not run — crash on first use

Both were written, reviewed and documented, and **neither had ever been
executed**. A one-epoch smoke test on eight images found both immediately:

```
FAIL  fpem:     RuntimeError: Tensor for argument weight is on cpu but expected on mps
FAIL  transfbp: ValueError: Unknown backbone 'vit_b_16'; choose from
                ['resnet18', 'resnet50', 'resnext50']
```

`transfbp` set `config.backbone = "vit_b_16"`, which `make_backbone` did not
know; it built its own ViT and then called the base `fit`, which overwrote it.
`make_backbone` now handles ViT-B/16 and VGG-16 (each family hides its
classifier under a different attribute), and `transfbp` uses the harness like
everything else.

`fpem` crashed for the deeper reason below.

### 2. Extra submodules were never trained, never moved, never restored

The training loop collected parameters like this:

```text
parameters = list(self.backbone.parameters()) + list(self.head.parameters())
for attr in ("embedding", "gate"):          # <- a hardcoded list
```

Any module not named `embedding` or `gate` was invisible. Three separate
consequences, all real:

- **Never moved to the device** — `fpem`'s projections and fusion stayed on
  CPU while the images were on MPS. That is the crash above.
- **Never given to the optimiser** — they would have trained at their random
  initialisation. For `fpem` that is the entire cross-attention fusion, i.e.
  the method; for `transfbp`, the cross-attention that is its contribution.
- **Never snapshotted for early stopping** — `_snapshot` saved only backbone
  and head, so restoring the best epoch produced a *mixture*: best-epoch
  backbone with last-epoch fusion. This affected `aanet` too, which had
  otherwise been running "fine".

Modules are now discovered by walking `vars(self)`, so whatever is added next
is included automatically.

### 3. `comboloss` implemented a different objective from the paper

The paper (verified verbatim from arXiv:2010.10721) defines

> ℒ_combo = α·L_reg + β·L_exp + γ·L_cls, with α=2, β=1, γ=1

where `L_reg` is L1 between a **regression output** and the label, `L_exp` is
L1 between that regression output and the expectation of the classification
distribution, and `L_cls` is a **class-balanced** cross-entropy.

What was implemented:

```text
  F.l1_loss(expectation, labels)      # not L_reg: no regression output exists
+ F.cross_entropy(output, target)     # not weighted
+ F.mse_loss(expectation, labels)     # not L_exp; MSE where the paper uses L1
```

There was only one head, a distribution. With no separate regression output,
`L_reg` and `L_exp` collapse to the same quantity and the term that ties the
two predictions together — the point of the loss — disappears. The
coefficients were also dropped, so `L_reg` carried half the weight it should.

Now: a regression head beside the classification head, the paper's α/β/γ, L1
throughout, and inverse-frequency class weights computed on the training split
only. `L_exp` detaches the expectation so it pulls the regressor toward the
distribution rather than collapsing both together.

### 4. `uol` — the setup file asserted something false

It recorded:

> "Preprint without a stable published implementation section at the time of
> writing" · "No published hyperparameters to quote."

**This is wrong.** arXiv:2409.00603 states all of them. The entry ran a
ResNet-18 under invented settings when the paper specifies:

| | Recorded before | Paper |
|---|---|---|
| Backbone | resnet18 (default) | **VGG16**, ImageNet-pretrained |
| Epochs | 30 (invented) | **100** |
| Scheduler | cosine | cosine annealing, min lr 1e-6 |
| Augmentation | hflip | resize 256 → 224 crop, hflip |
| Optimizer | adamw | Adam |
| LR / batch | 1e-4 / 32 | 1e-4 / 32 ✓ |

Now marked `source="paper"` with the quote attached. The only remaining
optimiser difference is Adam → AdamW, kept deliberately so every entry
decouples weight decay the same way, and recorded as a deviation.

### 5. Evaluation preprocessing did not match training

Methods whose papers specify "resize to 256, random 224 crop" got that in
training but were **evaluated on a plain resize to 224×224**. A face occupies
a visibly different fraction of the frame under those two transforms, so every
such method was tested on a distribution it had not trained on — a
self-inflicted domain shift that would have been read as the method
underperforming. Evaluation now centre-crops from 256, which is the standard
counterpart and what the papers do.

Affects `cnn-resnet18`, `cnn-resnext50`, `comboloss`, and now `uol`.

### 6. `gan2014`'s self-taught stage could collapse

The denoising autoencoder reconstructed `target = self.backbone(images)` with
the target **left attached to the graph**. Nothing prevented stage 1 from
minimising its loss by driving the backbone toward a constant — reconstructing
a constant is trivial — which would destroy the representation the method is
about. The target is now detached, so stage 1 learns the code layer on a fixed
backbone.

---

## Fidelity, method by method

`paper` = training setup quoted from the publication. `adapted` = the paper's
setup could not transfer or was not retrievable.

| Method | Setup | Architecture fidelity | The honest gap |
|---|---|---|---|
| `mean-baseline` | — | exact | none; it is a floor |
| `eisenthal2006` | adapted | mechanism only | paper's landmarks were hand-placed; its eigenface features came from a 92-image corpus. Neither transfers. |
| `kagian2008` | adapted | mechanism only | paper used **84** manual landmarks; this dataset ships 68 automatic ones, so the feature space is smaller |
| `fan2012` | adapted | close | ratio features and a nonlinear fit both preserved |
| `gan2014` | adapted | mechanism only | **no external unlabelled corpus.** The self-taught stage runs on this dataset's own training images, removing the method's main advantage. Read as a floor. |
| `cnn-resnet18` | **paper** | exact | L1 not L2 (label scale doubled) |
| `pi-cnn` | adapted | simplified | fixed horizontal bands, not landmark-driven boxes; single-stage, not cascaded fine-tuning |
| `ldl-ren2017` | adapted | **objective only** | the paper's SLDL is a structural SVM, not a CNN. This keeps the *objective* on the shared backbone so the comparison isolates it. |
| `cnn-resnext50` | **paper** | exact | as above |
| `r3cnn` | adapted | close | implementation section behind IEEE and not retrievable; optimiser settings inherited from the same authors' AaNet paper rather than invented. In-batch pairs. |
| `aanet` | **paper** | **simplified** | the paper modulates *convolution filters* via parameter generators ("filter tuning"/"filter rebirth"). This gates the pooled feature vector — the same idea, a weaker mechanism. |
| `comboloss` | **paper** | close (after fix 3) | backbone is ResNeXt-50, not the paper's SE-ResNeXt-50; torchvision has no SE variant, so the squeeze-excitation blocks are absent |
| `uol` | **paper** (after fix 4) | adapted | the paper's Gaussian-embedding comparator with Monte-Carlo sampling and a Wasserstein hinge is replaced by batch-internal pairwise ordering |
| `fpem` | **paper** | **architecture only — the largest gap** | the paper's PAPM fuses face-pretrained Swin-T with frozen FaceNet; MAEM adds CLIP ViT-B/16 + GPT-2. All are replaced by projections of one shared backbone, which cannot contribute knowledge the backbone lacks. Also built for live video with retouching, a setting absent here. |
| `transfbp` | adapted | partial | **TransMix omitted.** The paper's contribution is a two-stage attention-guided augmentation that mixes images from opposite ends of the score distribution and derives a supervisory score from the model's own attention. Implementing it would make this entry's training loop structurally different from every other. This entry under-represents the method. |

### Where a quote could not be obtained

- **`r3cnn`** — IEEE Trans. Affective Computing 13(1):122–134, paywalled.
- **`transfbp`** — published May 2026, full text not retrievable.
- **`fpem`** — the base learning rate "varies with different training phase"
  and is not stated; 5e-5 is this benchmark's choice.
- **`eisenthal2006` / `kagian2008` / `fan2012` / `gan2014`** — pre-deep-learning
  setups (SVR kernels, PCA dimensions) that do not map onto this pipeline.

These are marked `adapted`, and `adapted` means *this implementation*, not the
published method.

---

## What this means for the results

**Eleven of fifteen entries are reimplementations.** A reimplementation
scoring below its published figure is evidence about this implementation on
this dataset, and nothing else. `fpem` in particular is missing every
component that motivates it and should be expected to underperform badly.

**No published number appears in the results table**, deliberately. Every
published figure comes from SCUT-FBP5500 or SCUT-FBP under different images,
labels, and splits. A table mixing them would invite exactly the comparison
that is invalid.

**The uniform deviation applies to all**: early stopping on validation
(patience 5) replaces each paper's fixed schedule, and epochs are capped at 60.
A single fixed split cannot average out variance the way 5-fold
cross-validation does, so a fixed schedule here would report whatever the last
epoch happened to produce.

## Sources

- [ComboLoss (Xu & Xiang 2020), arXiv:2010.10721](https://arxiv.org/pdf/2010.10721)
- [Uncertainty-oriented Order Learning (Liang et al. 2024), arXiv:2409.00603](https://arxiv.org/html/2409.00603)
- [FPEM (Li et al., ICCV 2025), arXiv:2501.02509](https://arxiv.org/html/2501.02509v1)
- [AaNet (Lin et al., IJCAI 2019)](https://www.ijcai.org/proceedings/2019/119)
- [SCUT-FBP5500 (Liang et al. 2018)](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
