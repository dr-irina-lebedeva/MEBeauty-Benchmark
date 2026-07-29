# MEBeauty benchmark-v1

A reproducible comparison of facial-beauty-prediction methods from 2006 to 2026
on a single, fixed protocol.

## The protocol

| | |
|---|---|
| Splits | train 1,399 / val 185 / test 434, fixed |
| Label | `score` — per-rater-normalised mean over valid raters |
| Images | `cropped_256` (SCRFD-aligned 256×256) by default |
| Metrics | PC, SROCC, MAE, RMSE, plus six distribution measures for LDL methods |

**Every label rests on at least 10 ratings.** 449 images with thinner support
carry no label and are absent from all three splits; they remain in the
dataset with their pixels and metadata. This is why the split sizes are
smaller than the image count.

**The label corrects for rater scale.** Nobody rated every image, so a plain
mean lets a rater's habit — one averages 6.1 where another averages 4.0 on
the same faces — leak into whichever images they happened to see. Each
rater's scores are standardised in their own shrunken units before averaging.
`score_raw_mean` ships beside `score` for anyone who wants the plain mean;
`docs/DATASET_PROBLEMS_2026-07-28.md` records what the correction cost.

**Fixed splits, not k-fold.** The splits were rebuilt so every photograph of
one person sits in a single split. Re-folding at random would undo that and
reintroduce identity leakage — cross-validation would give tighter error bars
around a quietly inflated number.

**Results here are not comparable to published SCUT-FBP5500 or MEBeauty-2022
figures.** Different images, different labels, different splits. The 2022
release's split cannot even be reconstructed: it was generated without a random
seed (`DATASET_PROBLEMS`, Finding 20). Any table mixing them is wrong.

## Methods

| Method | Reference | Setup source | What it contributes |
|---|---|---|---|
| `mean-baseline` | — | default | the floor: PC = 0 by construction |
| `eisenthal2006` | Eisenthal et al. 2006 | adapted | geometry + symmetry, KNN/ridge ensemble |
| `kagian2008` | Kagian et al. 2008 | adapted | all pairwise distances, selection, SVR |
| `fan2012` | Fan et al. 2012 | adapted | ratios not lengths, polynomial fit |
| `gan2014` | Gan et al. 2014 | adapted | features from unlabelled faces |
| `cnn-resnet18` | Liang et al. 2018 | **paper** | CNN regression baseline |
| `pi-cnn` | Xu et al. 2017 | adapted | whole face + psychology-motivated regions |
| `ldl-ren2017` | Ren & Geng 2017 | adapted | predict the rating *distribution* |
| `cnn-resnext50` | Liang et al. 2018 | **paper** | the SCUT-FBP5500 best backbone |
| `r3cnn` | Lin et al. 2019/2022 | adapted | regression guided by relative ranking |
| `aanet` | Lin et al. 2019 | **paper** | attributes modulate the features |
| `comboloss` | Xu & Xiang 2020 | **paper** | regression + classification + expectation |
| `uol` | Liang et al. 2024 | adapted | ordinal scale with predicted uncertainty |
| `fpem` | Li et al. 2025 | adapted | three-stream cross-attention fusion |
| `transfbp` | Boukhari & Dornaika 2026 | adapted | ViT CLS attends over patch tokens |

`methods/setups.py` holds every method's training configuration, the paper's
own words where the setup is published, and what changed for this dataset.

**Before any result was produced, every entry was audited against its source
publication.** That audit found six defects -- two methods that could not run
at all, three that were silently wrong, and one false claim in the setup file.
It is written up in `docs/IMPLEMENTATION_VS_PAPERS.md`, which is the document
to read before trusting any number here.

## Where the setups come from

**4 of 15 are quoted verbatim** from their papers. The rest are `adapted` —
the paper's setup could not transfer, or was not retrievable — and each says so
in its own entry.

Examples of what is quoted:

> **ComboLoss (Xu & Xiang 2020)** — "The learning rate starts from 0.01 and is
> divided by 10 per 50 epochs. Weight decay and batch size are set as 0.001 and
> 64… trained via SGD with 0.9 momentum for 200 epochs… color jittering and
> random rotation are applied for data augmentation."

> **AaNet (Lin et al. 2019)** — "SGD with a batch size of 32, a momentum of
> 0.9… the learning rate is increased from 0 to a peak value of 0.01 in a
> warm-up schedule of 2K iterations, then decreased to 0, linearly, in 18K
> iterations… For ResNet-18, we set the peak learning rate and weight decay as
> 0.1 and 1e-4."

> **SCUT-FBP5500 (Liang et al. 2018)** — "Each raw RGB image was resized as
> 256×256, and a 224×224 random crop was sent to ResNeXt… initialized by
> pretrained CNN models of ImageNet and updated by mini-batch SGD."

Each method trains under **its own** paper's optimiser, schedule and
augmentation. A benchmark that imposes one global config measures the
benchmark author's tuning rather than the literature.

## Deliberate deviations, applied to every method

**1. Early stopping on validation replaces fixed schedules.** Published setups
could use fixed epoch counts because 5-fold cross-validation averaged the
variance away. A single fixed split cannot, and a fixed schedule here would
report whatever the last epoch happened to produce.

**2. Epoch counts are capped at 60.** An earlier version of this benchmark
rescaled epochs to match each paper's *optimiser step* count. That was wrong:
ComboLoss's 200 epochs over 4,400 images shows each image 200 times. Matching its step count instead -- 200 x ceil(4400/64) = 13,800 -- on 1,399 training images at batch 64 (ceil(1399/64) = 22 steps per epoch) would take ~627 epochs, showing each image **627** times: more overfitting, not less. Overfitting tracks epochs, not steps.

**3. L1 rather than L2 for the regression baselines.** MEBeauty's labels run
1–10 where SCUT's run 1–5, so a squared penalty puts roughly four times the
weight on the same relative error, letting the noisiest labels dominate.

## Guarantees the harness enforces

**No method can read test labels.** After fitting, every method re-predicts
with the test labels shuffled; the run fails if its output moves. This is the
one bug that would silently invalidate an entire table, so it is checked rather
than trusted.

**Per-image predictions are saved**, not just summaries. Paired significance
testing between two methods needs their predictions on the same images;
aggregate scores cannot support it afterwards.

**Confidence intervals are reported.** With n = 434, two methods differing by
0.01 PC are not distinguishable. `paired_bootstrap_difference` compares two
methods on the same images, which is far more sensitive than comparing two
independent intervals.

**Environment is recorded** with every result: git commit, Python, platform,
torch version, device.

## Running it

```bash
# Classical methods only — CPU, seconds, no torch needed
uv run python scripts/benchmark/run.py --methods eisenthal2006,kagian2008,fan2012

# Everything, each under its own paper's setup
uv run python scripts/benchmark/run.py --methods all

# Resume: leave finished methods alone. The full table takes longer than one
# session, so this is the normal way to run it.
uv run python scripts/benchmark/run.py --methods all --skip-existing

# Smoke test: override every epoch count (recorded in the result)
uv run python scripts/benchmark/run.py --methods all --epochs 2
```

Outputs to `reports/benchmark/`: one JSON per method (metrics, setup,
provenance, environment), an `.npz` of per-image predictions, and `summary.md`.
`summary.md` is rebuilt from every result file on disk, not from the current
invocation, so a resumed run produces the whole table rather than the last
chunk of it.

**`environment` records whether the working tree was dirty.** A commit hash on
its own is a false promise when the code has uncommitted changes: the result
did not come from that commit and cannot be reproduced by checking it out.
`reproducible_from_commit: false` means exactly that.

## Honest limits

**Reimplementations, not reproductions.** Eleven of fifteen are `adapted`. A
reimplementation scoring below its published number is evidence about *this
implementation on this dataset*, not about the original work.

**FPEM is architecture-only.** Its three pretrained encoders (Swin, FaceNet,
CLIP-aesthetic) are replaced by projections of one shared backbone, so the
external knowledge that motivates the method is absent. Expect it to
substantially underperform its published figure.

**TransFBP omits TransMix.** The paper's attention-guided augmentation would
make its training loop structurally different from every other entry, so it is
not implemented — meaning this entry under-represents the method.

**Gan 2014 has no external corpus.** Its self-taught stage runs on this
dataset's own training images without labels, which removes the method's main
advantage: seeing far more faces than are labelled.

**`aanet` conditions on gender and ethnicity by construction.** On a
multi-ethnic beauty dataset that belongs in the fairness analysis, not a
footnote.

## Status

Executed: `mean-baseline`, `eisenthal2006`, `kagian2008`, `fan2012`, and a
2-epoch smoke test of `cnn-resnet18`.

The eleven deep methods are **written and wired but not yet run** under their
published setups. No numbers for them exist.
