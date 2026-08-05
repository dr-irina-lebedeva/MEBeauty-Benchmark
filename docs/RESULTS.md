# MEBeauty benchmark-v1 — results

Generated from `benchmark/` by `scripts/benchmark/report.py`. Do not edit by hand.

## Protocol

| | |
|---|---|
| Protocol | `benchmark-v1` |
| Label | `score` — rater-normalised mean, ≥8 ratings/image (not a published column; see below) |
| Images | `cropped_256` |
| Splits | train 1399 / val 185 / test 434 |
| Seed | 0 |
| Device | mps, torch 2.13.0 |
| Commit | `1d02a242` (tree dirty: True) |

**Not comparable to published SCUT-FBP5500 or MEBeauty-2022 numbers.** Different images, labels and splits.

**These runs predate the v2.1.0 label change.** They target `score`, which the
Hugging Face release does not ship; the release publishes `score_adjusted`
(r = 0.99 with `score`) and `score_mean`. The methods have not been re-run
against a published label, so these figures are internal reference points, not
reproducible benchmark results.

## Results

| # | Method | PC | PC 95% CI | SROCC | MAE | RMSE | epochs | train | setup | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `uol` | **0.7981** | [0.759, 0.833] | 0.7769 | 0.5834 | 0.7354 | 26 | 15071s | paper | batch-internal ordering, not the paper's MC comparator |
| 2 | `fpem` | **0.7740** | [0.730, 0.814] | 0.7494 | 0.6172 | 0.7730 | 24 | 391s | paper | architecture only -- no Swin/FaceNet/CLIP/GPT-2 encoders |
| 3 | `ldl-ren2017` | **0.7725** | [0.726, 0.810] | 0.7463 | 0.6106 | 0.7777 | 12 | 196s | adapted | objective only; paper's SLDL is a structural SVM |
| 4 | `transfbp` | **0.7488** | [0.703, 0.792] | 0.7306 | 0.6460 | 0.8099 | 7 | 1208s | adapted | TransMix augmentation omitted (the paper's contribution) |
| 5 | `comboloss` | **0.7297** | [0.682, 0.772] | 0.7247 | 0.7037 | 0.8770 | 14 | 762s | paper |  |
| 6 | `r3cnn` | **0.6904** | [0.638, 0.739] | 0.6564 | 0.7093 | 0.8844 | 17 | 303s | adapted | optimiser inherited from the authors' AaNet paper |
| 7 | `pi-cnn` | **0.6549** | [0.602, 0.706] | 0.6487 | 0.7570 | 0.9308 | 19 | 1201s | adapted | fixed horizontal bands, not landmark-driven boxes |
| 8 | `gan2014` | **0.6305** | [0.565, 0.692] | 0.6158 | 0.7469 | 0.9589 | -- | 32s | adapted | no external unlabelled corpus |
| 9 | `cnn-resnext50` | **0.6016** | [0.539, 0.660] | 0.5696 | 0.8296 | 1.0497 | 9 | 541s | paper |  |
| 10 | `cnn-resnet18` | **0.5236** | [0.450, 0.589] | 0.4659 | 0.8551 | 1.0628 | 11 | 153s | paper |  |
| 11 | `aanet` | **0.4885** | [0.407, 0.565] | 0.4985 | 0.8264 | 1.0843 | 11 | 194s | paper | gates pooled features; paper modulates conv filters |
| 12 | `kagian2008` | **0.4435** | [0.365, 0.521] | 0.4504 | 0.8899 | 1.1347 | -- | 0s | adapted |  |
| 13 | `eisenthal2006` | **0.4316** | [0.357, 0.508] | 0.4233 | 0.8823 | 1.1053 | -- | 1s | adapted |  |
| 14 | `fan2012` | **0.4103** | [0.331, 0.491] | 0.4285 | 0.9301 | 1.1838 | -- | 0s | adapted |  |
| 15 | `mean-baseline` | **0.0000** | [0.000, 0.000] | 0.0000 | 0.9715 | 1.2205 | -- | 0s | default |  |

## Label-distribution metrics

Only methods that predict a full rating distribution. Chebyshev, Clark, Canberra and KL are distances (lower is better); Cosine and Intersection are similarities (higher is better).

| Method | Chebyshev ↓ | Clark ↓ | Canberra ↓ | KL ↓ | Cosine ↑ | Intersection ↑ |
|---|---|---|---|---|---|---|
| `uol` | 0.1120 | 1.4546 | 3.5455 | 0.1917 | 0.8972 | 0.7776 |
| `ldl-ren2017` | 0.1200 | 1.4871 | 3.6533 | 0.2089 | 0.8841 | 0.7680 |
| `comboloss` | 0.2689 | 2.2436 | 6.3045 | 1.1385 | 0.7226 | 0.5728 |

## Methods that stopped well before their schedule

Early stopping (patience 5) on a 185-image validation split is noisy for runs at high learning rates. An entry that halted in the first 40% of its schedule is reporting where it stopped, not what the method can do.

| Method | epochs run | of | optimizer | lr |
|---|---|---|---|---|
| `transfbp` | 7 | 25 | adamw | 3e-05 |
| `comboloss` | 14 | 60 | sgd | 0.01 |
| `cnn-resnext50` | 9 | 40 | sgd | 0.01 |
| `cnn-resnet18` | 11 | 40 | sgd | 0.01 |
| `aanet` | 11 | 40 | sgd | 0.1 |

## Significance

Not yet computed. Run `scripts/benchmark/compare.py` before describing any method as better than another: with n = 434 the leading entries sit inside each other's confidence intervals.

## How to read this

- **Eleven of the published methods are reimplementations** (`adapted`). A low score is evidence about this implementation on this dataset, not about the original work. See `docs/IMPLEMENTATION_VS_PAPERS.md`.
- **No implementation has been checked against its published number.** `scripts/benchmark/validate_on_scut.py` exists for that and needs SCUT-FBP5500, which requires accepting a release agreement.
- **`mean-baseline` predicts the training mean.** Its PC is 0 by construction; its MAE is the number every other entry must beat.

