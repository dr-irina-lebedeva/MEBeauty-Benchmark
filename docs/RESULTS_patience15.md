# MEBeauty benchmark-v1 — results

Generated from `benchmark_patience/` by `scripts/benchmark/report.py`. Do not edit by hand.

## Protocol

| | |
|---|---|
| Protocol | `benchmark-v1` |
| Label | `score` — rater-normalised mean, ≥10 ratings/image |
| Images | `cropped_256` |
| Splits | train 1399 / val 185 / test 434 |
| Seed | 0 |
| Device | mps, torch 2.13.0 |
| Commit | `1d02a242` (tree dirty: True) |

**Not comparable to published SCUT-FBP5500 or MEBeauty-2022 numbers.** Different images, labels and splits.

## Results

| # | Method | PC | PC 95% CI | SROCC | MAE | RMSE | epochs | train | setup | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `rw-ldl-noweight` | **0.8091** | [0.771, 0.842] | 0.7809 | 0.5703 | 0.7208 | 30 | 476s | default | ablation of rw-ldl |
| 2 | `rw-ldl` | **0.8089** | [0.769, 0.841] | 0.7837 | 0.5633 | 0.7184 | 30 | 484s | default | **proposed here** |
| 3 | `dinov2-partial` | **0.8026** | [0.765, 0.836] | 0.7757 | 0.5837 | 0.7337 | 20 | 1966s | adapted | NOT a published FBP method |
| 4 | `rw-ldl-kl` | **0.8012** | [0.761, 0.834] | 0.7778 | 0.5674 | 0.7306 | 30 | 477s | default | ablation of rw-ldl |
| 5 | `comboloss` | **0.7980** | [0.760, 0.831] | 0.7783 | 0.5738 | 0.7380 | 45 | 2280s | paper |  |
| 6 | `dinov2-linear` | **0.7870** | [0.752, 0.821] | 0.7727 | 0.6110 | 0.7639 | 40 | 2547s | adapted | NOT a published FBP method |
| 7 | `fpem` | **0.7843** | [0.743, 0.822] | 0.7622 | 0.6132 | 0.7634 | 42 | 764s | paper | architecture only -- no Swin/FaceNet/CLIP/GPT-2 encoders |
| 8 | `uol` | **0.7835** | [0.740, 0.821] | 0.7612 | 0.5944 | 0.7607 | 32 | 8124s | paper | batch-internal ordering, not the paper's MC comparator |
| 9 | `transfbp` | **0.7759** | [0.732, 0.815] | 0.7542 | 0.6173 | 0.7781 | 25 | 4164s | adapted | TransMix augmentation omitted (the paper's contribution) |
| 10 | `ldl-ren2017` | **0.7733** | [0.729, 0.811] | 0.7456 | 0.6051 | 0.7755 | 30 | 475s | adapted | objective only; paper's SLDL is a structural SVM |
| 11 | `r3cnn` | **0.7619** | [0.721, 0.800] | 0.7364 | 0.6390 | 0.7958 | 40 | 608s | adapted | optimiser inherited from the authors' AaNet paper |
| 12 | `cnn-resnext50` | **0.7100** | [0.660, 0.759] | 0.6817 | 0.6796 | 0.8703 | 26 | 1392s | paper |  |
| 13 | `pi-cnn` | **0.6995** | [0.633, 0.756] | 0.6784 | 0.6788 | 0.8732 | 40 | 2213s | adapted | fixed horizontal bands, not landmark-driven boxes |
| 14 | `cnn-resnet18` | **0.6873** | [0.629, 0.739] | 0.6599 | 0.7213 | 0.9151 | 40 | 634s | paper |  |
| 15 | `gan2014` | **0.6305** | [0.565, 0.692] | 0.6158 | 0.7469 | 0.9589 | -- | 32s | adapted | no external unlabelled corpus |
| 16 | `aanet` | **0.4660** | [0.384, 0.542] | 0.4675 | 0.8325 | 1.0881 | 40 | 640s | paper | gates pooled features; paper modulates conv filters |
| 17 | `kagian2008` | **0.4435** | [0.365, 0.521] | 0.4504 | 0.8899 | 1.1347 | -- | 0s | adapted |  |
| 18 | `eisenthal2006` | **0.4316** | [0.357, 0.508] | 0.4233 | 0.8823 | 1.1053 | -- | 1s | adapted |  |
| 19 | `fan2012` | **0.4103** | [0.331, 0.491] | 0.4285 | 0.9301 | 1.1838 | -- | 0s | adapted |  |
| 20 | `mean-baseline` | **0.0000** | [0.000, 0.000] | 0.0000 | 0.9715 | 1.2205 | -- | 0s | default |  |

## Label-distribution metrics

Only methods that predict a full rating distribution. Chebyshev, Clark, Canberra and KL are distances (lower is better); Cosine and Intersection are similarities (higher is better).

| Method | Chebyshev ↓ | Clark ↓ | Canberra ↓ | KL ↓ | Cosine ↑ | Intersection ↑ |
|---|---|---|---|---|---|---|
| `rw-ldl-noweight` | 0.1147 | 1.4677 | 3.5695 | 0.1952 | 0.8945 | 0.7788 |
| `rw-ldl` | 0.1160 | 1.4680 | 3.5849 | 0.1975 | 0.8924 | 0.7762 |
| `rw-ldl-kl` | 0.1155 | 1.4633 | 3.5615 | 0.1956 | 0.8933 | 0.7771 |
| `comboloss` | 0.4026 | 2.5595 | 7.5215 | 1.9428 | 0.6490 | 0.4668 |
| `uol` | 0.1137 | 1.4611 | 3.5690 | 0.1983 | 0.8919 | 0.7738 |
| `ldl-ren2017` | 0.1192 | 1.4986 | 3.6816 | 0.2108 | 0.8846 | 0.7685 |

## Run-to-run stability

Against `benchmark/`. A method whose score moves substantially between runs is reporting variance, not capability.

| Method | this run | other run | Δ |
|---|---|---|---|
| `cnn-resnet18` | 0.6873 | 0.5236 | +0.1637 **unstable** |
| `cnn-resnext50` | 0.7100 | 0.6016 | +0.1084 **unstable** |
| `r3cnn` | 0.7619 | 0.6904 | +0.0715 **unstable** |
| `comboloss` | 0.7980 | 0.7297 | +0.0683 **unstable** |
| `pi-cnn` | 0.6995 | 0.6549 | +0.0446 |
| `transfbp` | 0.7759 | 0.7488 | +0.0271 |
| `aanet` | 0.4660 | 0.4885 | -0.0225 |
| `uol` | 0.7835 | 0.7981 | -0.0146 |
| `fpem` | 0.7843 | 0.7740 | +0.0103 |
| `ldl-ren2017` | 0.7733 | 0.7725 | +0.0008 |
| `eisenthal2006` | 0.4316 | 0.4316 | +0.0000 |
| `fan2012` | 0.4103 | 0.4103 | +0.0000 |
| `gan2014` | 0.6305 | 0.6305 | +0.0000 |
| `kagian2008` | 0.4435 | 0.4435 | +0.0000 |
| `mean-baseline` | 0.0000 | 0.0000 | +0.0000 |

## Significance

Not yet computed. Run `scripts/benchmark/compare.py` before describing any method as better than another: with n = 434 the leading entries sit inside each other's confidence intervals.

## How to read this

- **Eleven of the published methods are reimplementations** (`adapted`). A low score is evidence about this implementation on this dataset, not about the original work. See `docs/IMPLEMENTATION_VS_PAPERS.md`.
- **No implementation has been checked against its published number.** `scripts/benchmark/validate_on_scut.py` exists for that and needs SCUT-FBP5500, which requires accepting a release agreement.
- **`mean-baseline` predicts the training mean.** Its PC is 0 by construction; its MAE is the number every other entry must beat.

