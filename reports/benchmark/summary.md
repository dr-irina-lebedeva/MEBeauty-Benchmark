# MEBeauty benchmark-v1 results

Not comparable to published SCUT-FBP5500 or MEBeauty-2022 numbers:
different images, labels and splits. Entries marked *reimpl.* preserve
the method's mechanism but not its exact features or weights -- see
`docs/IMPLEMENTATION_VS_PAPERS.md` for what was substituted and why.

15 of 15 methods have results.

| Method | Reference | PC | PC 95% CI | SROCC | MAE | RMSE | setup | epochs | train (s) |
|---|---|---|---|---|---|---|---|---|---|
| `uol` | Liang et al. 2024 | 0.7981 | [0.759, 0.833] | 0.7769 | 0.5834 | 0.7354 | paper | 26 | 15071 |
| `fpem` | Li et al. 2025 | 0.7740 | [0.730, 0.814] | 0.7494 | 0.6172 | 0.7730 | paper | 24 | 391 |
| `ldl-ren2017` | Ren & Geng 2017 | 0.7725 | [0.726, 0.810] | 0.7463 | 0.6106 | 0.7777 | adapted | 12 | 196 |
| `transfbp` | Boukhari & Dornaika 2026 | 0.7488 | [0.703, 0.792] | 0.7306 | 0.6460 | 0.8099 | adapted | 7 | 1208 |
| `comboloss` | Xu & Xiang 2020 | 0.7297 | [0.682, 0.772] | 0.7247 | 0.7037 | 0.8770 | paper | 14 | 762 |
| `r3cnn` | Lin et al. 2019/2022 | 0.6904 | [0.638, 0.739] | 0.6564 | 0.7093 | 0.8844 | adapted | 17 | 303 |
| `pi-cnn` | Xu et al. 2017 | 0.6549 | [0.602, 0.706] | 0.6487 | 0.7570 | 0.9308 | adapted | 19 | 1201 |
| `gan2014` | Gan et al. 2014 | 0.6305 | [0.565, 0.692] | 0.6158 | 0.7469 | 0.9589 | adapted | None | 32 |
| `cnn-resnext50` | Liang et al. 2018 | 0.6016 | [0.539, 0.660] | 0.5696 | 0.8296 | 1.0497 | paper | 9 | 541 |
| `cnn-resnet18` | Xie et al. 2015 / MEBeauty 2022 | 0.5236 | [0.450, 0.589] | 0.4659 | 0.8551 | 1.0628 | paper | 11 | 153 |
| `aanet` | Lin et al. 2019 | 0.4885 | [0.407, 0.565] | 0.4985 | 0.8264 | 1.0843 | paper | 11 | 194 |
| `kagian2008` | Kagian et al. 2008 | 0.4435 | [0.365, 0.521] | 0.4504 | 0.8899 | 1.1347 | adapted | None | 0 |
| `eisenthal2006` | Eisenthal et al. 2006 | 0.4316 | [0.357, 0.508] | 0.4233 | 0.8823 | 1.1053 | adapted | None | 1 |
| `fan2012` | Fan et al. 2012 | 0.4103 | [0.331, 0.491] | 0.4285 | 0.9301 | 1.1838 | adapted | None | 0 |
| `mean-baseline` | -- | 0.0000 | [0.000, 0.000] | 0.0000 | 0.9715 | 1.2205 | default | None | 0 |

## Training setups and their provenance

| Method | Reference | Source | Optimizer | LR | Batch | Epochs | Backbone |
|---|---|---|---|---|---|---|---|
| `eisenthal2006` | Eisenthal, Dror & Ruppin 2006, Neural Computation 18(1) | **adapted** | none | 0.0001 | 32 | 30 | resnet18 |
| `kagian2008` | Kagian et al. 2008, Vision Research 48(2) | **adapted** | none | 0.0001 | 32 | 30 | resnet18 |
| `fan2012` | Fan et al. 2012, Pattern Recognition 45(6) | **adapted** | none | 0.0001 | 32 | 30 | resnet18 |
| `gan2014` | Gan et al. 2014, Neurocomputing 133 | **adapted** | adamw | 0.0001 | 32 | 20 | resnet18 |
| `cnn-resnet18` | Liang et al. 2018, ICPR (SCUT-FBP5500 baseline) | **paper** | sgd | 0.01 | 32 | 40 | resnet18 |
| `cnn-resnext50` | Liang et al. 2018, ICPR (SCUT-FBP5500 best backbone) | **paper** | sgd | 0.01 | 32 | 40 | resnext50 |
| `pi-cnn` | Xu et al. 2017, ICASSP | **adapted** | sgd | 0.01 | 32 | 40 | resnet18 |
| `ldl-ren2017` | Ren & Geng 2017, IJCAI | **adapted** | adamw | 0.0001 | 32 | 30 | resnet18 |
| `r3cnn` | Lin, Liang & Jin 2019/2022, IEEE Trans. Affective Computing | **adapted** | sgd | 0.01 | 32 | 40 | resnet18 |
| `aanet` | Lin et al. 2019, IJCAI (AaNet / P-AaNet) | **paper** | sgd | 0.1 | 32 | 40 | resnet18 |
| `comboloss` | Xu & Xiang 2020, arXiv:2010.10721 | **paper** | sgd | 0.01 | 64 | 60 | resnext50 |
| `uol` | Liang et al. 2024, arXiv:2409.00603 (Uncertainty-oriented Order Learning) | **paper** | adamw | 0.0001 | 32 | 60 | vgg16 |
| `fpem` | Li et al. 2025, ICCV (FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos), arXiv:2501.02509 | **paper** | adamw | 5e-05 | 32 | 50 | resnet18 |
| `transfbp` | Boukhari & Dornaika 2026, Cognitive Computation | **adapted** | adamw | 3e-05 | 16 | 25 | vit_b_16 |
| `mean-baseline` | -- | **default** | none | 0.0001 | 32 | 30 | resnet18 |
