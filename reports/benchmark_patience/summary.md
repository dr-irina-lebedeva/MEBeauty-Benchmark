# MEBeauty benchmark-v1 results

Not comparable to published SCUT-FBP5500 or MEBeauty-2022 numbers:
different images, labels and splits. Entries marked *reimpl.* preserve
the method's mechanism but not its exact features or weights -- see
`docs/IMPLEMENTATION_VS_PAPERS.md` for what was substituted and why.

20 of 20 methods have results.

| Method | Reference | PC | PC 95% CI | SROCC | MAE | RMSE | setup | epochs | train (s) |
|---|---|---|---|---|---|---|---|---|---|
| `rw-ldl-noweight` | This work | 0.8091 | [0.771, 0.842] | 0.7809 | 0.5703 | 0.7208 | default | 30 | 476 |
| `rw-ldl` | This work | 0.8089 | [0.769, 0.841] | 0.7837 | 0.5633 | 0.7184 | default | 30 | 484 |
| `dinov2-partial` | DINOv2 (Oquab et al. 2024) | 0.8026 | [0.765, 0.836] | 0.7757 | 0.5837 | 0.7337 | adapted | 20 | 1966 |
| `rw-ldl-kl` | This work | 0.8012 | [0.761, 0.834] | 0.7778 | 0.5674 | 0.7306 | default | 30 | 477 |
| `comboloss` | Xu & Xiang 2020 | 0.7980 | [0.760, 0.831] | 0.7783 | 0.5738 | 0.7380 | paper | 45 | 2280 |
| `dinov2-linear` | DINOv2 (Oquab et al. 2024) | 0.7870 | [0.752, 0.821] | 0.7727 | 0.6110 | 0.7639 | adapted | 40 | 2547 |
| `fpem` | Li et al. 2025 | 0.7843 | [0.743, 0.822] | 0.7622 | 0.6132 | 0.7634 | paper | 42 | 764 |
| `uol` | Liang et al. 2024 | 0.7835 | [0.740, 0.821] | 0.7612 | 0.5944 | 0.7607 | paper | 32 | 8124 |
| `transfbp` | Boukhari & Dornaika 2026 | 0.7759 | [0.732, 0.815] | 0.7542 | 0.6173 | 0.7781 | adapted | 25 | 4164 |
| `ldl-ren2017` | Ren & Geng 2017 | 0.7733 | [0.729, 0.811] | 0.7456 | 0.6051 | 0.7755 | adapted | 30 | 475 |
| `r3cnn` | Lin et al. 2019/2022 | 0.7619 | [0.721, 0.800] | 0.7364 | 0.6390 | 0.7958 | adapted | 40 | 608 |
| `cnn-resnext50` | Liang et al. 2018 | 0.7100 | [0.660, 0.759] | 0.6817 | 0.6796 | 0.8703 | paper | 26 | 1392 |
| `pi-cnn` | Xu et al. 2017 | 0.6995 | [0.633, 0.756] | 0.6784 | 0.6788 | 0.8732 | adapted | 40 | 2213 |
| `cnn-resnet18` | Xie et al. 2015 / MEBeauty 2022 | 0.6873 | [0.629, 0.739] | 0.6599 | 0.7213 | 0.9151 | paper | 40 | 634 |
| `gan2014` | Gan et al. 2014 | 0.6305 | [0.565, 0.692] | 0.6158 | 0.7469 | 0.9589 | adapted | None | 32 |
| `aanet` | Lin et al. 2019 | 0.4660 | [0.384, 0.542] | 0.4675 | 0.8325 | 1.0881 | paper | 40 | 640 |
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
| `rw-ldl` | This work (reliability-weighted LDL) | **default** | adamw | 0.0001 | 32 | 30 | resnet18 |
| `rw-ldl-noweight` | This work (ablation: no precision weighting) | **default** | adamw | 0.0001 | 32 | 30 | resnet18 |
| `rw-ldl-kl` | This work (ablation: no multinomial likelihood) | **default** | adamw | 0.0001 | 32 | 30 | resnet18 |
| `dinov2-linear` | Oquab et al. 2024 (DINOv2) + this benchmark | **adapted** | adamw | 0.001 | 32 | 40 | facebook/dinov2-base |
| `dinov2-partial` | Oquab et al. 2024 (DINOv2) + this benchmark | **adapted** | adamw | 1e-05 | 16 | 30 | facebook/dinov2-base |
| `mean-baseline` | -- | **default** | none | 0.0001 | 32 | 30 | resnet18 |
