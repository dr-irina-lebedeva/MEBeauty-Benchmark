# Methods

Twenty-one methods spanning twenty years, from hand-crafted facial geometry to
foundation models.

```bash
fbp-benchmark list          # names, eras, and what each one needs
```

The table below is generated from the registry — edit `@register(...)` in
`src/fbp_benchmark/methods/`, not this file, then run `make results`.

<!-- METHODS:START -->

**Baseline** — reference points that bound the table

| Method | Paper |
|---|---|
| `mean-baseline` | -- — floor: predicts the training mean |

**Classical** — landmark geometry with a shallow regressor

| Method | Paper |
|---|---|
| `eisenthal2006` | [Eisenthal, Dror & Ruppin 2006, Neural Computation 18(1)](https://doi.org/10.1162/089976606774841602) |
| `fan2012` | [Fan et al. 2012, Pattern Recognition 45(6)](https://doi.org/10.1016/j.patcog.2011.11.024) |
| `kagian2008` | [Kagian et al. 2008, Vision Research 48(2)](https://www.sciencedirect.com/science/article/pii/S0042698907005032) |

**Deep** — convolutional networks trained end to end

| Method | Paper |
|---|---|
| `aanet` | [Lin et al. 2019, IJCAI (AaNet / P-AaNet)](https://doi.org/10.24963/ijcai.2019/119) |
| `cnn-resnet18` | [Liang et al. 2018, ICPR (SCUT-FBP5500 baseline)](https://arxiv.org/abs/1801.06345) |
| `cnn-resnext50` | [Liang et al. 2018, ICPR (SCUT-FBP5500 best backbone)](https://arxiv.org/abs/1801.06345) |
| `comboloss` | [Xu & Xiang 2020, arXiv:2010.10721](https://arxiv.org/abs/2010.10721) |
| `fpem` | [Li et al. 2025, ICCV (FPEM: Face Prior Enhanced Facial Attractiveness Prediction for Live Videos), arXiv:2501.02509](https://arxiv.org/abs/2501.02509) |
| `gan2014` | [Gan et al. 2014, Neurocomputing 144](https://doi.org/10.1016/j.neucom.2014.05.028) — reimplementation; no external unlabelled corpus |
| `ldl-ren2017` | [Ren & Geng 2017, IJCAI](https://www.ijcai.org/proceedings/2017/369) |
| `pi-cnn` | [Xu et al. 2017, ICASSP](https://ieeexplore.ieee.org/document/7952438) |
| `r3cnn` | [Lin, Liang & Jin 2019/2022, IEEE Trans. Affective Computing](https://doi.org/10.1109/TAFFC.2019.2933523) |
| `rw-ldl` | This work (reliability-weighted LDL) — proposed |
| `rw-ldl-kl` | This work (ablation: no multinomial likelihood) — ablation: KL instead of multinomial |
| `rw-ldl-noweight` | This work (ablation: no precision weighting) — ablation: no reliability weighting |
| `transfbp` | [Xu, Jinhai & Yuan 2018, arXiv:1803.07253 (TransFBP)](https://arxiv.org/abs/1803.07253) — frozen face-verification features (InceptionResnetV1, VGGFace2) + Bayesian ridge; no fine-tuning. See licensing note below the table. |
| `uol` | [Liang et al. 2024, arXiv:2409.00603 (Uncertainty-oriented Order Learning)](https://arxiv.org/abs/2409.00603) |

**Foundation** — large pretrained backbones, frozen or lightly adapted

| Method | Paper |
|---|---|
| `dinov2-linear` | [Oquab et al. 2024 (DINOv2) + this benchmark](https://arxiv.org/abs/2304.07193) — frozen backbone, linear head |
| `dinov2-partial` | [Oquab et al. 2024 (DINOv2) + this benchmark](https://arxiv.org/abs/2304.07193) — last blocks unfrozen |
| `rater-dinov2` | proposed in this benchmark — proposed: rater effects on a foundation backbone |
| `vit-fbp` | [Boukhari 2023, IJEETC 13(3) (ViT-FBP)](https://www.ijeetc.com/vol13/IJEETC-V13N3-252.pdf) — plain ViT-B/16, fine-tuned end to end |
| `xattn-vit` | Boukhari & Dornaika 2026 (cross-attention ViT) — ViT-B/16 backbone |

Entries without a link are published in venues with no stable open URL; the citation is given in full. Most entries are **reimplementations** — they preserve the published mechanism, not the original weights or feature extractors, so a score is evidence about this implementation on this dataset rather than a verdict on the original work.

<!-- METHODS:END -->

**Pretrained weights: `transfbp`.** Its backbone is InceptionResnetV1
pretrained on VGGFace2, obtained via
[facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MIT-licensed
code). VGGFace2 was released for non-commercial research and withdrawn by its
authors in 2021. The weights are downloaded by the user at run time and are
not redistributed by this repository.
