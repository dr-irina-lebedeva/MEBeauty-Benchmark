# Results

Every method, both protocols. The README carries a condensed version of these
tables; this is the full set.

The tables below are generated from `results/` — run `make results` rather
than editing them. See [evaluation-protocol.md](evaluation-protocol.md) for
which numbers to cite.

![Pearson correlation by method, grouped by era](figures/results.png)

<!-- RESULTS:START -->

### Held-out split

`dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction` config `fbp_extended`, label `beauty_score`, seed 0 — train 1,962 / val 250 / test 250. The runs were recorded under the previous id `dr-irina-lebedeva/MEBeauty`, which is the same dataset.

| Method | Era | PC | SROCC | MAE | RMSE | Time |
|---|---|---|---|---|---|---|
| `mean-baseline` | baseline | -0.0000 | 0.0000 | 0.9107 | 1.1512 | 0s |
| `eisenthal2006` | classical | 0.3769 | 0.3928 | 0.8549 | 1.0672 | 1s |
| `kagian2008` | classical | 0.3694 | 0.3849 | 0.8989 | 1.1091 | 0s |
| `fan2012` | classical | 0.3115 | 0.3295 | 0.9235 | 1.1416 | 0s |
| `transfbp` | deep | 0.8136 | 0.8203 | 0.5162 | 0.6742 | 16s |
| `rw-ldl` | deep | 0.7643 | 0.7540 | 0.5749 | 0.7429 | 612s |
| `comboloss` | deep | 0.7607 | 0.7571 | 0.5894 | 0.7504 | 2560s |
| `rw-ldl-kl` | deep | 0.7530 | 0.7411 | 0.6012 | 0.7598 | 212s |
| `rw-ldl-noweight` | deep | 0.7470 | 0.7302 | 0.5984 | 0.7666 | 283s |
| `uol` | deep | 0.7353 | 0.7334 | 0.6130 | 0.7924 | 6735s |
| `fpem` | deep | 0.7262 | 0.7165 | 0.6246 | 0.7913 | 463s |
| `ldl-ren2017` | deep | 0.7202 | 0.7137 | 0.6381 | 0.7987 | 192s |
| `pi-cnn` | deep | 0.7087 | 0.7127 | 0.6374 | 0.8138 | 1975s |
| `r3cnn` | deep | 0.6928 | 0.6988 | 0.6603 | 0.8411 | 350s |
| `cnn-resnext50` | deep | 0.6672 | 0.6457 | 0.6635 | 0.8611 | 1289s |
| `gan2014` | deep | 0.6342 | 0.6353 | 0.7160 | 0.9060 | 88s |
| `aanet` | deep | 0.4969 | 0.5786 | 0.7587 | 1.0133 | 286s |
| `cnn-resnet18` | deep | 0.4065 | 0.4817 | 0.8336 | 1.0609 | 206s |
| `rater-dinov2` | foundation | 0.7798 | 0.7734 | 0.5662 | 0.7206 | 1885s |
| `dinov2-partial` | foundation | 0.7711 | 0.7669 | 0.5658 | 0.7373 | 1408s |
| `vit-fbp` | foundation | 0.7541 | 0.7506 | 0.6007 | 0.7661 | 2527s |
| `xattn-vit` | foundation | 0.7412 | 0.7392 | 0.6109 | 0.7727 | 1548s |
| `dinov2-linear` | foundation | 0.7223 | 0.7319 | 0.6476 | 0.8267 | 857s |

### 5-fold cross-validation

Every image is tested exactly once across the folds, so this is the comparison to trust — the held-out split has only 250 test images, too few to separate methods within about 0.04 correlation.

| Method | Era | PC (mean ± sd) | SROCC (mean ± sd) | MAE (mean ± sd) | RMSE (mean ± sd) |
|---|---|---|---|---|---|
| `transfbp` | deep | 0.8081 ± 0.0103 | 0.8118 ± 0.0118 | 0.5265 ± 0.0079 | 0.6916 ± 0.0207 |
| `dinov2-partial` | foundation | 0.8035 ± 0.0136 | 0.8022 ± 0.0104 | 0.5371 ± 0.0061 | 0.7056 ± 0.0231 |
| `rater-dinov2` | foundation | 0.7959 ± 0.0196 | 0.7932 ± 0.0183 | 0.5562 ± 0.0182 | 0.7243 ± 0.0291 |
| `xattn-vit` | foundation | 0.7801 ± 0.0105 | 0.7779 ± 0.0176 | 0.5707 ± 0.0230 | 0.7394 ± 0.0151 |
| `rw-ldl` | deep | 0.7766 ± 0.0197 | 0.7753 ± 0.0152 | 0.5777 ± 0.0206 | 0.7464 ± 0.0362 |
| `vit-fbp` | foundation | 0.7730 ± 0.0094 | 0.7705 ± 0.0115 | 0.5776 ± 0.0204 | 0.7466 ± 0.0239 |
| `comboloss` | deep | 0.7506 ± 0.0182 | 0.7520 ± 0.0174 | 0.5999 ± 0.0132 | 0.7835 ± 0.0169 |

<!-- RESULTS:END -->

**The top of the table is not resolved.** `transfbp` leads both protocols, but
its margin over `dinov2-partial` does not survive the larger sample: on the
250-image held-out split the gap is +0.043 correlation (95% CI 0.006 to 0.081,
p = 0.02), while pooled across all five folds it falls to +0.007 (95% CI
−0.005 to 0.018, p = 0.26, n = 2,462). Read `transfbp` and `dinov2-partial` as
tied at the top, and treat the held-out margin as an artefact of 250 images.

Further down the table the two protocols can disagree outright:
`dinov2-partial` and `rater-dinov2` are indistinguishable on the held-out
split (p = 0.48) but separate under CV with the sign reversed (p < 0.001).
This is why the CV table is the one to cite.
