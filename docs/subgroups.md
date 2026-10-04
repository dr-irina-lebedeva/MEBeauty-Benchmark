# Subgroup performance

Per-group metrics for the published reference model, `dinov2-partial`. The
headline is in its [model card](https://huggingface.co/dr-irina-lebedeva/MEBeauty-FBP-models);
this is the full analysis.

Reproduce everything here with:

```bash
uv run python scripts/subgroups.py --method dinov2-partial
```

It reads the committed prediction dumps, so no retraining is involved. Pass
`--method` to analyse any other method with predictions in `results/`.

## Method

Cross-validation pools the **out-of-fold** predictions: every one of the 2,462
images is scored exactly once, by the fold that did not train on it. That is
the figure to cite. Confidence intervals are 10,000 bootstrap resamples of the
images within each group, reported as 2.5th/97.5th percentiles. Gap intervals
resample the two groups independently.

Metrics come from `fbp_benchmark.metrics`; the script only groups and
resamples.

## By ethnicity

| Ethnicity | n | PC | PC 95% CI | MAE | MAE 95% CI | RMSE |
|---|---|---|---|---|---|---|
| `mideastern` | 287 | 0.8446 | [0.7723, 0.8934] | 0.5145 | [0.4634, 0.5728] | 0.7014 |
| `indian` | 289 | 0.8341 | [0.7952, 0.8664] | 0.5139 | [0.4691, 0.5613] | 0.6498 |
| `asian` | 342 | 0.7903 | [0.7451, 0.8314] | 0.5808 | [0.5254, 0.6398] | 0.7935 |
| `hispanic` | 291 | 0.7801 | [0.7115, 0.8335] | 0.5062 | [0.4575, 0.5574] | 0.6716 |
| `caucasian` | 964 | 0.7648 | [0.7237, 0.7992] | 0.5401 | [0.5126, 0.5684] | 0.7024 |
| `black` | 289 | 0.7231 | [0.6613, 0.7768] | 0.5520 | [0.5034, 0.6024] | 0.7005 |
| **all** | **2,462** | **0.8008** | [0.7808, 0.8185] | **0.5371** | [0.5195, 0.5555] | **0.7060** |

- largest PC gap: `mideastern` vs `black` = **+0.1214**, 95% CI [+0.0324,
  +0.2019] — **excludes zero**
- largest MAE gap: `hispanic` vs `asian` = −0.0746, 95% CI [−0.1507, +0.0015] —
  includes zero

The correlation gap between `mideastern` and `black` faces is the one
difference in this analysis that survives its confidence interval. `black`
faces are ordered least well of the six groups under both protocols.

## By gender

| Gender | n | PC | PC 95% CI | MAE | MAE 95% CI | RMSE |
|---|---|---|---|---|---|---|
| `female` | 1,303 | 0.7474 | [0.7088, 0.7817] | 0.5554 | [0.5308, 0.5815] | 0.7302 |
| `male` | 1,159 | 0.7215 | [0.6869, 0.7534] | 0.5164 | [0.4918, 0.5420] | 0.6778 |
| **all** | **2,462** | **0.8008** | [0.7809, 0.8188] | **0.5371** | [0.5190, 0.5552] | **0.7060** |

- largest PC gap: `female` vs `male` = +0.0258, 95% CI [−0.0252, +0.0754] —
  includes zero
- largest MAE gap: `male` vs `female` = **−0.0390**, 95% CI [−0.0750, −0.0029]
  — **excludes zero**, marginally

Absolute error is slightly lower on `male` faces, but ranking quality is
indistinguishable between the two.

## Held-out split, by group

The same breakdown for the published checkpoint's held-out run. **Indicative
only**: 250 test images split six ways leaves 28–98 per group, too few for a
stable correlation. No intervals are given, because at these sizes they would
span most of the plausible range.

| Group | n | PC | MAE | RMSE |
|---|---|---|---|---|
| `asian` | 35 | 0.7970 | 0.5975 | 0.7462 |
| `black` | 29 | 0.6845 | 0.5987 | 0.8032 |
| `caucasian` | 98 | 0.7064 | 0.5886 | 0.7701 |
| `hispanic` | 30 | 0.7819 | 0.5288 | 0.6796 |
| `indian` | 30 | 0.8965 | 0.4784 | 0.6360 |
| `mideastern` | 28 | 0.7546 | 0.5459 | 0.6971 |
| `female` | 130 | 0.6897 | 0.6139 | 0.7816 |
| `male` | 120 | 0.6771 | 0.5138 | 0.6861 |
| all | 250 | 0.7711 | 0.5658 | 0.7373 |

It agrees with the cross-validated table on direction — `black` lowest,
`indian` high — which is reassuring but not independent evidence: these 250
images are a subset of the 2,462.

## Reading these numbers

**Subgroup sizes differ by 3.4×.** `caucasian` has 964 images, `mideastern`
287. Intervals are correspondingly wider for the smaller groups, so a group
showing no gap may simply be too small to reveal one. Absence of a significant
difference is not evidence of parity.

**Per-group correlations are lower than the pooled figure, and that is an
artefact.** Every per-gender PC (0.7474, 0.7215) sits below the pooled 0.8008.
Splitting the data removes between-group variation that the pooled correlation
benefits from — classic range restriction. Compare groups with each other,
never with the pooled number.

**These intervals describe sampling noise only.** They say nothing about
whether a group's *labels* are themselves biased. If raters scored one group
with more disagreement, or systematically higher or lower, that shows up as
label noise the model is being asked to reproduce, not as anything these
confidence intervals can detect. Rater demographics are known for only 10 of
593 raters, so that question cannot be answered from this dataset.

**Six categories are a coarse description of people.** The groups are the
dataset's own single-label annotations. They do not capture mixed heritage, and
they are not self-identified. Treat them as a rough partition for error
analysis, not as ground truth about identity.
