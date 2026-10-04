# Evaluation protocol

Which numbers to report, and why.

Two protocols ship, and a result must state which it used.

**Cross-validation is the one to report.** The held-out split has 250 test
images; cross-validation scores all 2,462, and averages over five fits rather
than resting on one. Differences below roughly 0.04 correlation are not
resolvable on 250 images, and the two protocols can disagree: `dinov2-partial`
and `rater-dinov2` are indistinguishable on the held-out split (p = 0.48) but
separate cleanly under CV, with the sign reversed (p < 0.001).

**Name the label.** `beauty_score` corrects for rater leniency and is the
recommended target; `plain_mean_score` is the uncorrected average. They
correlate 0.97 but differ by up to 1.2 on individual images.

**The ceiling is about 0.90.** Roughly 19% of the test-label variance is rater
sampling noise, so a perfect predictor would not reach 1.0. The strongest
method here reaches 0.81.

**Some methods are unstable.** `cnn-resnet18` ranges 0.41-0.71 across four
seeds under its published SGD schedule. `aanet` and `uol` are unstable even at
a *fixed* seed: four runs of `aanet` at seed 0, same commit and same device,
gave PC 0.4969, 0.5270, 0.5448 and 0.5522. Early stopping picks the best epoch
by validation MAE, and that choice is sensitive to MPS reduction
nondeterminism, so a flat validation curve resolves differently run to run.
Schedules are not retuned here, so a single-seed number for an unstable method
is one draw; cross-validation averages five and is the safer figure.

All published results come from commit `f2ee365`. `aanet` and `uol` differ from
earlier published runs not because of a code change — their code paths are
functionally unchanged across PR #4 — but because of that epoch-selection
sensitivity.

**Test differences, do not eyeball them.**
`fbp_benchmark.metrics.paired_bootstrap_difference` returns the difference, a
95% interval and a p-value for any two methods' predictions.

Splits are fixed and grouped so that photographs of the same person never
cross a boundary; the benchmark never re-derives them.

## What the harness guarantees

- **No test-label leakage.** Every method runs twice, the second time with the
  test labels shuffled. If predictions move, the run fails. This is the single
  mistake that would invalidate a whole table, so it is checked rather than
  trusted.
- **Distribution metrics only when they mean something.** Scored only if the
  method predicted a distribution *and* the dataset ships a real histogram —
  never against one reconstructed from a mean.
- **Era-grouped ranking.** The leaderboard ranks within era. Putting a 2006
  geometric regressor on the same line as a fine-tuned transformer invites a
  conclusion neither supports.

## Why this benchmark exists

Facial beauty prediction has a reproducibility problem. Published numbers come
from different datasets, different splits, different label definitions and
different training schedules — and then get printed in the same table. This
repository fixes everything except the method.

Three things are held constant:

- **One dataset, one split.** Loaded from the Hub, not rebuilt locally.
  Photographs of the same person never appear in two splits.
- **One label.** `beauty_score`, named in every result file.
- **One evaluation.** The harness owns the metrics; no method can define its
  own.

One thing is deliberately *not* held constant: each method trains under **its
own paper's schedule** (`setups.py`), not a shared config. A benchmark that
retunes every method measures the benchmark author's tuning rather than the
literature. Where a published setting could not transfer, the deviation is
recorded in that method's entry with the paper's own words beside it.

## Troubleshooting

| | |
|---|---|
| `GatedRepoError` / `401` | accept the dataset terms, then `hf auth login` |
| `needs individual ratings` | that method requires `--dataset configs/mebeauty_rater_aware.yaml` |
| Results differ in the third decimal | expected across CUDA / MPS / CPU **for the same code version**; ordering holds. `aanet` and `uol` vary more than this even at a fixed seed — see above |
