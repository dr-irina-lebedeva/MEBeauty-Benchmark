# FBP-Benchmark

A reproducible benchmark for **facial beauty prediction** — twenty methods spanning
twenty years, from hand-crafted facial geometry to foundation models, all trained and
scored under one protocol.

No download script, no preprocessing, no `data/` directory to populate. The dataset
streams from the Hugging Face Hub:

```bash
pip install "fbp-benchmark[all]"
huggingface-cli login          # the dataset is gated; approval is automatic
fbp-benchmark run --method cnn-resnet18
```

**Dataset:** [dr-irina-lebedeva/MEBeauty](https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty) ·
**Original release:** [fbplab/MEBeauty-database](https://github.com/fbplab/MEBeauty-database) ·
**Paper:** [Neural Computing and Applications (2022)](https://doi.org/10.1007/s00521-021-06535-0)

---

## Why this exists

Facial beauty prediction has a reproducibility problem. Published numbers come from
different datasets, different splits, different label definitions and different training
schedules — and then get printed in the same table. This repository fixes everything
except the method.

Three things are held constant:

- **One dataset, one split.** Loaded from the Hub, not rebuilt locally. Photographs of
  the same person never appear in two splits.
- **One label.** `beauty_score`, named in every result file.
- **One evaluation.** The harness owns the metrics; no method can define its own.

One thing is deliberately *not* held constant: each method trains under **its own
paper's schedule** (`setups.py`), not a shared config. A benchmark that retunes every
method measures the benchmark author's tuning rather than the literature. Where a
published setting could not transfer, the deviation is recorded in that method's entry
with the paper's own words beside it.

## The methods

```bash
fbp-benchmark list
```

| Era | Methods | What they are |
|---|---|---|
| **classical** | `eisenthal2006`, `kagian2008`, `fan2012` | landmark geometry + a shallow regressor |
| **deep** | `gan2014`, `cnn-resnet18`, `cnn-resnext50`, `pi-cnn`, `ldl-ren2017`, `r3cnn`, `aanet`, `comboloss`, `uol`, `fpem` | CNNs fine-tuned end to end |
| **foundation** | `transfbp`, `dinov2-linear`, `dinov2-partial` | large pretrained backbones, frozen or lightly adapted |
| **proposed** | `rw-ldl` and two ablations | reliability-weighted label-distribution learning |
| **baseline** | `mean-baseline` | predicts the training mean — the floor every method must clear |

Most entries are **reimplementations**: they preserve the published mechanism, not the
original weights or feature extractors. A low score is evidence about *this
implementation on this dataset*, not a verdict on the original work. Each class
docstring says what was substituted.

## Running it

```bash
fbp-benchmark run                          # every method, holdout split
fbp-benchmark run --era classical          # one era
fbp-benchmark run --method rw-ldl          # one method
fbp-benchmark run --protocol cv --fold 0   # 5-fold cross-validation
fbp-benchmark report                       # render results/ as a leaderboard
```

Results land in `results/<method>.json` with per-image predictions beside them. Every
file records the git commit, whether the tree was modified, library versions and the
device — so a number can be reproduced, or knowingly discounted.

From Python:

```python
from fbp_benchmark import load_protocol, run

protocol = load_protocol()  # MEBeauty, from the Hub
result = run("cnn-resnet18", protocol)
print(result.metrics)  # {'PC': ..., 'SROCC': ..., 'MAE': ...}
```

## Using your own dataset

Nothing here is specific to MEBeauty except the defaults. A dataset works if one row is
one face, with an image column and a numeric label. Describe it in YAML:

```yaml
# configs/my_dataset.yaml
repo_id: your-org/your-dataset
config: default
image_column: image
label_column: beauty_score
score_range: [1.0, 5.0]     # SCUT-style 1-5 rather than MEBeauty's 1-10
distribution_column: null   # no histogram -> LDL methods are skipped
landmark_column: null       # no landmarks -> classical methods are skipped
```

```bash
fbp-benchmark run --dataset configs/my_dataset.yaml
```

Missing optional columns disable the methods that need them, with a readable error
rather than a silent fallback — a distribution method handed zeros would report a
plausible bad score, which reads as *weak method* instead of *misconfigured run*.

## Adding a method

Decorate a class. There is no second list to keep in sync:

```python
from fbp_benchmark.registry import register


@register("my-method", era="deep", reference="Author et al., 2027", trainable=True)
class MyMethod:
    def fit(self, protocol): ...
    def predict(self, split): ...  # -> Prediction(scores=...)
```

`trainable=True` means the method is built from an entry in `setups.py`, so its
schedule is recorded rather than improvised.

## What the harness enforces

- **No test-label leakage.** Every method runs twice, the second time with the test
  labels shuffled. If predictions move, the run fails. This is the single mistake that
  would invalidate a whole table, so it is checked rather than trusted.
- **Distribution metrics only when they mean something.** Scored only if the method
  predicted a distribution *and* the dataset ships a real histogram — never against one
  reconstructed from a mean.
- **Era-grouped ranking.** The leaderboard ranks within era. Putting a 2006 geometric
  regressor on the same line as a fine-tuned transformer invites a conclusion neither
  supports.

## Layout

```
src/fbp_benchmark/
  data.py            load from the Hub; DatasetSpec is the only place a dataset is described
  registry.py        @register - name, era, reference, requirements
  runner.py          train, predict, score, write
  metrics.py         point and distribution measures
  setups.py          each method's schedule, quoted from its paper
  reproducibility.py seeding and environment capture
  methods/
    base.py          the Method interface and the mean baseline
    training.py      shared DataLoader / early-stopping machinery
    classical.py     landmark geometry
    deep.py          the CNN era
    foundation.py    large pretrained backbones
    proposed.py      reliability-weighted LDL
configs/             dataset specifications
results/             one JSON per method
```

## Development

```bash
uv sync
make check           # lint + fast tests
make test-slow       # end-to-end; downloads pretrained weights
```

## Citation

```bibtex
@article{lebedeva2022mebeauty,
  title   = {MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author  = {Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal = {Neural Computing and Applications},
  volume  = {34},
  number  = {17},
  pages   = {14169--14183},
  year    = {2022},
  doi     = {10.1007/s00521-021-06535-0}
}
```

## Licence and scope

Code is MIT. **The dataset is not** — it is for non-commercial academic research on
facial attractiveness assessment only, and specifically not for face recognition,
identification or any biometric use. See the
[dataset card](https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty) for the full
terms, which you accept when requesting access.

Attractiveness ratings are subjective opinions of the people who gave them. A model
trained here predicts what those raters said; it does not measure a property of anyone
shown.

**Maintainer:** Irina Lebedeva, PhD — dr.irina.lebedeva@gmail.com ·
[irina-lebedeva.com](https://irina-lebedeva.com/) ·
[LinkedIn](https://www.linkedin.com/in/ailina/)
