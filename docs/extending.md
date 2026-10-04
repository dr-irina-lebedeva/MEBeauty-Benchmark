# Extending the benchmark

## Using your own dataset

Nothing here is specific to MEBeauty except the defaults. A dataset works if
one row is one face, with an image column and a numeric label. Describe it in
YAML:

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

Missing optional columns disable the methods that need them, with a readable
error rather than a silent fallback — a distribution method handed zeros would
report a plausible bad score, which reads as *weak method* instead of
*misconfigured run*.

## Adding a method

Decorate a class. There is no second list to keep in sync:

```python
from fbp_benchmark.registry import register


@register("my-method", era="deep", reference="Author et al., 2027", trainable=True)
class MyMethod:
    def fit(self, protocol): ...
    def predict(self, split): ...  # -> Prediction(scores=...)
```

`trainable=True` means the method is built from an entry in `setups.py`, so
its schedule is recorded rather than improvised.

## Layout

```
src/fbp_benchmark/
  data.py            load from the Hub; DatasetSpec is the only place a dataset is described
  registry.py        @register - name, era, reference, requirements
  runner.py          train, predict, score, write
  metrics.py         point and distribution measures
  setups.py          each method's schedule, quoted from its paper
  report.py          generated tables for README.md and docs/
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
docs/                everything the README links to
```

## Development

```bash
uv sync
make check           # lint + fast tests
make test-slow       # end-to-end; downloads pretrained weights
make results         # regenerate the tables in README.md and docs/
```

Three generated blocks are kept in sync with `results/` and the registry, and
CI fails if any has drifted:

| File | Block | Content |
|---|---|---|
| `README.md` | `RESULTS` | CV table, plus the baseline and best method per era |
| `docs/results.md` | `RESULTS` | every method, both protocols |
| `docs/methods.md` | `METHODS` | the method catalogue |

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contribution process.
