# Contributing

## Setup

```bash
uv sync --locked --all-extras --dev
make check
```

## Adding a method

1. Put it in the era module it belongs to — `classical.py`, `deep.py` or
   `foundation.py` — and decorate it:

   ```python
   @register("my-method", era="deep", reference="Author et al., 2027", trainable=True)
   class MyMethod:
       def fit(self, protocol): ...
       def predict(self, split): ...  # -> Prediction(scores=...)
   ```

2. If `trainable=True`, add an entry to `setups.py` recording the schedule and
   where it came from: `source="paper"` with a `quote`, or `source="adapted"`
   with a `deviation` explaining what could not transfer. A test fails if you
   skip this.

3. Declare what the method needs beyond images and a label —
   `requires=("landmarks",)`, `("distributions",)`, `("attributes",)` or
   `("ratings",)`. Without it the method fails deep inside a training loop
   instead of before it starts, part-way through someone's sweep.

## House rules

These exist because each one has already gone wrong here:

- **Do not invent hyperparameters.** They come from the paper, or the
  deviation is recorded. A benchmark that retunes every method measures the
  benchmark author's tuning.
- **Do not let a method define its own evaluation.** Metrics live in
  `metrics.py`. A method returns predictions; the harness scores them.
- **Do not remove the leak check.** Every method is run twice with the test
  labels shuffled. It is the one guard against the mistake that invalidates a
  whole results table.
- **Do not hand-edit the README results.** They are generated —
  `make results` — and CI fails if they drift from `results/`.
- **Do not add dataset preparation code.** The dataset is consumed from the
  Hub as published.

## Before opening a pull request

```bash
make check       # lint, types, fast tests, README freshness
make test-slow   # if you touched a method or the training loop
```

Results committed to `results/` must come from a clean tree — each file
records the commit that produced it, and a dirty tree makes that record a lie.
