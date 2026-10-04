# Reproducing the results

Every published number came from these commands. Results carry the commit that
produced them, so a run can be reproduced or knowingly discounted.

```bash
git clone https://github.com/dr-irina-lebedeva/MEBeauty-Benchmark
cd MEBeauty-Benchmark
uv sync --locked --all-extras --dev     # or: pip install -e ".[all]"

hf auth login                           # dataset is gated, approval automatic

# Held-out table: every method, its own paper's schedule
fbp-benchmark run --out results

# Cross-validation table: the comparison to trust
for fold in 0 1 2 3 4; do
  fbp-benchmark run --method dinov2-partial --protocol cv --fold $fold \
    --out results/cv/fold$fold
done

fbp-benchmark report                    # render, or --update-readme
```

Two methods need a richer dataset config than the default:

```bash
# rw-ldl and rater-dinov2 need the individual per-rater ratings
fbp-benchmark run --method rater-dinov2 --dataset configs/mebeauty_rater_aware.yaml
```

**What to expect.** The full held-out sweep took ~5.5 hours on an Apple
M-series laptop; `uol` alone is 2h20m. Deep methods are seeded, and most
reproduce exactly given the same commit, torch version and device; across a
different device (CUDA vs MPS vs CPU) expect third-decimal differences.
`aanet` and `uol` are the exceptions — they do not reproduce exactly even at a
fixed seed, because early-stopping epoch selection amplifies MPS
nondeterminism.

## Using the published weights

```python
from fbp_benchmark import load_pretrained, predict

model = load_pretrained("dinov2-partial")
score = predict(model, "face.jpg")  # 1-10
```

`load_pretrained` fetches `config.json` and `<method>.pt` from
[dr-irina-lebedeva/MEBeauty-FBP-models](https://huggingface.co/dr-irina-lebedeva/MEBeauty-FBP-models)
and loads them into the registered method. `predict` applies the evaluation
preprocessing — resize 256, centre crop, ImageNet statistics, test-time flip
averaging, clipped to the score range — so a score matches the published
metrics. Pass `weights=` a local path to skip the download.

## From Python

```python
from fbp_benchmark import load_protocol, run

protocol = load_protocol()
result = run("dinov2-partial", protocol)
print(result.metrics)  # {'PC': ..., 'SROCC': ..., 'MAE': ..., 'RMSE': ...}
```

Images and labels stream from the Hub and are cached by `datasets` (about
750 MB on first use).

## Command-line options

```bash
fbp-benchmark list                                    # the method catalogue
fbp-benchmark run --method dinov2-partial             # train and evaluate one method
fbp-benchmark run --era classical                     # an entire era
fbp-benchmark run --method comboloss --protocol cv --fold 0
fbp-benchmark report                                  # render results/ as a table
```

Each run writes `results/<method>.json` with its metrics, the protocol it used
and the commit that produced it, plus per-image predictions as `.npz`. Change
the destination with `--out`.

| Option | Purpose |
|---|---|
| `--protocol cv --fold N` | 5-fold cross-validation instead of the held-out split |
| `--epochs 1` | smoke test; overrides every schedule and is recorded in the result |
| `--dataset configs/mebeauty_rater_aware.yaml` | evaluate on a different dataset or config |
| `--save-weights DIR` | export trained weights |
| `--seed N` | change the seed (default 0) |

The classical methods and the baseline run on CPU in seconds; the deep and
foundation methods assume a GPU. `uol` is the slowest at roughly two hours.

## Trained models

Weights are saved on request and can be published to the Hub alongside the
dataset:

```bash
fbp-benchmark run --method dinov2-partial --save-weights checkpoints
hf upload dr-irina-lebedeva/MEBeauty-FBP-models checkpoints/
```

A checkpoint stores each module's `state_dict` keyed by name (`backbone`,
`head`, and any method-specific parts), so it reloads without needing this
package's internals. Methods with no tensors — the classical regressors and
the baseline — write nothing rather than an empty file.

> **Published weights.** `dinov2-partial` is published as the reference model
> at [dr-irina-lebedeva/MEBeauty-FBP-models](https://huggingface.co/dr-irina-lebedeva/MEBeauty-FBP-models),
> with loading code in its model card. It was retrained with `--save-weights`
> and reproduced the published metrics bit-exactly.
>
> The other methods' weights were not retained: that sweep predated
> `--save-weights`, and re-running it is ~5.5 hours of compute. Reproduce
> locally with the commands above, or open an issue if more hosted weights
> would help you.
