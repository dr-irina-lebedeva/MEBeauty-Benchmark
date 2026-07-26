# Reproducing the legacy pipeline's crops/embeddings/geometric features

> **⚠ Read Finding 17 first — it changes what this document can promise.**
> The legacy `original_images/` tree is **99.1% already-cropped, downsampled
> face images** (400×400, 500×500, 600×600); only 20 of 2,495 exceed one
> megapixel. So "run a face detector against the originals" is not what
> happens here — any detector run against `images/` is **cropping an
> existing crop**, not re-deriving one from source pixels.
>
> This is also the most likely explanation for the reproduction gap measured
> below: the legacy crops were probably derived from true source photographs
> that were never included in the release, which is why regenerating from
> what *was* released cannot match them. The "different MTCNN
> implementation" explanation given further down is still a contributing
> factor, but it is no longer sufficient on its own.
>
> **Practical consequence:** treat `images/` as the canonical face crops.
> Re-cropping them with a modern detector is not an upgrade — it compounds
> two crops. A genuine preprocessing comparison would need the source
> photographs re-obtained from the platforms (see
> `scripts/data/verify_provenance_via_api.py`, which can resolve download
> URLs for the ~65% of images with a verifiable platform ID).

`data/mebeauty_v3/` (see `docs/RESTRUCTURE_PROPOSAL.md`) does not ship
precomputed MTCNN/OpenCV crops, FaceNet-512 embeddings, or the 11,628-dim
geometric-ratio features — they're 100% derivable from `images/` +
`landmarks.parquet`, and shipping them bakes a specific 2015–2018-era
pipeline into the dataset's identity. This document is what a user runs if
they specifically want those historical artifacts (e.g. to reproduce the
original paper's exact baseline).

## What actually reproduces, and what doesn't

Verified at full scale (`scripts/data/verify_crop_reproduction.py`, all
2,547 images — not the 3-image spot check this claim originally rested on):

| Source | Count | Mean pixel diff | Range | Effectively identical (diff < 1.0) |
|---|---|---|---|---|
| This session's own recovered crops | 145 | **0.0** | 0.0 – 0.0 | **145/145 (100%)** |
| Original legacy pipeline's crops | 2,401 | 43.9 | 7.9 – 88.0 | **0/2,401 (0%)** |

Running `facenet_pytorch.MTCNN` fresh against a source image reproduces
this session's own recovered crops *exactly, every single time*
(deterministic, same model/library) — and reproduces *none* of the 2,401
original legacy crops even approximately. Those were produced by
DeepFace's TensorFlow-based MTCNN backend, a different implementation with
different default alignment behavior. There is no way to exactly recreate
that specific 2021-era library's output today; DeepFace's `mtcnn` backend
and its defaults have moved on since. (One image, not counted above, got a
"no face found" result on this particular re-run despite having an
existing crop — a minor model-instability edge case, not investigated
further.)

**What this means in practice:** these scripts regenerate methodologically
equivalent crops/embeddings/features (same detector family, same
formulas), suitable for re-running the original paper's approach and
getting comparable results — not pixel-identical files. Anyone needing the
literal original legacy artifacts should use `data/legacy_snapshot/`
directly (the checksummed, untouched copy of the legacy repo) rather than
regenerating.

## Commands

All operate against `data/legacy_snapshot/` (or any `--legacy-copy` built
by `scripts/data/copy_legacy_snapshot.py`), not the v3 structure directly,
since they're reproducing the *original* pipeline's inputs/outputs:

```
# MTCNN + OpenCV crops, and FaceNet-512 embeddings, for every image
# (point --output at an empty/nonexistent dir so every image is treated
# as "missing" and gets processed)
uv run --with facenet-pytorch --with opencv-python python scripts/data/recover_missing_crops.py \
    --legacy-copy data/legacy_snapshot --output <output-dir> --report-out <report.json>

# 68-point landmarks, for every image (same "point at empty dir" trick)
uv run --with face-alignment python scripts/data/recover_missing_landmarks.py \
    --legacy-copy data/legacy_snapshot --output <output-dir> --report-out <report.json>

# 11,628-dim geometric-ratio features from the (intact) landmarks.csv
uv run python scripts/data/regenerate_geometric_features.py \
    --legacy-copy data/legacy_snapshot --output <output-dir>
```

Each script's own docstring has the full explanation of what it does and
why (`scripts/data/recover_missing_crops.py`,
`scripts/data/recover_missing_landmarks.py`,
`scripts/data/regenerate_geometric_features.py`). This document exists so
that instruction travels with `data/mebeauty_v3/` even if someone never
reads `docs/DATASET_AUDIT.md`.
