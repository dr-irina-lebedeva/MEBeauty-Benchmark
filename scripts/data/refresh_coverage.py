"""Make the shipped summaries describe the shipped data.

`COVERAGE.json` and the derived configurations' `metadata.parquet` files are
written **mid-pipeline**, by `build_v3_dataset.py`, `standardize_images.py`
and `build_face_crops.py` -- all of which run before identity consolidation,
before the PNG-to-JPEG conversion re-derives content hashes, and before the
labels exist. Every later step therefore leaves them describing a dataset
that is no longer on disk. Two concrete instances this script exists to catch:

- `COVERAGE.json` claimed 2,492 images and 1,749/222/516 split rows, from
  before 25 duplicate photographs were merged and before any label existed.
- `cropped_256/images/metadata.parquet` carried 2,469 rows for 2,467 crops:
  the PNG conversion changed two `image_id`s and the remap added the new rows
  without retiring the old ones.

Rather than hand-edit either, this script **derives** them from what is
actually shipped, and is safe to re-run after any change.

A metadata row whose image is missing is dropped and reported. An image with
no metadata row is a hard error: that is a real gap, not staleness, and
silently inventing a row would hide it.

It also rewrites `README.md` -- the dataset card -- for the same reason and
from the same numbers. `standardize_images.py` writes an earlier version of
the card's Hugging Face config block; **this script is authoritative
afterwards**, and it declares every configuration it finds on disk rather
than a hardcoded pair, so `cropped_256` stops being invisible to
`load_dataset`.

    uv run python scripts/data/refresh_coverage.py --v3 data/mebeauty_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")

#: Derived image configurations, each `<name>/images/` + a metadata.parquet
#: beside the images it describes.
DERIVED_CONFIGS = ("cropped_256", "standardized_256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument(
        "--report-out", help="Optional JSON report path (defaults to none)"
    )
    return parser.parse_args()


def reconcile_config(v3_dir: Path, config: str) -> dict:
    """Drop metadata rows with no image; fail if an image has no row."""
    images_dir = v3_dir / config / "images"
    metadata_path = images_dir / "metadata.parquet"
    if not metadata_path.is_file():
        return {"status": "no metadata.parquet"}

    on_disk = {path.stem for path in images_dir.glob("*.jpg")}
    frame = pd.read_parquet(metadata_path)
    described = set(frame["image_id"])

    missing = on_disk - described
    if missing:
        raise AssertionError(
            f"{config}: {len(missing)} image(s) have no metadata row, "
            f"e.g. {sorted(missing)[:3]}"
        )

    orphans = described - on_disk
    if orphans:
        frame = frame[frame["image_id"].isin(on_disk)].reset_index(drop=True)
        frame.to_parquet(metadata_path, index=False)

    return {
        "images_on_disk": len(on_disk),
        "rows_before": len(described),
        "rows_after": len(frame),
        "orphan_rows_dropped": len(orphans),
    }


def write_dataset_card(v3_dir: Path, coverage: dict, configs: dict) -> None:
    """Rewrite README.md so the card describes what is on disk.

    The Hugging Face config block is generated from the directories that
    actually exist. `native` stays the default: it is the authoritative
    configuration, and the derived 256x256 ones are conveniences.
    """
    available = ["native"] + [
        config for config, summary in configs.items() if "images_on_disk" in summary
    ]
    blocks = []
    for config in available:
        path = "images/**" if config == "native" else f"{config}/images/**"
        default = "\n    default: true" if config == "native" else ""
        blocks.append(
            f"  - config_name: {config}{default}\n"
            f"    data_files:\n      - split: train\n        path: {path}"
        )
    frontmatter = "---\nconfigs:\n" + "\n".join(blocks) + "\n---\n"

    counts = coverage["rating_counts"]
    body = f"""
# MEBeauty v3 (local, unreleased)

Flat, content-addressed images with labels only in metadata -- see
`docs/RESTRUCTURE_PROPOSAL.md` for why. **Local only** -- not uploaded, not
licensed, not committed to git. Known problems that block release are listed
in `docs/DATASET_PROBLEMS_2026-07-28.md`; read it before publishing anything.

## Coverage

{coverage["unique_images"]} images, of which **{coverage["labelled_images"]} carry a label**
(train {counts["train"]} / val {counts["val"]} / test {counts["test"]}). The remaining
{coverage["unlabelled_images"]} have fewer than 10 ratings from valid raters; they keep their
pixels and metadata and appear in no split. Full counts in `COVERAGE.json`.

## Labels

`score` (1-10) is the **mean over valid raters after each rater's scale is
normalised**, computed only from the `generic` attractiveness task. It is
fully recomputable from `ratings/by_rater/ratings_by_rater.parquet`, which is
the point: the 2021 legacy label was not.

Three alternatives ship beside it so no consumer is forced to accept that
choice -- `score_raw_mean` (no normalisation; equals the mean of the shipped
distribution), `score_all_raters` (no screening at all), and `score_adjusted`
(a jointly-fitted per-rater affine model, which agrees with `score` at
r = 0.98). `n_ratings`, `std` and `ci95` give the support behind each label.

A rater is excluded only for using fewer than 3 distinct scores -- behaviour,
never disagreement with the majority. Filtering raters on agreement would
delete exactly the minority aesthetic variation this dataset exists to study.

## Layout

- `images/<image_id>.jpg` + `images/metadata.parquet` -- {coverage["unique_images"]} unique
  images keyed by SHA-256 of their contents, with metadata as a *sibling* of
  the image files (not at the dataset root -- verified that this matters:
  sibling parquet files elsewhere in the tree confuse Hugging Face's
  `ImageFolder` split auto-detection). `load_dataset("imagefolder",
  data_dir="images")` works directly. Columns include `file_name` (the
  ImageFolder-required link), legacy provenance, gender, ethnicity,
  `source_url`, and label-collision flags.
- `landmarks.parquet` -- image_id, landmarks (native `list<float>`, 136
  values = 68 points, not a string -- queryable without re-parsing).
- `ratings/aggregate/{{train,val,test}}.parquet` -- the canonical splits,
  keyed by image_id, one row per **labelled** image, with `score` and the
  alternatives above.
- `ratings/distributions.parquet` -- per-image soft labels over the 1-10
  scale for label distribution learning. Counts of the integer scores raters
  actually gave, so its mean is `score_raw_mean`, not `score`.
- `ratings/by_rater/ratings_by_rater.parquet` -- individual pseudonymized
  rater scores; every row kept, with `rater_valid` marking the screen, so the
  filter is auditable and reversible.
- `cropped_256/`, `standardized_256/` -- derived 256x256 configurations.
- `reproduce_legacy_baseline/` -- pointer to `docs/REPRODUCE_LEGACY_BASELINE.md`;
  crops/embeddings/geometric-features are deliberately not shipped as data.

## Splits

Fixed, not k-fold: every photograph of one person sits in a single split, so
re-folding at random would reintroduce identity leakage.
"""
    (v3_dir / "README.md").write_text(frontmatter + body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()

    configs = {}
    for config in DERIVED_CONFIGS:
        configs[config] = reconcile_config(v3_dir, config)
        print(f"{config}: {configs[config]}")

    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    native_on_disk = len(list((v3_dir / "images").glob("*.jpg")))
    if native_on_disk != len(metadata):
        raise AssertionError(
            f"images/: {native_on_disk} files but {len(metadata)} metadata rows"
        )

    landmarks = pd.read_parquet(v3_dir / "landmarks.parquet")
    split_rows = {
        split: len(
            pd.read_parquet(v3_dir / "ratings" / "aggregate" / f"{split}.parquet")
        )
        for split in SPLITS
    }
    labelled = sum(split_rows.values())

    unresolved = int(
        (metadata["has_label_collision"] & ~metadata["label_collision_resolved"]).sum()
    )
    coverage = {
        "unique_images": len(metadata),
        "labelled_images": labelled,
        "unlabelled_images": len(metadata) - labelled,
        "label_collisions_total": int(metadata["has_label_collision"].sum()),
        "label_collisions_resolved": int(metadata["label_collision_resolved"].sum()),
        "label_collisions_remaining": unresolved,
        "landmark_rows": len(landmarks),
        "rating_counts": split_rows,
        "note": (
            "Images without a label have fewer than 10 ratings from valid "
            "raters. They keep their pixels and metadata and enter no split."
        ),
        "configurations": {
            "native": {
                "images": len(metadata),
                "note": "authoritative; heterogeneous resolution (Finding 17)",
            },
        },
    }
    for config, summary in configs.items():
        if "images_on_disk" in summary:
            coverage["configurations"][config] = {
                "images": summary["images_on_disk"],
                "note": f"derived convenience config; uniform 256x256 ({config})",
            }

    (v3_dir / "COVERAGE.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(coverage, indent=2))
    print(f"Wrote {v3_dir / 'COVERAGE.json'}")

    write_dataset_card(v3_dir, coverage, configs)
    print(f"Wrote {v3_dir / 'README.md'}")

    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"configs": configs, "coverage": coverage}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
