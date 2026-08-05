"""Write the image configurations as split-sharded parquet for Hugging Face.

Two configurations, and the distinction between them is not cosmetic:

* `native` -- the surviving legacy face crops at their original resolution.
  **These are not original photographs.** 2,447 of 2,467 were already cropped
  before this project began (`is_preprocessed_crop`), and calling them
  originals would misdescribe what a user receives. Resolution is
  heterogeneous, 400x400 to 6240x6630.
* `cropped_256` -- the aligned face crop, RGB 256x256. A detector located the
  face, and the crop is taken so that facial geometry lands in the same place
  in every image: eyes at y = 103 +/- 10 px, mouth at y = 196 +/- 9 px,
  interocular distance 113 +/- 11 px. This is the config to train on when the
  model should not have to learn to find the face first.

Images are embedded in the parquet rather than shipped as loose files. That is
what makes `load_dataset` work with no download script and gives the Hub's
viewer something to render; at 113 MB and 72 MB neither config needs sharding,
though the writer shards anyway if a config ever grows past the limit.

Split membership travels with the rows, so `load_dataset(..., split="test")`
returns exactly the released test set and cannot silently disagree with
`images.parquet`.

    uv run python scripts/release/build_image_configs.py \\
        --v3 data/mebeauty_v3 --release data/release/public
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: Rows per shard. Chosen so a shard lands in the 100-500 MB band the Hub
#: prefers even for the heaviest native images.
SHARD_ROWS = 1000

SPLIT_FILENAMES = {"train": "train", "val": "validation", "test": "test"}

#: Columns carried alongside the image. Deliberately the same for both configs,
#: so switching config changes only the pixels.
#:
#: **Everything derivable was dropped.** `width`/`height` are readable from the
#: image; `split` is the parquet filename; `label_status` is
#: `rating_count >= 10`; `source_platform` is the URL's domain; `sha256` is the
#: hash of bytes the user already holds; and `source_status`,
#: `verification_method` and `preprocessing_version` were single-valued
#: columns -- 2,462 identical strings carrying no information. They are stated
#: once in the card instead of repeated on every row.
CARRIED = [
    "release_id",
    "sha256",
    "score",
    "score_mean",
    "score_std",
    "score_median",
    "rating_count",
    "rating_distribution",
    "label_status",
    "legacy_gender_label",
    "legacy_ethnicity_label",
    "source_platform",
    "source_url",
    "source_status",
    "verification_method",
    "split",
    "cv_fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--release", required=True)
    return parser.parse_args()


def features_for(landmark_field: str):
    """Typed features, including the HF `Image` type on `image`.

    Writing the parquet by hand with a plain Arrow struct is not enough: the
    Hub identifies an image column by extension metadata that `datasets`
    embeds, and without it every consumer gets `{"bytes": ..., "path": ...}`
    dicts instead of decoded images. Building through `datasets` is what makes
    the viewer render and `load_dataset` return real images.
    """
    from datasets import Features, Image, Sequence, Value

    return Features(
        {
            "image": Image(),
            "release_id": Value("string"),
            "score_mean": Value("float64"),
            "rating_count": Value("int32"),
            "rating_distribution": Sequence(Value("int32")),
            landmark_field: Sequence(Value("float64")),
            "legacy_gender_label": Value("string"),
            "legacy_ethnicity_label": Value("string"),
            "cv_fold": Value("int32"),
            "split_60_40": Value("string"),
        }
    )


def build_config(
    name: str,
    images: pd.DataFrame,
    image_dir: Path,
    landmark_column: str,
    landmark_field: str,
    out: Path,
) -> dict:
    from datasets import Dataset

    out.mkdir(parents=True, exist_ok=True)
    features = features_for(landmark_field)
    written = {}

    for split, filename in SPLIT_FILENAMES.items():
        subset = images[images["split"] == split].reset_index(drop=True)
        shards = max(1, (len(subset) + SHARD_ROWS - 1) // SHARD_ROWS)
        for index in range(shards):
            chunk = subset.iloc[index * SHARD_ROWS : (index + 1) * SHARD_ROWS]
            rows = []
            for row in chunk.itertuples():
                data = (image_dir / f"{row.sha256}.jpg").read_bytes()
                rows.append(
                    {
                        "image": {"bytes": data, "path": f"{row.release_id}.jpg"},
                        "release_id": row.release_id,
                        "score_mean": float(row.score_mean),
                        "rating_count": int(row.rating_count),
                        "rating_distribution": [
                            int(v) for v in row.rating_distribution
                        ],
                        landmark_field: [
                            float(v) for v in getattr(row, landmark_column)
                        ],
                        "legacy_gender_label": row.legacy_gender_label,
                        "legacy_ethnicity_label": row.legacy_ethnicity_label,
                        "cv_fold": None if pd.isna(row.cv_fold) else int(row.cv_fold),
                        "split_60_40": row.split_60_40,
                    }
                )
            dataset = Dataset.from_list(rows, features=features)
            target = out / f"{filename}-{index:05d}-of-{shards:05d}.parquet"
            dataset.to_parquet(target)
            written[target.name] = len(rows)
            print(
                f"  {name}/{target.name}: {len(rows)} rows, "
                f"{target.stat().st_size / 1e6:.1f} MB"
            )
    return written


def main() -> None:
    args = parse_args()
    v3 = Path(args.v3).expanduser().resolve()
    release = Path(args.release).expanduser().resolve()
    # Build artefact, kept outside public/ so it is never uploaded.
    images = pd.read_parquet(release.parent / "images.parquet")

    print(f"native ({len(images)} images):")
    native = build_config(
        "native",
        images,
        v3 / "images",
        "landmarks_native",
        "landmarks",
        release / "data" / "native",
    )
    print(f"cropped_256 ({len(images)} images):")
    cropped = build_config(
        "cropped_256",
        images,
        v3 / "cropped_256" / "images",
        "landmarks_cropped",
        "landmarks",
        release / "data" / "cropped_256",
    )
    print(f"\nWrote {len(native)} native shards and {len(cropped)} cropped_256 shards")


if __name__ == "__main__":
    main()
