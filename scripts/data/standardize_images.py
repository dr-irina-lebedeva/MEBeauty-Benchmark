"""Build the `standardized_256` configuration alongside the native images.

The native images stay authoritative and are never read-modify-written here:
this script only *reads* `images/` and `landmarks.parquet` and writes a new
`standardized_256/` tree next to them.

Why a second configuration rather than replacing the native files
(docs/DATASET_AUDIT.md, Finding 17): the surviving legacy images are
heterogeneous -- 1,302 at 500x500, 1,095 at 400x400, 75 at 600x600, and 20
full-size photographs. Uniform 256x256 is convenient for benchmarking (it is
the size torchvision's standard pipeline resizes to before a 224 centre
crop), but shipping *only* that would permanently discard resolution that
cannot be recovered: the original photographs are not in the release, and
the upstream repository is no longer under this project's control. So native
is the default and standardized is the reproducible convenience layer.

Preprocessing, all deterministic:

- convert to RGB (3 legacy files are PNG; they become JPEG for uniformity)
- aspect-preserving resize with Lanczos + antialiasing, longer side to 256
- centre-pad the short axis onto a 256x256 canvas -- faces are never stretched
- JPEG quality 95, no chroma subsampling

Quality 95 rather than lossless PNG because the *resize* is the dominant
lossy step here: downsampling 400->256 removes the high-frequency detail that
JPEG artifacts occupy, so lossless encoding would quadruple the download to
preserve detail the resample already discarded. Native remains the
authoritative copy regardless.

No normalization or augmentation is baked into the files -- those belong in
the training pipeline, not the dataset.

    uv run --with pillow python scripts/data/standardize_images.py \\
        --v3 data/mebeauty_v3 --target 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import PIL
from PIL import Image

from mebeauty_benchmark.legacy.standardize import (
    compute_transform,
    has_out_of_bounds_landmarks,
    transform_landmarks,
)

JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="Path to data/mebeauty_v3")
    parser.add_argument("--target", type=int, default=256, help="Output square size")
    return parser.parse_args()


def preprocessing_version(target: int) -> str:
    """Pin everything that can change the output bytes.

    Pillow's resampling and JPEG encoder can both shift between releases, so
    the version string records the library version alongside the settings --
    without it, "deterministic" would only mean "deterministic on this
    machine today".
    """
    return (
        f"mebeauty-standardize/1.0;target={target};resample=LANCZOS;"
        f"jpeg_quality={JPEG_QUALITY};subsampling=0;pillow={PIL.__version__}"
    )


_CONFIG_FRONTMATTER = """---
configs:
  - config_name: native
    default: true
    data_files:
      - split: train
        path: images/**
  - config_name: standardized_{target}
    data_files:
      - split: train
        path: standardized_{target}/images/**
---

"""


def write_config_frontmatter(v3_dir: Path, target: int) -> None:
    """Declare both configurations for Hugging Face, in the dataset README.

    `build_v3_dataset.py` writes this README describing the native layout;
    this runs afterwards and prepends the config block, so the file ends up
    describing whatever actually exists. Verified against the real directory
    that both `load_dataset(..., name="native")` and
    `name="standardized_{target}"` resolve -- this area has bitten before
    (sibling ratings parquet files silently produced empty phantom splits),
    so it is checked rather than assumed.
    """
    readme = v3_dir / "README.md"
    body = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if body.startswith("---"):  # already has frontmatter; replace it
        body = body.split("---", 2)[-1].lstrip("\n")
    readme.write_text(
        _CONFIG_FRONTMATTER.format(target=target) + body, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    native_images = v3_dir / "images"
    output_dir = v3_dir / f"standardized_{args.target}"
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(native_images / "metadata.parquet")
    landmarks = pd.read_parquet(v3_dir / "landmarks.parquet")
    landmark_by_id = dict(
        zip(landmarks["image_id"], landmarks["landmarks"], strict=True)
    )
    version = preprocessing_version(args.target)
    print(f"{len(metadata)} images -> {output_images}")
    print(f"  {version}")

    rows, landmark_rows = [], []
    for count, row in enumerate(metadata.itertuples(), start=1):
        source = native_images / row.file_name
        with Image.open(source) as image:
            native_width, native_height = image.size
            transform = compute_transform(native_width, native_height, args.target)
            rgb = image.convert("RGB")
            resized = rgb.resize(
                (transform.resized_width, transform.resized_height),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=None,
            )

        canvas = Image.new("RGB", (args.target, args.target), (0, 0, 0))
        canvas.paste(resized, (transform.pad_left, transform.pad_top))

        # Always .jpg: the 3 legacy PNGs become JPEG so the configuration is
        # uniform in format as well as size.
        destination = output_images / f"{row.image_id}.jpg"
        canvas.save(
            destination,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,
            optimize=True,
        )
        standardized_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()

        native_landmarks = landmark_by_id.get(row.image_id)
        out_of_bounds = None
        if native_landmarks is not None:
            native_list = [float(value) for value in native_landmarks]
            moved = transform_landmarks(native_list, transform)
            out_of_bounds = has_out_of_bounds_landmarks(
                native_list, native_width, native_height
            )
            landmark_rows.append(
                {
                    "image_id": row.image_id,
                    "landmarks": moved,
                    "has_out_of_bounds_landmarks": out_of_bounds,
                }
            )

        rows.append(
            {
                "image_id": row.image_id,
                "file_name": destination.name,
                "native_sha256": row.image_id,
                "standardized_sha256": standardized_sha256,
                "original_width": native_width,
                "original_height": native_height,
                "resize_scale": round(transform.scale, 8),
                "pad_left": transform.pad_left,
                "pad_top": transform.pad_top,
                "pad_right": transform.pad_right,
                "pad_bottom": transform.pad_bottom,
                "output_width": transform.output_width,
                "output_height": transform.output_height,
                "preprocessing_version": version,
                "has_out_of_bounds_landmarks": out_of_bounds,
                "gender": row.gender,
                "ethnicity": row.ethnicity,
                "crop_batch": row.crop_batch,
                "legacy_filename": row.legacy_filename,
                "legacy_path": row.legacy_path,
                "inferred_platform": row.inferred_platform,
                "inferred_photo_id": row.inferred_photo_id,
                "has_label_collision": row.has_label_collision,
                "has_near_duplicate": row.has_near_duplicate,
            }
        )
        if count % 500 == 0:
            print(f"  {count}/{len(metadata)}")

    standardized_metadata = pd.DataFrame(rows)
    standardized_metadata.to_parquet(output_images / "metadata.parquet", index=False)
    pd.DataFrame(landmark_rows).to_parquet(
        output_dir / "landmarks.parquet", index=False
    )

    padded = standardized_metadata[
        (standardized_metadata["pad_left"] > 0) | (standardized_metadata["pad_top"] > 0)
    ]
    summary = {
        "images": len(standardized_metadata),
        "target": args.target,
        "preprocessing_version": version,
        "needed_padding": len(padded),
        "square_no_padding": len(standardized_metadata) - len(padded),
        "landmark_rows": len(landmark_rows),
        "with_out_of_bounds_landmarks": int(
            standardized_metadata["has_out_of_bounds_landmarks"].fillna(False).sum()
        ),
        "scale_range": [
            float(standardized_metadata["resize_scale"].min()),
            float(standardized_metadata["resize_scale"].max()),
        ],
    }
    (output_dir / "COVERAGE.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_config_frontmatter(v3_dir, args.target)

    # build_v3_dataset.py writes the top-level COVERAGE.json before this script
    # runs and therefore cannot know about the second configuration. Fold it in
    # here rather than hand-editing, so a rebuild does not silently drop it.
    root_coverage_path = v3_dir / "COVERAGE.json"
    if root_coverage_path.is_file():
        root_coverage = json.loads(root_coverage_path.read_text(encoding="utf-8"))
        root_coverage["configurations"] = {
            "native": {
                "images": root_coverage.get("unique_images"),
                "note": "authoritative; heterogeneous resolution (Finding 17)",
            },
            f"standardized_{args.target}": {
                "images": summary["images"],
                "preprocessing_version": version,
                "note": f"derived convenience config; uniform {args.target}x{args.target}",
            },
        }
        root_coverage_path.write_text(
            json.dumps(root_coverage, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
    print(f"\nNative images untouched at {native_images}")
    print(f"Declared configs 'native' + 'standardized_{args.target}' in README.md")


if __name__ == "__main__":
    main()
