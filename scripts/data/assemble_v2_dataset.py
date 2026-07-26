"""Assemble one self-contained, complete-coverage local dataset directory.

Merges the legacy pipeline's outputs with this session's recovered
artifacts (Findings 3-4 of `docs/DATASET_AUDIT.md`) into a single directory
tree, instead of leaving legacy and recovered artifacts in separate
folders. Recovered crops/landmarks used the same detection *methods*
(MTCNN, OpenCV, 68-point landmarks) as the legacy pipeline, just a working
implementation -- this fills gaps in an already-used method, it does not
select a new canonical preprocessing pipeline (that choice is still
explicitly open, see `docs/DATASET_AUDIT.md`'s "Open" section).

This produces **local-only** output. Nothing here is uploaded, licensed, or
committed to git -- see `docs/DATASET_AUDIT.md`'s "Open" section for what
still blocks any public release.

    uv run python scripts/data/assemble_v2_dataset.py \\
        --legacy-copy data/legacy_snapshot \\
        --recovered-crops data/recovered_crops \\
        --recovered-landmarks data/recovered_landmarks \\
        --geometric-features data/geometric_features \\
        --reports reports/legacy_audit \\
        --output data/mebeauty_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from mebeauty_benchmark.legacy.paths import normalize_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--recovered-crops",
        required=True,
        help="Output dir from recover_missing_crops.py",
    )
    parser.add_argument(
        "--recovered-landmarks",
        required=True,
        help="Output dir from recover_missing_landmarks.py",
    )
    parser.add_argument(
        "--geometric-features",
        required=True,
        help="Output dir from regenerate_geometric_features.py",
    )
    parser.add_argument(
        "--reports", required=True, help="reports/legacy_audit directory"
    )
    parser.add_argument(
        "--output", required=True, help="Directory for the assembled v2 dataset"
    )
    return parser.parse_args()


def merge_tree(
    sources: list[Path], dest: Path, valid_stems: set[str] | None = None
) -> int:
    """Copy every file from `sources` into `dest`, keyed by relative path.

    If `valid_stems` is given, skip any file whose relative path (without
    extension) isn't in it -- drops orphaned legacy artifacts that don't
    correspond to any current `original_images/` file (Finding 3: one
    MTCNN crop + embedding has no source image left).
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_orphans = 0
    for source in sources:
        if not source.is_dir():
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if (
                valid_stems is not None
                and str(relative.with_suffix("")) not in valid_stems
            ):
                skipped_orphans += 1
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
    if skipped_orphans:
        print(
            f"  skipped {skipped_orphans} orphaned file(s) with no matching original_images entry"
        )
    return copied


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0


def merge_landmarks(
    legacy_root: Path, recovered_landmarks: Path, output_dir: Path
) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    with (legacy_root / "landmarks.csv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image = normalize_image_path(row["image"])
            if image in seen:
                continue
            seen.add(image)
            rows.append({"image": image, "landmarks": row["landmarks"]})
    with (recovered_landmarks / "landmarks_recovered.csv").open(
        encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            if row["image"] in seen:
                continue
            seen.add(row["image"])
            rows.append(row)
    rows.sort(key=lambda r: r["image"])

    landmarks_out = output_dir / "landmarks.csv"
    with landmarks_out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "landmarks"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def merge_geometric_features(
    geometric_features_dir: Path, recovered_landmarks: Path, output_dir: Path
) -> tuple[np.ndarray, list[str]]:
    """Merge legacy + recovered geometric features, deduped by image.

    Some ratio features are mathematically undefined when their two
    denominator landmark points coincide (division by zero) -- confirmed
    for 2 recovered images, both extreme profile poses where the 68-point
    frontal landmark scheme collapses distinct points onto the same pixel.
    `compute_geometric_features` faithfully reproduces the legacy formula,
    including this edge case, so it returns `inf`. This function converts
    those `inf` values to `nan` (the correct IEEE-754 marker for "value is
    undefined", and one every numeric library's `isnan`/`dropna` already
    knows to handle) and returns which images were affected, rather than
    leaving a silent `inf` for a future user to discover mid-training.
    """
    index_rows = list(
        csv.DictReader(
            (geometric_features_dir / "geometric_features_index.csv").open(
                encoding="utf-8"
            )
        )
    )
    legacy_features = np.load(geometric_features_dir / "geometric_features.npz")[
        "features"
    ]
    recovered = np.load(recovered_landmarks / "geometric_features_recovered.npz")
    recovered_features = recovered["features"]
    recovered_images = recovered["image"]

    seen: set[str] = set()
    feature_list = []
    image_list = []
    for row, vector in zip(index_rows, legacy_features, strict=True):
        image = normalize_image_path(row["image"])
        if image in seen:
            continue
        seen.add(image)
        feature_list.append(vector)
        image_list.append(image)
    for image, vector in zip(recovered_images, recovered_features, strict=True):
        if image in seen:
            continue
        seen.add(image)
        feature_list.append(vector)
        image_list.append(image)

    order = sorted(range(len(image_list)), key=lambda i: image_list[i])
    features_matrix = np.vstack([feature_list[i] for i in order]).astype(np.float32)
    images_sorted = np.array([image_list[i] for i in order])

    inf_mask = np.isinf(features_matrix)
    affected_rows = np.where(inf_mask.any(axis=1))[0]
    affected_images = [str(images_sorted[i]) for i in affected_rows]
    features_matrix[inf_mask] = np.nan

    np.savez_compressed(
        output_dir / "geometric_features.npz",
        features=features_matrix,
        image=images_sorted,
    )
    return features_matrix, affected_images


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    recovered_crops = Path(args.recovered_crops).expanduser().resolve()
    recovered_landmarks = Path(args.recovered_landmarks).expanduser().resolve()
    geometric_features_dir = Path(args.geometric_features).expanduser().resolve()
    reports_dir = Path(args.reports).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Copying original_images/ ...")
    shutil.copytree(
        legacy_root / "original_images",
        output_dir / "original_images",
        dirs_exist_ok=True,
    )
    valid_stems = {
        str(p.relative_to(legacy_root / "original_images").with_suffix(""))
        for p in (legacy_root / "original_images").rglob("*")
        if p.is_file()
    }

    print("Merging MTCNN crops (legacy + recovered) ...")
    merge_tree(
        [
            legacy_root / "cropped_images" / "images_crop_align_mtcnn",
            recovered_crops / "mtcnn",
        ],
        output_dir / "crops" / "mtcnn",
        valid_stems,
    )

    print("Merging OpenCV crops (legacy + recovered) ...")
    merge_tree(
        [
            legacy_root / "cropped_images" / "images_crop_align_opencv",
            recovered_crops / "opencv",
        ],
        output_dir / "crops" / "opencv",
        valid_stems,
    )

    print("Merging FaceNet-512 embeddings (legacy + recovered) ...")
    merge_tree(
        [legacy_root / "FaceNet_512_features", recovered_crops / "facenet_512"],
        output_dir / "embeddings" / "facenet_512",
        valid_stems,
    )

    print("Merging landmarks (legacy, deduped, + recovered) ...")
    landmark_rows = merge_landmarks(legacy_root, recovered_landmarks, output_dir)

    print("Merging geometric features (legacy, deduped, + recovered) ...")
    features_matrix, undefined_ratio_images = merge_geometric_features(
        geometric_features_dir, recovered_landmarks, output_dir
    )
    if undefined_ratio_images:
        print(
            f"  {len(undefined_ratio_images)} image(s) have some undefined (NaN) ratio "
            f"features -- coincident landmark points, e.g. extreme profile poses: "
            f"{undefined_ratio_images}"
        )

    print("Copying canonical ratings/splits ...")
    ratings_out = output_dir / "ratings"
    ratings_out.mkdir(exist_ok=True)
    ratings_total = 0
    for split in ["train", "val", "test"]:
        source = reports_dir / "canonical_splits" / f"{split}.csv"
        shutil.copy2(source, ratings_out / f"{split}.csv")
        ratings_total += sum(1 for _ in csv.DictReader(source.open(encoding="utf-8")))

    print("Copying provenance, checksums, and metadata tables ...")
    shutil.copy2(reports_dir / "image_provenance.csv", output_dir / "provenance.csv")
    shutil.copy2(
        reports_dir / "canonical_dataset" / "checksums.sha256",
        output_dir / "checksums.sha256",
    )
    shutil.copy2(
        reports_dir / "canonical_dataset" / "images.parquet",
        output_dir / "images.parquet",
    )
    shutil.copy2(
        reports_dir / "canonical_dataset" / "ratings.parquet",
        output_dir / "ratings.parquet",
    )

    summary = {
        "original_images": count_files(output_dir / "original_images"),
        "crops_mtcnn": count_files(output_dir / "crops" / "mtcnn"),
        "crops_opencv": count_files(output_dir / "crops" / "opencv"),
        "embeddings_facenet_512": count_files(
            output_dir / "embeddings" / "facenet_512"
        ),
        "landmark_rows": len(landmark_rows),
        "geometric_feature_rows": int(features_matrix.shape[0]),
        "geometric_feature_rows_with_undefined_ratios": undefined_ratio_images,
        "canonical_rating_rows": ratings_total,
    }
    (output_dir / "COVERAGE.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# MEBeauty v2 (local, unreleased)

Assembled by `scripts/data/assemble_v2_dataset.py` from the legacy pipeline's
outputs plus this session's recovered artifacts. **Local only** -- not
uploaded, not licensed, not committed to git. See `docs/DATASET_AUDIT.md`
for the full findings and `docs/DATASET_CARD.md` / `docs/DATASHEET.md` for
what a real release would need first.

## Coverage

```json
{json.dumps(summary, indent=2)}
```

## Layout

- `original_images/{{gender}}/{{ethnicity}}/` -- source images, untouched
- `crops/mtcnn/`, `crops/opencv/` -- face crops (legacy + recovered, merged)
- `embeddings/facenet_512/` -- FaceNet-512 embeddings (legacy + recovered)
- `landmarks.csv`, `geometric_features.npz` -- 68-point landmarks and
  derived geometric-ratio features (legacy, deduped, + recovered). A few
  ratio features are `NaN` (mathematically undefined, not missing data) for
  the images listed in `geometric_feature_rows_with_undefined_ratios` above
  -- extreme profile poses where two landmark points coincide.
- `ratings/{{train,val,test}}.csv` -- canonical `benchmark-v1` split
  (filename- and content-duplicate leakage removed)
- `provenance.csv` -- inferred (unverified) image source platform
- `checksums.sha256`, `images.parquet`, `ratings.parquet` -- reproducibility
  metadata, same as `reports/legacy_audit/canonical_dataset/`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nAssembled: {output_dir}")


if __name__ == "__main__":
    main()
