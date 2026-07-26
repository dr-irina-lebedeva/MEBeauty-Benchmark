"""Assemble the canonical, tabular metadata layer for the dataset release.

Joins every audit output (inventory manifest, provenance inference,
duplicate-image groups, label collisions, canonical splits, and per-image
derived-artifact coverage) into two flat tables plus a plain checksum file:

- `images.parquet` -- one row per file under `original_images/`: gender,
  ethnicity, size, SHA-256, which derived artifacts exist for it (MTCNN/
  OpenCV crop, FaceNet embedding, landmarks), inferred source provenance,
  whether it is part of a content-duplicate group, and whether it has a
  cross-label conflict.
- `ratings.parquet` -- one row per canonical train/val/test rating.
- `checksums.sha256` -- plain `sha256sum`-format manifest for
  `original_images/`.

Parquet is what Hugging Face's Dataset Viewer and automatic Croissant
metadata generation expect. This table contains no pixel data and no rater
identifiers -- see `docs/DATASET_AUDIT.md` for what is still open (licensing,
GDPR review) before any of this, or the images themselves, can be released.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from mebeauty_benchmark.legacy.paths import normalize_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--manifest", required=True, help="Snapshot manifest CSV (path, size, sha256)"
    )
    parser.add_argument(
        "--reports", required=True, help="reports/legacy_audit directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the canonical tables"
    )
    return parser.parse_args()


def relative_file_set(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    reports_dir = Path(args.reports).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_path: dict[str, dict[str, str]] = {}
    with open(args.manifest, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["relative_path"].startswith("original_images/"):
                relative = row["relative_path"].split("/", 1)[1]
                manifest_by_path[relative] = row

    provenance_by_path: dict[str, dict[str, str]] = {}
    with (reports_dir / "image_provenance.csv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            provenance_by_path[row["image"]] = row

    label_collisions = json.loads(
        (reports_dir / "label_collisions.json").read_text(encoding="utf-8")
    )
    collided_images: set[str] = set()
    for conflict in label_collisions["conflicts"]:
        collided_images.update(conflict["label_paths"])

    duplicate_report = json.loads(
        (reports_dir / "duplicate_images.json").read_text(encoding="utf-8")
    )
    duplicate_group_by_path: dict[str, str] = {}
    for index, group in enumerate(duplicate_report["groups"]):
        for path in group["paths"]:
            duplicate_group_by_path[path] = f"dup-{index:03d}"

    with (legacy_root / "landmarks.csv").open(encoding="utf-8") as file:
        landmark_paths = {
            normalize_image_path(row["image"]) for row in csv.DictReader(file)
        }

    mtcnn_paths = relative_file_set(
        legacy_root / "cropped_images" / "images_crop_align_mtcnn"
    )
    opencv_paths = relative_file_set(
        legacy_root / "cropped_images" / "images_crop_align_opencv"
    )
    facenet_paths = relative_file_set(legacy_root / "FaceNet_512_features")

    def recovered_paths(report_path: Path, key: str) -> set[str]:
        if not report_path.is_file():
            return set()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        section = report[key] if key else report
        return {r["image"] for r in section["results"] if r["status"] == "recovered"}

    crop_report = reports_dir / "crop_recovery_report.json"
    recovered_mtcnn = recovered_paths(crop_report, "mtcnn")
    recovered_opencv = recovered_paths(crop_report, "opencv")
    landmark_report = reports_dir / "landmark_recovery_report.json"
    recovered_landmarks = recovered_paths(landmark_report, "")

    image_rows = []
    for relative, info in sorted(manifest_by_path.items()):
        gender, ethnicity, _ = relative.split("/", 2)
        provenance = provenance_by_path.get(relative, {})
        facenet_key = str(Path(relative).with_suffix(".csv"))
        image_rows.append(
            {
                "image": relative,
                "gender": gender,
                "ethnicity": ethnicity,
                "extension": Path(relative).suffix.lower(),
                "size_bytes": int(info["size_bytes"]),
                "sha256": info["sha256"],
                "has_mtcnn_crop": relative in mtcnn_paths,
                "has_opencv_crop": relative in opencv_paths,
                "has_facenet_embedding": facenet_key in facenet_paths,
                "has_landmarks": relative in landmark_paths,
                "mtcnn_crop_recovered": relative in recovered_mtcnn,
                "opencv_crop_recovered": relative in recovered_opencv,
                "landmarks_recovered": relative in recovered_landmarks,
                "inferred_platform": provenance.get("inferred_platform", "unknown"),
                "inferred_photo_id": provenance.get("inferred_photo_id", ""),
                "inferred_source_url": provenance.get("inferred_source_url", ""),
                "provenance_confidence": provenance.get("confidence", "unknown"),
                "content_duplicate_group": duplicate_group_by_path.get(relative, ""),
                "has_label_collision": relative in collided_images,
            }
        )
    images_df = pd.DataFrame(image_rows)
    images_df.to_parquet(output_dir / "images.parquet", index=False)

    rating_rows = []
    for split in ["train", "val", "test"]:
        with (reports_dir / "canonical_splits" / f"{split}.csv").open(
            encoding="utf-8"
        ) as file:
            for row in csv.DictReader(file):
                rating_rows.append(
                    {
                        "image": row["image"],
                        "score": float(row["score"]),
                        "split": split,
                    }
                )
    ratings_df = pd.DataFrame(rating_rows)
    ratings_df.to_parquet(output_dir / "ratings.parquet", index=False)

    checksum_lines = [
        f"{info['sha256']}  {relative}"
        for relative, info in sorted(manifest_by_path.items())
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(f"images.parquet:   {len(images_df)} rows -> {output_dir / 'images.parquet'}")
    print(
        f"ratings.parquet:  {len(ratings_df)} rows -> {output_dir / 'ratings.parquet'}"
    )
    print(
        f"checksums.sha256: {len(checksum_lines)} lines -> {output_dir / 'checksums.sha256'}"
    )


if __name__ == "__main__":
    main()
