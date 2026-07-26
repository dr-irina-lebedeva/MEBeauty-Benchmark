"""Regenerate geometric_features.csv from the intact landmarks.csv.

The legacy `geometric_features.csv` saved each row's feature vector with
`str(numpy_array)`, which silently truncates long arrays with an ellipsis
-- 98% of rows lost the middle of their vector (Finding 2 of the dataset
audit). `landmarks.csv` sits next to it, is not truncated, and the
feature formula is fully documented in
`get_landmarks_geom.features.ipynb`. This re-runs that exact formula
(ported to `mebeauty_benchmark.legacy.geometry`) over the untouched
landmark coordinates and saves the result in a format that can't silently
truncate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mebeauty_benchmark.legacy.geometry import (
    FEATURE_DIM,
    compute_geometric_features,
    parse_landmark_string,
)
from mebeauty_benchmark.legacy.paths import normalize_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    landmarks_path = Path(args.legacy_copy).expanduser().resolve() / "landmarks.csv"
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images: list[str] = []
    scores: list[float] = []
    failures: list[tuple[str, str]] = []
    feature_rows: list[np.ndarray] = []

    with landmarks_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image = normalize_image_path(row["image"])
            try:
                coords = parse_landmark_string(row["landmarks"])
                features = compute_geometric_features(coords)
            except ValueError as error:
                failures.append((image, str(error)))
                continue
            images.append(image)
            scores.append(float(row["score"]))
            feature_rows.append(features.astype(np.float32))

    feature_matrix = (
        np.vstack(feature_rows) if feature_rows else np.zeros((0, FEATURE_DIM))
    )

    features_path = output_dir / "geometric_features.npz"
    np.savez_compressed(features_path, features=feature_matrix)

    index_path = output_dir / "geometric_features_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "image", "score"])
        for row_index, (image, score) in enumerate(zip(images, scores, strict=True)):
            writer.writerow([row_index, image, score])

    print(f"Computed {len(images)} feature vectors of dimension {FEATURE_DIM}")
    print(f"Failed to parse {len(failures)} row(s): {failures[:5]}")
    print(f"Features: {features_path} (shape {feature_matrix.shape}, float32)")
    print(f"Index:    {index_path}")


if __name__ == "__main__":
    main()
