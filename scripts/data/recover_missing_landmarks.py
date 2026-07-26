"""Recover 68-point facial landmarks for images the legacy pipeline silently
failed to landmark (Finding 4 of `docs/DATASET_AUDIT.md`).

Uses `face_alignment` (a maintained, PyTorch-based detector) rather than the
original dlib predictor. Its 2D landmark output follows the same standard
68-point ordering (ibug/300-W) that dlib's model uses -- the same numbering
`mebeauty_benchmark.legacy.geometry`'s `_RATIO_POINTS` assumes -- so the
recovered landmarks feed the existing geometric-feature formula unchanged.
The underlying model differs from the legacy dlib predictor, so exact pixel
values will differ slightly; only the point semantics are guaranteed to
match. Every image is opened as RGB (one legacy image is RGBA and crashes
the detector otherwise) and downscaled before detection if it exceeds
`MAX_SIDE` on its long side (one legacy image is 35 megapixels and gets
OOM-killed at full resolution); recovered landmark coordinates are always
rescaled back to the original image's pixel space.

Requires an extra dependency deliberately kept out of pyproject.toml:

    uv run --with face-alignment python scripts/data/recover_missing_landmarks.py \\
        --legacy-copy data/legacy_snapshot --output data/recovered_landmarks \\
        --report-out reports/legacy_audit/landmark_recovery_report.json

Recovered landmarks are written under `--output`, never merged into the
legacy `landmarks.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import face_alignment
import numpy as np
from PIL import Image

from mebeauty_benchmark.legacy.geometry import compute_geometric_features
from mebeauty_benchmark.legacy.paths import normalize_image_path

# One legacy image is 5304x6630px; full-resolution CPU inference on it gets
# OOM-killed. Downscale anything over this and rescale landmarks back.
MAX_SIDE = 1600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for recovered artifacts"
    )
    parser.add_argument(
        "--report-out", required=True, help="Where to write the recovery report"
    )
    return parser.parse_args()


def find_missing_landmarks(images_root: Path, landmarks_csv: Path) -> list[str]:
    all_images = {
        p.relative_to(images_root).as_posix()
        for p in images_root.rglob("*")
        if p.is_file()
    }
    with landmarks_csv.open(encoding="utf-8") as file:
        existing = {normalize_image_path(row["image"]) for row in csv.DictReader(file)}
    return sorted(all_images - existing)


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    images_root = legacy_root / "original_images"
    landmarks_csv = legacy_root / "landmarks.csv"

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_out = Path(args.report_out).expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)

    missing = find_missing_landmarks(images_root, landmarks_csv)
    print(f"{len(missing)} images missing a landmarks row", flush=True)

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu"
    )
    print("detector ready", flush=True)

    feature_rows = []
    feature_index = []
    results = []
    landmarks_out = output_dir / "landmarks_recovered.csv"
    with landmarks_out.open("w", newline="", encoding="utf-8") as landmarks_file:
        writer = csv.DictWriter(landmarks_file, fieldnames=["image", "landmarks"])
        writer.writeheader()
        for i, relative in enumerate(missing, start=1):
            print(f"[{i}/{len(missing)}] {relative}", flush=True)
            try:
                image = Image.open(images_root / relative).convert("RGB")
                scale = 1.0
                if max(image.size) > MAX_SIDE:
                    scale = MAX_SIDE / max(image.size)
                    image = image.resize(
                        (
                            max(1, round(image.width * scale)),
                            max(1, round(image.height * scale)),
                        ),
                        Image.LANCZOS,
                    )
                predictions = detector.get_landmarks(np.array(image))
                if not predictions:
                    results.append({"image": relative, "status": "no_face_detected"})
                    continue
                points = predictions[0]  # first/most confident face, 68x2
                flat = (points / scale).reshape(
                    -1
                )  # x1,y1,x2,y2,...,x68,y68, original-image scale
                writer.writerow(
                    {"image": relative, "landmarks": ",".join(f"{v:.6f}" for v in flat)}
                )
                landmarks_file.flush()
                feature_rows.append(compute_geometric_features(flat))
                feature_index.append(relative)
                results.append({"image": relative, "status": "recovered"})
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                results.append(
                    {"image": relative, "status": "error", "detail": str(exc)}
                )
            # Partial progress survives a crash mid-run.
            report_out.write_text(
                json.dumps(
                    {"method": "face_alignment (in progress)", "results": results},
                    indent=2,
                ),
                encoding="utf-8",
            )

    if feature_rows:
        features_matrix = np.vstack(feature_rows).astype(np.float32)
        np.savez_compressed(
            output_dir / "geometric_features_recovered.npz",
            features=features_matrix,
            image=np.array(feature_index),
        )

    summary = {
        "attempted": len(missing),
        "recovered": sum(1 for r in results if r["status"] == "recovered"),
        "no_face_detected": sum(
            1 for r in results if r["status"] == "no_face_detected"
        ),
        "errors": sum(1 for r in results if r["status"] == "error"),
    }
    report = {
        "method": "face_alignment.FaceAlignment(LandmarksType.TWO_D), "
        f"RGB-converted, downscaled to max {MAX_SIDE}px before detection",
        **summary,
        "results": results,
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Summary: {summary}")
    print(f"Landmarks: {landmarks_out}")
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
