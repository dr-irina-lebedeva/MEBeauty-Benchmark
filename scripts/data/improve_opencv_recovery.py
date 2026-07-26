"""Second-pass recovery for images the first OpenCV Haar-cascade attempt
in `recover_missing_crops.py` missed.

Loosening a Haar cascade's parameters (more cascades, histogram
equalization, finer scale steps) finds *a* candidate box for essentially
every remaining miss -- but that is a known false-positive risk, not a
genuine improvement on its own (confirmed here: one such candidate was
centered on an ear). Every candidate is therefore cross-validated against
an independent detector (`facenet_pytorch.MTCNN`, run on the full image)
via IoU -- only candidates that an unrelated detection method also agrees
on are kept. This is a verification step on the same OpenCV-family
detections, not a switch to a different detector for the crop itself.

Requires extra dependencies deliberately kept out of pyproject.toml:

    uv run --with "opencv-python<5" --with facenet-pytorch python scripts/data/improve_opencv_recovery.py \\
        --legacy-copy data/legacy_snapshot --crop-report reports/legacy_audit/crop_recovery_report.json \\
        --output data/recovered_crops/opencv

Note: pin `opencv-python<5` -- 5.x's default build does not expose
`cv2.CascadeClassifier` in this environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from facenet_pytorch import MTCNN
from PIL import Image

CASCADE_FILES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt_tree.xml",
]
SCALE_NEIGHBOR_SWEEP = [(1.1, 5), (1.05, 3), (1.05, 4), (1.03, 3), (1.1, 3)]
IOU_THRESHOLD = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--crop-report",
        required=True,
        help="Report from recover_missing_crops.py to read/update",
    )
    parser.add_argument(
        "--output", required=True, help="Where to write the additional OpenCV crops"
    )
    return parser.parse_args()


def find_candidate_box(
    cascades: list, gray, gray_eq
) -> tuple[int, int, int, int] | None:
    for source in (gray, gray_eq):
        for cascade in cascades:
            for scale_factor, min_neighbors in SCALE_NEIGHBOR_SWEEP:
                faces = cascade.detectMultiScale(
                    source,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(20, 20),
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    return int(x), int(y), int(w), int(h)
    return None


def iou(box_a: tuple[int, int, int, int], box_b) -> float:
    ax, ay, aw, ah = box_a
    bx1, by1, bx2, by2 = box_b
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = aw * ah
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    images_root = Path(args.legacy_copy).expanduser().resolve() / "original_images"
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.crop_report).expanduser().resolve()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    misses = [
        r["image"]
        for r in report["opencv"]["results"]
        if r["status"] == "no_face_detected"
    ]
    print(f"{len(misses)} prior OpenCV misses to retry")

    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + name) for name in CASCADE_FILES
    ]
    mtcnn = MTCNN(keep_all=True, device="cpu")

    verified = 0
    rejected = 0
    for relative in misses:
        image_path = images_root / relative
        cv_image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        candidate = find_candidate_box(cascades, gray, gray_eq)

        result = next(r for r in report["opencv"]["results"] if r["image"] == relative)
        if candidate is None:
            continue

        boxes, _ = mtcnn.detect(Image.open(image_path).convert("RGB"))
        best_iou = (
            max((iou(candidate, box) for box in boxes), default=0.0)
            if boxes is not None
            else 0.0
        )
        if best_iou < IOU_THRESHOLD:
            rejected += 1
            result["note"] = (
                f"expanded Haar tuning found a candidate box but MTCNN did not corroborate it "
                f"(best IoU {best_iou:.3f} < {IOU_THRESHOLD}); rejected as a likely false positive"
            )
            continue

        x, y, w, h = candidate
        dest = output_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), cv_image[y : y + h, x : x + w])
        result["status"] = "recovered"
        result["method"] = "haar_cascade_expanded_tuning_mtcnn_verified"
        result["iou_with_mtcnn"] = round(best_iou, 3)
        verified += 1

    report["opencv"]["recovered"] = sum(
        1 for r in report["opencv"]["results"] if r["status"] == "recovered"
    )
    report["opencv"]["no_face_detected"] = sum(
        1 for r in report["opencv"]["results"] if r["status"] == "no_face_detected"
    )
    report["method_opencv"] = (
        "cv2 haarcascade (default/alt/alt2/alt_tree, incl. histogram-equalized, multiple "
        "scaleFactor/minNeighbors), candidates cross-validated against facenet_pytorch.MTCNN "
        f"via IoU>={IOU_THRESHOLD} to reject false positives"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Verified (kept): {verified}")
    print(f"Rejected (false positive, likely): {rejected}")
    print(f"Still no face found: {len(misses) - verified - rejected}")
    print(f"\nWrote crops to: {output_dir}")
    print(f"Updated report: {report_path}")


if __name__ == "__main__":
    main()
