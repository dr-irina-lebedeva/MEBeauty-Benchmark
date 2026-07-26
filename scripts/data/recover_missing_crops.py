"""Recover face crops (and FaceNet-512 embeddings) for images the legacy
pipeline silently failed to crop (Finding 3 of `docs/DATASET_AUDIT.md`).

`face_crop_align.py` in the legacy repo wraps face detection in a bare
`except: print(...)` -- failures are dropped with no record of which image
failed or why. This re-runs detection on exactly the images missing a crop,
using maintained, actively-developed detector implementations (not the
original DeepFace/dlib stack), and -- unlike the original -- records every
outcome, including genuine "no face detected" failures, instead of
swallowing them.

Requires extra dependencies deliberately kept out of pyproject.toml (heavy,
only needed for this one-off recovery):

    uv run --with facenet-pytorch --with opencv-python python scripts/data/recover_missing_crops.py \\
        --legacy-copy data/legacy_snapshot --output data/recovered_crops \\
        --report-out reports/legacy_audit/crop_recovery_report.json

Recovered crops/embeddings are written under `--output`, never merged into
the legacy `cropped_images/`/`FaceNet_512_features/` directories, so it is
always unambiguous which artifacts came from the original pipeline and
which were reconstructed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


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


def find_missing(images_root: Path, existing_root: Path) -> list[str]:
    all_images = {
        p.relative_to(images_root).as_posix()
        for p in images_root.rglob("*")
        if p.is_file()
    }
    existing = (
        {
            p.relative_to(existing_root).as_posix()
            for p in existing_root.rglob("*")
            if p.is_file()
        }
        if existing_root.is_dir()
        else set()
    )
    return sorted(all_images - existing)


def recover_mtcnn(
    missing: list[str],
    images_root: Path,
    crop_out: Path,
    embedding_out: Path,
    device: str,
) -> list[dict]:
    mtcnn = MTCNN(image_size=224, margin=20, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    results = []
    for relative in missing:
        dest = crop_out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = Image.open(images_root / relative).convert("RGB")
            face = mtcnn(image, save_path=str(dest))
            if face is None:
                results.append({"image": relative, "status": "no_face_detected"})
                continue
            with torch.no_grad():
                embedding = (
                    facenet(face.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                )
            embedding_dest = (embedding_out / relative).with_suffix(".csv")
            embedding_dest.parent.mkdir(parents=True, exist_ok=True)
            embedding_dest.write_text(
                ",".join(f"{v:.18e}" for v in embedding), encoding="utf-8"
            )
            results.append({"image": relative, "status": "recovered"})
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed (that's the fix)
            results.append({"image": relative, "status": "error", "detail": str(exc)})
    return results


def recover_opencv(missing: list[str], images_root: Path, crop_out: Path) -> list[dict]:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    results = []
    for relative in missing:
        dest = crop_out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = cv2.imread(str(images_root / relative))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                results.append({"image": relative, "status": "no_face_detected"})
                continue
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            cv2.imwrite(str(dest), image[y : y + h, x : x + w])
            results.append({"image": relative, "status": "recovered"})
        except Exception as exc:  # noqa: BLE001
            results.append({"image": relative, "status": "error", "detail": str(exc)})
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "attempted": len(results),
        "recovered": sum(1 for r in results if r["status"] == "recovered"),
        "no_face_detected": sum(
            1 for r in results if r["status"] == "no_face_detected"
        ),
        "errors": sum(1 for r in results if r["status"] == "error"),
    }


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    images_root = legacy_root / "original_images"
    mtcnn_existing = legacy_root / "cropped_images" / "images_crop_align_mtcnn"
    opencv_existing = legacy_root / "cropped_images" / "images_crop_align_opencv"

    output_dir = Path(args.output).expanduser().resolve()
    mtcnn_crop_out = output_dir / "mtcnn"
    opencv_crop_out = output_dir / "opencv"
    embedding_out = output_dir / "facenet_512"
    report_out = Path(args.report_out).expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)

    missing_mtcnn = find_missing(images_root, mtcnn_existing)
    missing_opencv = find_missing(images_root, opencv_existing)
    print(
        f"{len(missing_mtcnn)} images missing an MTCNN crop, {len(missing_opencv)} missing an OpenCV crop"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn_results = recover_mtcnn(
        missing_mtcnn, images_root, mtcnn_crop_out, embedding_out, device
    )
    opencv_results = recover_opencv(missing_opencv, images_root, opencv_crop_out)

    report = {
        "method_mtcnn": "facenet_pytorch.MTCNN (image_size=224, margin=20)",
        "method_embedding": "facenet_pytorch.InceptionResnetV1(pretrained='vggface2')",
        "method_opencv": "cv2 haarcascade_frontalface_default",
        "mtcnn": {**summarize(mtcnn_results), "results": mtcnn_results},
        "opencv": {**summarize(opencv_results), "results": opencv_results},
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"MTCNN:   {summarize(mtcnn_results)}")
    print(f"OpenCV:  {summarize(opencv_results)}")
    print(f"\nRecovered crops: {output_dir}")
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
