"""Build a modern, aligned 256x256 face-crop configuration.

The legacy crops were produced in 2021 by `DeepFace.detectFace` wrapped in a
bare `except`, which silently discarded every failure (Finding 3: 143 of 151
`male/indian` images lost their crop and embedding this way). This replaces
them with a current detector that records every outcome instead.

**Detector: SCRFD-10GF via InsightFace `buffalo_l`.** As of 2026 the
SCRFD/RetinaFace family sits on the best accuracy/efficiency frontier for
face detection, and SCRFD leads the WIDER FACE small-model subsets. At 2,493
images throughput is irrelevant, so this is chosen purely on accuracy.

**Important limitation, stated because it bounds what this can achieve.**
`images/` is *already* preprocessed face crops -- 2,472 of 2,493 are uniform
400/500/600px tight crops, and only 20 exceed one megapixel (Finding 17). So
this crops a crop. It buys consistent framing, proper alignment and honest
failure reporting; it cannot recover detail the 2021 preprocessing already
discarded, and it is not equivalent to re-cropping the source photographs,
which are not in this release.

**Faces are aligned, not merely cropped.** The 5-point landmark similarity
transform removes in-plane rotation and normalizes inter-ocular distance, so
every output frames the face the same way -- the property that makes crops
comparable across a benchmark. `--margin` widens the standard ArcFace
template, which is tighter than facial-attractiveness work usually wants
(hair and jawline carry signal here, unlike in recognition).

Images with **more than one** detected face are still cropped: the primary
face is chosen by `select_primary_face`, combining size and centrality. Every
such image is listed in the report with both signals, and the cases where
they disagree between comparably sized faces are flagged `ambiguous` so a
human can overrule the heuristic. Images with **no** detected face are not
cropped and are listed; on this corpus there are none.

    uv run --with insightface --with onnxruntime --with opencv-python-headless \\
        python scripts/data/build_face_crops.py \\
        --v3 data/mebeauty_v3 \\
        --report-out reports/legacy_audit/face_crops.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Detection confidence below which a candidate is not counted as a face.
#: Calibrated on a 150-image sample rather than copied from the MTCNN pass:
#: SCRFD scores this corpus in the 0.65-0.91 band (median 0.84), so MTCNN's
#: 0.95 would have rejected *every* image. At 0.6 all 150 sample images keep
#: their primary face while the value still sits well clear of the noise
#: floor.
MIN_DETECTION_SCORE = 0.6

#: Detector input resolution. These are 400-600px crops, and the default
#: 640x640 upscales them so far that SCRFD detects nothing at all -- verified,
#: 0 of 20 obvious portraits. 320 detects 150 of 150; 480 misses 7.
DET_SIZE = (320, 320)

#: Output edge length. 256 is torchvision's standard resize before a 224
#: centre crop, so the common training pipeline needs no further resampling.
TARGET_SIZE = 256

#: Extra padding around the ArcFace template, as a fraction. The recognition
#: template crops tight to the face; attractiveness ratings respond to hair,
#: jawline and head shape, so some context is kept.
DEFAULT_MARGIN = 0.25

#: Area ratio below which two faces count as "comparably sized", so picking
#: the larger is close to arbitrary. Measured on this corpus: where the
#: largest and the most central face disagree, the ratio is 1.02-1.80x;
#: where they agree it runs up to 217x. 1.5 sits in that gap.
AMBIGUOUS_AREA_RATIO = 1.5

#: Maintainer decisions for images where the heuristic was guessing between
#: two comparably sized faces (reviewed 2026-07-28). Keyed by legacy path;
#: the value is the intended face's horizontal centre as a fraction of image
#: width, and the detected face nearest that point is selected.
#:
#: Recorded as a position rather than a detection index deliberately: an
#: index is meaningless if the detector, its version or its threshold ever
#: reorders results, and would then silently pin the *wrong* person. A
#: position stays meaningful against the image itself.
FACE_SELECTION_OVERRIDES = {
    "female/black/steward-masweneng-Ws4YEqBafus-unsplash.jpg": (
        0.538,
        "woman on the right",
    ),
    "female/black/national-cancer-institute-duNbFJRhaJQ-unsplash.jpg": (
        0.480,
        "woman in the middle",
    ),
    "female/mideastern/hamid-tajik-QXbJ3yhMNK4-unsplash.jpg": (
        0.440,
        "woman on the right",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--report-out", required=True, help="JSON report path")
    parser.add_argument("--target", type=int, default=TARGET_SIZE, help="Output size")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Context padding around the face template (default {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_DETECTION_SCORE,
        help=f"Minimum detection confidence (default {MIN_DETECTION_SCORE})",
    )
    return parser.parse_args()


def aligned_crop(
    image, landmarks: np.ndarray, target: int, margin: float
) -> tuple[np.ndarray, float]:
    """Similarity-transform the face onto a `target`x`target` canvas.

    Uses InsightFace's ArcFace 5-point template, scaled down by `1 + margin`
    and re-centred so the extra room becomes context around the face rather
    than a larger face.

    Returns the crop and the fraction of it that falls outside the source
    image. That fraction is not incidental: the sources are already tight
    crops (Finding 17), so an aligned, rotation-corrected window routinely
    reaches past their edges -- measured over a 60-image sample, 53% of
    images need some fill and the worst needs 55% of the frame.

    Those pixels are filled by **replicating the source edge**, not with
    black. A black wedge is a hard, high-contrast shape that a CNN can
    latch onto as a feature, and its size correlates with head pose --
    exactly the kind of spurious signal that quietly inflates a benchmark.
    Replication is bland by comparison. The fraction is recorded per image
    so anyone can filter on it rather than discover it later.
    """
    import cv2
    from insightface.utils.face_align import arcface_dst
    from skimage import transform as trans

    scale = target / 112.0 / (1.0 + margin)
    destination = arcface_dst * scale
    destination += (target - 112.0 * scale) / 2.0

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, destination)
    matrix = tform.params[0:2, :]

    crop = cv2.warpAffine(
        image, matrix, (target, target), borderMode=cv2.BORDER_REPLICATE
    )
    # Warp a solid mask with the same transform to measure exactly which
    # output pixels had no source pixel behind them.
    covered = cv2.warpAffine(
        np.full(image.shape[:2], 255, dtype=np.uint8),
        matrix,
        (target, target),
        borderValue=0,
    )
    return crop, float((covered == 0).mean())


def select_primary_face(faces: list, width: int, height: int) -> tuple[int, dict]:
    """Choose which detected face the image's single label describes.

    The sources are already crops the 2021 pipeline centred on its intended
    subject, so *both* size and centrality are real evidence about who that
    subject was. They are combined as `area / (1 + centre_distance)`, which
    favours a large face but penalises one off to the side.

    The two signals are also reported separately, because they do not always
    agree: on this corpus they pick the same face for 13 of 18 multi-face
    images, and every disagreement is between two comparably sized faces
    (area ratio 1.02-1.80x) where "largest" is close to arbitrary. Those
    cases are marked `ambiguous` so a human can overrule the heuristic
    instead of trusting it blindly.
    """
    import numpy as np

    centre_x, centre_y = width / 2, height / 2
    areas, distances = [], []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        areas.append(float((x2 - x1) * (y2 - y1)))
        distances.append(
            float(np.hypot((x1 + x2) / 2 - centre_x, (y1 + y2) / 2 - centre_y))
        )

    diagonal = float(np.hypot(width, height))
    scores = [a / (1.0 + d / diagonal) for a, d in zip(areas, distances)]
    chosen = int(np.argmax(scores))
    largest = int(np.argmax(areas))
    most_central = int(np.argmin(distances))
    ranked = sorted(areas, reverse=True)
    area_ratio = (
        ranked[0] / ranked[1] if len(ranked) > 1 and ranked[1] else float("inf")
    )

    return chosen, {
        "faces": len(faces),
        "chosen_index": chosen,
        "largest_index": largest,
        "most_central_index": most_central,
        "rules_agree": largest == most_central,
        "area_ratio": round(area_ratio, 3),
        "ambiguous": largest != most_central and area_ratio < AMBIGUOUS_AREA_RATIO,
    }


def main() -> None:
    args = parse_args()
    import cv2
    from insightface.app import FaceAnalysis

    v3_dir = Path(args.v3).expanduser().resolve()
    native_images = v3_dir / "images"
    output_dir = v3_dir / f"cropped_{args.target}"
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(native_images / "metadata.parquet")
    print(f"{len(metadata)} images -> {output_images}")

    # detection only: the recognition/genderage/landmark models are loaded by
    # default and are pure overhead here.
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    rows: list[dict] = []
    no_face: list[str] = []
    multi_face: list[dict] = []
    unreadable: list[str] = []

    for index, row in enumerate(metadata.itertuples(), start=1):
        source = native_images / row.file_name
        image = cv2.imread(str(source))
        if image is None:
            # Never swallowed silently -- Finding 3 exists because the 2021
            # pipeline did exactly that.
            unreadable.append(row.legacy_path)
            continue

        faces = [f for f in app.get(image) if f.det_score >= args.min_score]
        if len(faces) == 0:
            no_face.append(row.legacy_path)
            continue
        selection = None
        if len(faces) > 1:
            index, selection = select_primary_face(
                faces, image.shape[1], image.shape[0]
            )
            override = FACE_SELECTION_OVERRIDES.get(row.legacy_path)
            if override is not None:
                target_x, description = override
                wanted = target_x * image.shape[1]
                index = min(
                    range(len(faces)),
                    key=lambda i: abs(
                        (faces[i].bbox[0] + faces[i].bbox[2]) / 2 - wanted
                    ),
                )
                selection = {
                    **selection,
                    "chosen_index": index,
                    "ambiguous": False,
                    "maintainer_override": description,
                }
            multi_face.append(
                {
                    "image_id": row.image_id,
                    "legacy_path": row.legacy_path,
                    "scores": [round(float(f.det_score), 4) for f in faces],
                    **selection,
                }
            )
            faces = [faces[index]]

        face = faces[0]
        crop, edge_fill = aligned_crop(
            image, face.kps.astype(np.float32), args.target, args.margin
        )
        destination = output_images / f"{row.image_id}.jpg"
        cv2.imwrite(str(destination), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        box = [round(float(v), 2) for v in face.bbox]
        rows.append(
            {
                "image_id": row.image_id,
                "file_name": f"images/{row.image_id}.jpg",
                "det_score": round(float(face.det_score), 4),
                "edge_fill_fraction": round(edge_fill, 4),
                "n_faces_detected": 1 if selection is None else selection["faces"],
                "face_choice_ambiguous": bool(
                    selection is not None and selection["ambiguous"]
                ),
                "bbox_x1": box[0],
                "bbox_y1": box[1],
                "bbox_x2": box[2],
                "bbox_y2": box[3],
                "source_width": int(row.width),
                "source_height": int(row.height),
            }
        )
        if index % 250 == 0:
            print(f"  {index}/{len(metadata)}")

    crops = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    crops.to_parquet(output_images / "metadata.parquet", index=False)

    # Stale outputs must go, or an image excluded upstream keeps shipping here.
    expected = set(crops["image_id"] + ".jpg")
    for stale in output_images.iterdir():
        if (
            stale.is_file()
            and stale.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and stale.name not in expected
        ):
            stale.unlink()

    report = {
        "detector": "SCRFD-10GF (InsightFace buffalo_l)",
        "detector_input_size": list(DET_SIZE),
        "target_size": args.target,
        "margin": args.margin,
        "min_detection_score": args.min_score,
        "alignment": "5-point ArcFace similarity transform, margin-expanded",
        "border_mode": "replicate (never black -- see aligned_crop docstring)",
        "images_considered": len(metadata),
        "cropped": len(crops),
        "no_face_count": len(no_face),
        "multi_face_count": len(multi_face),
        "multi_face_ambiguous_count": sum(1 for x in multi_face if x["ambiguous"]),
        "multi_face_rule": (
            "primary face = max(area / (1 + centre_distance / image_diagonal)); "
            "largest and most-central reported separately, and disagreement "
            "between comparably sized faces is flagged ambiguous"
        ),
        "unreadable": unreadable,
        "no_face_images": no_face,
        "multi_face_images": multi_face,
        "note": (
            "Sources are already 400-600px crops (Finding 17), so these are "
            "crops of crops -- consistent alignment, not recovered detail."
        ),
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\ncropped {len(crops)} | no face {len(no_face)} | multi {len(multi_face)}")
    if unreadable:
        print(f"unreadable: {len(unreadable)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
