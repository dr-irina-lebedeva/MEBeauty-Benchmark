"""Count detected faces per image, to size the multi-face ambiguity.

Finding 19: a 300-image sample suggested ~1.7% of images contain more than
one face. That matters because the dataset records one rating, one gender
label and one ethnicity label per *file* -- so when two people share a frame,
nothing says which face any of them describes, or which face the legacy crop
pipeline selected. This replaces the sample with an exact count.

Faces are counted with MTCNN above a confidence threshold. The threshold
matters: low-confidence detections on a tightly cropped portrait are usually
background artefacts (a face on a poster, a reflection, a partial head at the
frame edge), and counting them would overstate the problem. 0.95 is strict
enough that a second detection is very likely a real second person.

Note the interaction with Finding 17: these images are already crops, so a
second face means the *original* photograph contained at least two people and
the crop kept both. Re-cropping to isolate the intended face is not possible
from the released pixels.

    uv run --with facenet-pytorch --with torch --with pillow python \\
        scripts/data/count_faces_per_image.py \\
        --images data/legacy_snapshot/original_images \\
        --output reports/legacy_audit/faces_per_image.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image root to scan")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Minimum detection confidence to count a face",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = Path(args.images).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p
        for p in images_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"{len(paths)} images under {images_root}")

    detector = MTCNN(keep_all=True, device="cpu")
    counts: Counter[int] = Counter()
    multi_face: list[dict] = []
    unreadable: list[str] = []

    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            relative = path.relative_to(images_root).as_posix()
            try:
                with Image.open(path) as image:
                    _, probabilities = detector.detect(image.convert("RGB"))
            except (OSError, ValueError) as error:
                # Reported, never silently skipped -- Finding 3 exists because
                # the legacy pipeline swallowed exactly this.
                unreadable.append(f"{relative}: {type(error).__name__}")
                continue

            faces = (
                0
                if probabilities is None
                else int(sum(1 for p in probabilities if p and p > args.confidence))
            )
            counts[faces] += 1
            if faces >= 2:
                multi_face.append(
                    {
                        "image": relative,
                        "faces": faces,
                        "confidences": [
                            round(float(p), 4)
                            for p in probabilities
                            if p and p > args.confidence
                        ],
                    }
                )
            if index % 250 == 0:
                print(f"  {index}/{len(paths)}")

    scanned = sum(counts.values())
    report = {
        "images_scanned": scanned,
        "unreadable": unreadable,
        "confidence_threshold": args.confidence,
        "faces_per_image": {str(k): counts[k] for k in sorted(counts)},
        "multi_face_count": len(multi_face),
        "multi_face_fraction": round(len(multi_face) / scanned, 4) if scanned else None,
        "zero_face_count": counts.get(0, 0),
        "multi_face_images": multi_face,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nfaces per image:")
    for faces in sorted(counts):
        print(f"  {faces}: {counts[faces]}")
    print(
        f"\nmulti-face: {len(multi_face)} of {scanned} "
        f"({len(multi_face) / scanned:.2%})"
        if scanned
        else ""
    )
    if unreadable:
        print(f"unreadable: {len(unreadable)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
