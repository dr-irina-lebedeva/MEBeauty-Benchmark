"""Screen every image for likely public figures, using a celebrity-trained
face recogniser instead of filename patterns.

Finding 13 found three named public figures by matching filenames, and that
method is provably incomplete: the second pass found a case the first pass's
own regex could not structurally have caught, and it can never see an image
whose filename carries no name at all (101 images have no filename signal
whatsoever). This screens the *pixels*.

Method. `facenet-pytorch` ships InceptionResnetV1 with a classification head
over VGGFace2's 8,631 identities -- a dataset built by image-searching
celebrities, so its classes are overwhelmingly public figures. A face that
matches one of those identities with high confidence is more likely to be a
public figure than one that does not. The maximum softmax probability is
therefore used as a *ranking* signal for human review.

What this does NOT do, stated plainly because the distinction matters:

- It does not name anyone. The class-index-to-name mapping is not published
  with the weights, so the output is "resembles known identity #4823", not
  "this is X". A human must look at the ranked images.
- It is not a determination. High confidence can be a lookalike; low
  confidence does not mean the person is unknown. VGGFace2's 8,631 identities
  are a tiny fraction of all public figures.

It is therefore a triage tool that turns "review 2,547 images" into "review
the top N", and its value depends entirely on whether known positives
actually rank highly -- which `--validate` measures rather than assumes, by
checking where Finding 13's three confirmed public figures land.

    uv run --with facenet-pytorch --with torch --with pillow python \\
        scripts/data/screen_public_figures.py \\
        --images data/legacy_snapshot/original_images \\
        --output reports/legacy_audit/public_figure_screen.json --validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Finding 13's confirmed public figures, used to measure whether this
# screening method actually surfaces known positives.
KNOWN_PUBLIC_FIGURES = {
    "female/black/michelle-obama-1129160_1920.jpg": "Michelle Obama",
    "female/indian/deepika-padukone-2779557_1920.jpg": "Deepika Padukone",
    "female/indian/aditi-rao-hydari-1748439_1920.jpg": "Aditi Rao Hydari",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        required=True,
        help="Image root. Use the legacy snapshot so Finding 13's three "
        "confirmed cases are present for validation -- they are excluded "
        "from data/mebeauty_v3/.",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--pretrained",
        default="vggface2",
        choices=["vggface2", "casia-webface"],
        help="Identity set to match against",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Images per forward pass"
    )
    parser.add_argument(
        "--top", type=int, default=50, help="How many top-ranked images to report"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Report where the known public figures rank (the whole point)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Re-detect and align each face with MTCNN before classifying. The "
        "VGGFace2 weights were trained on MTCNN-aligned crops, so skipping this "
        "is a plausible reason for weak matching -- worth testing rather than "
        "assuming either way.",
    )
    return parser.parse_args()


def load_batch(paths: list[Path]) -> torch.Tensor:
    """Images are already tight face crops (Finding 17), so they are resized
    to the model's 160x160 input directly rather than re-detected -- running a
    detector over an existing crop would only crop a crop."""
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(
            torch.frombuffer(resized.tobytes(), dtype=torch.uint8)
            .reshape(160, 160, 3)
            .numpy()
            .copy()
        ).float()
        # facenet-pytorch's expected normalisation.
        tensors.append(((tensor - 127.5) / 128.0).permute(2, 0, 1))
    return torch.stack(tensors)


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

    model = InceptionResnetV1(pretrained=args.pretrained, classify=True).eval()
    print(f"matching against {model.logits.out_features} {args.pretrained} identities")

    detector = None
    if args.align:
        from facenet_pytorch import MTCNN

        detector = MTCNN(image_size=160, margin=0, post_process=True, device="cpu")
        print("aligning each face with MTCNN before classifying")

    scored: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            if detector is None:
                batch = load_batch(batch_paths)
                kept = batch_paths
            else:
                tensors, kept = [], []
                for path in batch_paths:
                    with Image.open(path) as image:
                        face = detector(image.convert("RGB"))
                    if face is not None:
                        tensors.append(face)
                        kept.append(path)
                if not tensors:
                    continue
                batch = torch.stack(tensors)
            probabilities = torch.softmax(model(batch), dim=1)
            confidence, identity = probabilities.max(dim=1)
            for path, conf, ident in zip(
                kept, confidence.tolist(), identity.tolist(), strict=True
            ):
                scored.append(
                    {
                        "image": path.relative_to(images_root).as_posix(),
                        "confidence": round(float(conf), 6),
                        "matched_identity_index": int(ident),
                    }
                )
            if (start // args.batch_size) % 20 == 0:
                print(f"  {min(start + args.batch_size, len(paths))}/{len(paths)}")

    scored.sort(key=lambda row: -row["confidence"])
    rank_by_image = {row["image"]: index + 1 for index, row in enumerate(scored)}

    validation = None
    if args.validate:
        validation = []
        for image, name in KNOWN_PUBLIC_FIGURES.items():
            rank = rank_by_image.get(image)
            validation.append(
                {
                    "image": image,
                    "known_as": name,
                    "rank": rank,
                    "percentile": round(100 * (1 - (rank - 1) / len(scored)), 2)
                    if rank
                    else None,
                    "confidence": next(
                        (r["confidence"] for r in scored if r["image"] == image), None
                    ),
                }
            )
        print("\nValidation -- where the three confirmed public figures rank:")
        for entry in validation:
            if entry["rank"] is None:
                print(f"  {entry['known_as']}: NOT FOUND in image set")
            else:
                print(
                    f"  {entry['known_as']}: rank {entry['rank']}/{len(scored)} "
                    f"(top {100 - entry['percentile']:.1f}%), "
                    f"confidence {entry['confidence']:.4f}"
                )

    report = {
        "images_scored": len(scored),
        "identity_set": args.pretrained,
        "identity_count": model.logits.out_features,
        "method": (
            "Max softmax over a celebrity-trained identity classifier, used as a "
            "ranking signal for human review. Does not name anyone and is not a "
            "determination -- see this script's docstring."
        ),
        "validation_known_public_figures": validation,
        "confidence_distribution": {
            "max": scored[0]["confidence"],
            "p99": scored[int(len(scored) * 0.01)]["confidence"],
            "p95": scored[int(len(scored) * 0.05)]["confidence"],
            "median": scored[len(scored) // 2]["confidence"],
        },
        "top_ranked_for_review": scored[: args.top],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
