"""Separate genuine duplicate photographs from repeat appearances of a person.

`find_repeat_identities.py` reports that two images show the same face. That
is not the same question as whether they are the same *photograph*, and the
two need opposite treatment:

- **Same photograph**, saved twice under different names and cropped or
  colour-graded differently, is redundant data. It inflates a benchmark if
  the copies straddle a split, and its ratings belong to one image.
- **Same person, different photograph** is legitimate: two real observations
  of one face under different pose and lighting. Deleting either would
  discard genuine data. It only needs both copies in the same split.

Face similarity cannot separate them -- it is designed to be invariant to
exactly the pose and lighting differences that distinguish the two cases.

**The test used here is the background.** Both images are aligned on their
facial landmarks. If they are the same photograph, that alignment also lines
up everything *around* the face; if they are different photographs, the face
matches and the surroundings do not. So the discriminator is the correlation
between the two aligned crops **outside** the face region.

The face region itself is excluded deliberately: it correlates highly in both
cases and would wash out the signal.

    uv run --with insightface --with onnxruntime --with opencv-python-headless \\
        python scripts/data/classify_duplicate_pairs.py \\
            --v3 data/mebeauty_v3 \\
            --identities reports/legacy_audit/repeat_identities.json \\
            --output reports/legacy_audit/duplicate_classification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Face-similarity floor for a pair to be considered at all. Deliberately
#: permissive: the background test below does the real work, so casting wide
#: here costs only compute, while casting narrow would hide duplicates.
MIN_FACE_SIMILARITY = 0.5

#: Background correlation above which two images are judged the same
#: photograph. Validated against this data -- see the report's
#: `background_correlation_distribution`, which is bimodal with a clear gap.
SAME_PHOTO_CORRELATION = 0.85

#: Fraction of the aligned crop's half-width treated as "the face", and thus
#: masked out of the background comparison.
FACE_RADIUS_FRACTION = 0.62


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--identities", required=True, help="repeat_identities.json")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SAME_PHOTO_CORRELATION,
        help=f"Background correlation for 'same photo' (default {SAME_PHOTO_CORRELATION})",
    )
    return parser.parse_args()


def background_mask(size: int) -> np.ndarray:
    """A boolean mask selecting everything outside the central face region."""
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2.0
    radius = size * 0.5 * FACE_RADIUS_FRACTION
    return ((xx - centre) ** 2 + (yy - centre) ** 2) > radius**2


def background_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation between the two crops' background pixels."""
    x = a[mask].astype(np.float64).ravel()
    y = b[mask].astype(np.float64).ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        # A flat studio backdrop carries no information either way; treat it
        # as undecidable rather than as evidence of a match.
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    import cv2

    v3_dir = Path(args.v3).expanduser().resolve()
    crops = v3_dir / "cropped_256" / "images"
    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    path_by_id = dict(zip(metadata["image_id"], metadata["legacy_path"]))

    identities = json.loads(Path(args.identities).read_text())
    groups = [
        g for g in identities["groups"] if g["min_similarity"] >= MIN_FACE_SIMILARITY
    ]

    cache: dict[str, np.ndarray] = {}

    def crop_of(image_id: str) -> np.ndarray | None:
        if image_id not in cache:
            image = cv2.imread(str(crops / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            cache[image_id] = image
        return cache[image_id]

    mask = None
    records = []
    for group in groups:
        members = [i["image_id"] for i in group["images"]]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                a, b = crop_of(first), crop_of(second)
                if a is None or b is None:
                    continue
                if mask is None:
                    mask = background_mask(a.shape[0])
                correlation = background_correlation(a, b, mask)
                records.append(
                    {
                        "image_a": first,
                        "image_b": second,
                        "path_a": path_by_id.get(first),
                        "path_b": path_by_id.get(second),
                        "face_similarity": group["min_similarity"],
                        "background_correlation": (
                            None if np.isnan(correlation) else round(correlation, 4)
                        ),
                    }
                )

    scored = [r for r in records if r["background_correlation"] is not None]
    same_photo = [r for r in scored if r["background_correlation"] >= args.threshold]
    same_person = [r for r in scored if r["background_correlation"] < args.threshold]
    undecidable = [r for r in records if r["background_correlation"] is None]

    values = np.array([r["background_correlation"] for r in scored])
    report = {
        "method": (
            "Align both images on facial landmarks, then correlate the pixels "
            "OUTSIDE the face. Same photograph -> background aligns too; same "
            "person, different photograph -> it does not."
        ),
        "same_photo_threshold": args.threshold,
        "pairs_examined": len(records),
        "same_photograph": len(same_photo),
        "same_person_different_photo": len(same_person),
        "undecidable_flat_background": len(undecidable),
        "background_correlation_distribution": {
            "p10": round(float(np.percentile(values, 10)), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "median": round(float(np.median(values)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
        },
        "same_photograph_pairs": sorted(
            same_photo, key=lambda r: -r["background_correlation"]
        ),
        "same_person_pairs": sorted(
            same_person, key=lambda r: -r["background_correlation"]
        ),
        "undecidable_pairs": undecidable,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"pairs examined            : {len(records)}")
    print(f"  same photograph         : {len(same_photo)}")
    print(f"  same person, diff photo : {len(same_person)}")
    print(f"  undecidable (flat bg)   : {len(undecidable)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
