"""Find near-duplicate images that exact SHA-256 matching (Finding 7) misses.

Byte-identical duplicate detection only catches a photo saved twice
unchanged. It cannot catch the same photo (or the same photoshoot moment)
re-saved at different compression/size, which is common with stock photos.
This uses perceptual hashing (`imagehash.phash`, 16x16 = 256-bit) instead,
which is robust to that kind of re-encoding.

Requires an extra dependency deliberately kept out of pyproject.toml:

    uv run --with imagehash --with pillow python scripts/data/find_near_duplicate_images.py \\
        --images data/mebeauty_v3/images --threshold 20 \\
        --output reports/legacy_audit/near_duplicate_images.json

`--threshold` is a Hamming distance out of 256 bits. 20 is deliberately
tight: checked directly against this dataset's actual distribution, not
picked blind -- the closest genuine near-duplicate pair found was at
distance 16, and the very next-closest pair (58) was visually confirmed to
be two different people with similar styling, not a duplicate. There is a
wide, clean gap between those two numbers in this dataset; 20 sits in it.
A looser threshold on a different dataset would need re-validating the
same way, not reusing this number blindly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imagehash
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", required=True, help="Directory of images to scan (flat)"
    )
    parser.add_argument(
        "--threshold", type=int, default=20, help="Max Hamming distance (of 256 bits)"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Hashing {len(files)} images (perceptual hash, 16x16)...")

    hashes: dict[str, imagehash.ImageHash] = {}
    for path in files:
        with Image.open(path) as image:
            hashes[path.stem] = imagehash.phash(image, hash_size=16)

    items = list(hashes.items())
    pairs = []
    for i in range(len(items)):
        id1, h1 = items[i]
        for j in range(i + 1, len(items)):
            id2, h2 = items[j]
            distance = h1 - h2
            if distance <= args.threshold:
                pairs.append(
                    {
                        "image_id_a": id1,
                        "image_id_b": id2,
                        "hamming_distance": int(distance),
                    }
                )

    pairs.sort(key=lambda p: p["hamming_distance"])
    report = {
        "images_scanned": len(files),
        "threshold": args.threshold,
        "near_duplicate_pair_count": len(pairs),
        "pairs": pairs,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{len(pairs)} near-duplicate pair(s) found at distance <= {args.threshold}")
    for pair in pairs:
        print(
            f"  {pair['hamming_distance']}: {pair['image_id_a']} <-> {pair['image_id_b']}"
        )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
