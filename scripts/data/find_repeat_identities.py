"""Find people who appear in more than one image.

Two separate reasons this matters, and they are easy to conflate:

1. **Benchmark validity.** If the same person appears in both `train` and
   `test`, a model can score well by recognising the person rather than by
   predicting attractiveness. That is train/test leakage, and no previous
   check would have caught it: Finding 7 hashes bytes, Finding 12 hashes
   pixels perceptually. Both find the same *photograph* twice. Neither finds
   the same *person* photographed twice.

2. **Public-figure screening.** A person who appears repeatedly is more
   likely to be someone the collectors found many photographs of -- which
   correlates with being a public figure. This is a weak, indirect signal,
   not a determination, and is treated as such: it ranks images for human
   review and names nobody.

**Model: ArcFace w600k_r50 (InsightFace `buffalo_l`).** Trained on WebFace600K
-- roughly 600,000 identities, against the 8,631 of the VGGFace2 classifier
used for Finding 18's screen. It produces embeddings rather than names, so it
answers "are these two faces the same person?" and never "who is this?".

**Threshold.** Cosine similarity above `--threshold` marks a pair as the same
person. 0.5 is InsightFace's own conventional operating point for this model
family. The distance distribution is written to the report so the choice can
be checked against this data rather than taken on faith, and every match is
listed for human confirmation -- a same-identity claim about real people is
not something to assert from a cosine score alone.

    uv run --with insightface --with onnxruntime --with opencv-python-headless \\
        python scripts/data/find_repeat_identities.py \\
            --v3 data/mebeauty_v3 \\
            --output reports/legacy_audit/repeat_identities.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Cosine similarity above which two faces are treated as the same person.
#: InsightFace's conventional operating point for ArcFace embeddings.
DEFAULT_THRESHOLD = 0.5

#: Detector input size, matching `build_face_crops.py` -- the default 640
#: detects nothing on this corpus of 400-600px crops.
DET_SIZE = (320, 320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity for a same-person match (default {DEFAULT_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2
    from insightface.app import FaceAnalysis

    v3_dir = Path(args.v3).expanduser().resolve()
    crops_dir = v3_dir / "cropped_256" / "images"
    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    path_by_id = dict(zip(metadata["image_id"], metadata["legacy_path"]))

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    image_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    skipped: list[str] = []

    crop_paths = sorted(crops_dir.glob("*.jpg"))
    print(f"Embedding {len(crop_paths)} aligned crops ...")
    for index, path in enumerate(crop_paths, start=1):
        image = cv2.imread(str(path))
        faces = app.get(image)
        if not faces:
            # The crop is already a detected, aligned face, so a failure here
            # is a genuine oddity worth reporting rather than skipping quietly.
            skipped.append(path.stem)
            continue
        face = max(faces, key=lambda f: f.det_score)
        vector = face.normed_embedding.astype(np.float32)
        image_ids.append(path.stem)
        embeddings.append(vector)
        if index % 250 == 0:
            print(f"  {index}/{len(crop_paths)}")

    matrix = np.vstack(embeddings)
    print(f"{len(matrix)} embeddings; comparing every pair ...")

    # Embeddings are L2-normalised, so the dot product is cosine similarity.
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)

    upper = np.triu_indices(len(matrix), k=1)
    all_scores = similarity[upper]
    pairs = [
        (int(i), int(j), float(similarity[i, j]))
        for i, j in zip(*np.where(similarity >= args.threshold))
        if i < j
    ]
    pairs.sort(key=lambda p: -p[2])

    # Union-find: collapse pairwise matches into identity groups, so one
    # person photographed four times is a single group, not six pairs.
    parent = list(range(len(matrix)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    groups: dict[int, list[int]] = {}
    for index in range(len(matrix)):
        groups.setdefault(find(index), []).append(index)
    identity_groups = [g for g in groups.values() if len(g) > 1]
    identity_groups.sort(key=len, reverse=True)

    splits = {
        split: set(
            pd.read_parquet(v3_dir / "ratings" / "aggregate" / f"{split}.parquet")[
                "image_id"
            ]
        )
        for split in ("train", "val", "test")
    }

    def split_of(image_id: str) -> str:
        for name, members in splits.items():
            if image_id in members:
                return name
        return "unrated"

    group_records = []
    for group in identity_groups:
        members = [image_ids[i] for i in group]
        member_splits = sorted({split_of(m) for m in members})
        rated = [s for s in member_splits if s != "unrated"]
        group_records.append(
            {
                "size": len(members),
                "spans_splits": len(rated) > 1,
                "splits": member_splits,
                "images": [
                    {
                        "image_id": m,
                        "legacy_path": path_by_id.get(m),
                        "split": split_of(m),
                    }
                    for m in members
                ],
                "min_similarity": round(
                    float(
                        min(similarity[i, j] for i in group for j in group if i != j)
                    ),
                    4,
                ),
            }
        )

    leaking = [g for g in group_records if g["spans_splits"]]
    report = {
        "model": "ArcFace w600k_r50 (InsightFace buffalo_l)",
        "identities_in_training_set": "~600K (WebFace600K)",
        "threshold": args.threshold,
        "images_embedded": len(matrix),
        "images_skipped": skipped,
        "similarity_distribution": {
            "max": round(float(all_scores.max()), 4),
            "p99_9": round(float(np.percentile(all_scores, 99.9)), 4),
            "p99": round(float(np.percentile(all_scores, 99)), 4),
            "median": round(float(np.median(all_scores)), 4),
        },
        "matching_pairs": len(pairs),
        "identity_groups": len(group_records),
        "groups_spanning_splits": len(leaking),
        "note": (
            "A same-person match is a claim about real people; every group "
            "below needs human confirmation before being acted on. High "
            "similarity is evidence, not proof."
        ),
        "groups": group_records,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nmatching pairs: {len(pairs)}")
    print(f"identity groups: {len(group_records)}")
    print(f"groups spanning splits (leakage risk): {len(leaking)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
