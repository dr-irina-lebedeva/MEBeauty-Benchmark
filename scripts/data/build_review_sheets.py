"""Build contact sheets so every image can be reviewed by a human.

No automated method available here can establish that a dataset contains no
public figures. A face model can say two faces match; it cannot say who
someone is without a labelled reference set of celebrity photographs, and the
identity classifier used in Finding 18 covers 8,631 people skewed toward
English-language media. Both are triage. Only human recognition decides.

So this makes the human pass tractable rather than pretending to replace it:
every image in the dataset, tiled into numbered sheets, **ordered so the
likeliest candidates come first**. The ordering is Finding 18's aligned
perceptual screen -- max softmax over a celebrity-trained classifier -- which
is validated: all three known public figures ranked inside the top 19%.
Images the screen never scored follow afterwards, so the sheets still cover
100% of the dataset and reviewing can stop wherever the reviewer judges the
returns have run out.

Each tile is captioned with its index so a reviewer can call out
"sheet 3, number 17" and have it resolve to an exact file through the
accompanying CSV.

    uv run --with opencv-python-headless python \\
        scripts/data/build_review_sheets.py \\
            --v3 data/mebeauty_v3 \\
            --screen reports/legacy_audit/public_figure_screen_aligned.json \\
            --output reports/legacy_audit/review_sheets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Tiles per sheet, as columns x rows. 8x5 keeps each face large enough to
#: recognise on a normal screen while fitting 40 to a page.
GRID = (8, 5)

#: Pixel size of each tile.
TILE = 190


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument("--screen", required=True, help="Perceptual screen JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    v3_dir = Path(args.v3).expanduser().resolve()
    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")
    crops = v3_dir / "cropped_256" / "images"

    rank_by_path = {
        entry["image"]: index
        for index, entry in enumerate(
            json.loads(Path(args.screen).read_text())["top_ranked_for_review"]
        )
    }
    # Unscored images sort after every scored one, then by path so the order
    # is stable between runs.
    metadata["rank"] = metadata["legacy_path"].map(
        lambda p: rank_by_path.get(p, len(rank_by_path))
    )
    metadata = metadata.sort_values(["rank", "legacy_path"]).reset_index(drop=True)

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*"):
        stale.unlink()

    columns, rows = GRID
    per_sheet = columns * rows
    index_rows = []

    for sheet_number, start in enumerate(range(0, len(metadata), per_sheet), start=1):
        chunk = metadata.iloc[start : start + per_sheet]
        tiles = []
        for position, row in enumerate(chunk.itertuples(), start=1):
            image = cv2.imread(str(crops / f"{row.image_id}.jpg"))
            if image is None:
                image = np.zeros((TILE, TILE, 3), np.uint8)
            tile = cv2.resize(image, (TILE, TILE))
            cv2.rectangle(tile, (0, 0), (34, 18), (0, 0, 0), -1)
            cv2.putText(
                tile,
                str(position),
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            tiles.append(tile)
            index_rows.append(
                {
                    "sheet": sheet_number,
                    "position": position,
                    "legacy_path": row.legacy_path,
                    "image_id": row.image_id,
                    "screen_rank": int(row.rank),
                }
            )

        while len(tiles) < per_sheet:
            tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
        sheet = np.vstack(
            [np.hstack(tiles[r * columns : (r + 1) * columns]) for r in range(rows)]
        )
        cv2.imwrite(str(output_dir / f"sheet_{sheet_number:03d}.jpg"), sheet)

    pd.DataFrame(index_rows).to_csv(output_dir / "index.csv", index=False)
    sheets = sheet_number
    print(f"{len(metadata)} images -> {sheets} sheets of up to {per_sheet}")
    print("  ordered by celebrity-likelihood; sheet 1 is the highest risk")
    print("  index.csv maps (sheet, position) -> file")
    print(f"Folder: {output_dir}")


if __name__ == "__main__":
    main()
