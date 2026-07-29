"""Build the v3 dataset structure: flat, content-addressed images with
labels only in metadata (no folder-encoded gender/ethnicity).

Why: `docs/RESTRUCTURE_PROPOSAL.md` found that the current
`{gender}/{ethnicity}/<file>` layout is the direct mechanism behind two of
the audit's largest bugs -- Finding 8's label collisions (only possible
because the same photo can be physically filed in two label folders) and
Finding 6's 252-relabeled-row gap (relabeling meant moving a file, which
happened inconsistently). Keying each image by its SHA-256 instead:

- Makes duplicate-content images (Finding 7) collapse into one row
  automatically -- 7 of Finding 8's 8 label collisions were exactly this
  (same photo, two labels) and no longer need special-casing.
- Makes relabeling a single metadata edit, not a file move.
- Removes the class of filename-parsing bug Finding 6 hit (a legacy
  filename containing a space broke path normalization); the new filename
  is always a bare hex hash.

Original filenames/paths are preserved as metadata, not discarded.

Deliberately not split into `images/train/`, `images/val/`, `images/test/`
subfolders (the pattern Hugging Face's `ImageFolder` uses for automatic
split detection): not every image has a canonical rating (some exist only
for landmarks/feature research), so a folder split would either misfile
them or need a fourth "unsplit" bucket. Split membership lives in
`ratings/aggregate/*.parquet` instead, which any consumer needs to join
against for the actual score anyway -- a folder-only split wouldn't save
that step, and would partially reintroduce "physical location implies
membership," the exact pattern this restructure removed for labels.

Needs Pillow (to record each image's real pixel dimensions -- see
`add_dimensions` and Finding 17), so run it with `--with pillow`:

    uv run --with pillow python scripts/data/build_v3_dataset.py \\
        --legacy-copy data/legacy_snapshot \\
        --v2 data/mebeauty_v2 \\
        --images-parquet reports/legacy_audit/canonical_dataset/images.parquet \\
        --output data/mebeauty_v3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Finding 13 (docs/DATASET_AUDIT.md): images of named, recognizable public
# figures, actively rated in the canonical split -- a content/consent issue,
# not a data-quality bug. Excluded permanently, not just flagged, pending a
# fuller identity-screening pass. Matched on legacy_path (not just filename)
# so this stays exact even if another image somewhere reuses either name.
EXCLUDED_LEGACY_PATHS = {
    "female/black/michelle-obama-1129160_1920.jpg": "Finding 13: photograph of Michelle Obama",
    "female/indian/deepika-padukone-2779557_1920.jpg": "Finding 13: portrait depicting Deepika Padukone",
    "female/indian/aditi-rao-hydari-1748439_1920.jpg": "Finding 13: portrait depicting Aditi Rao Hydari "
    "(found on a second, broader filename pass -- the first pass's regex only matched two-word "
    "names and missed this three-word one; treat this as evidence the screen is still incomplete, "
    "not as confirmation it's now thorough)",
    # Finding 19: 24 images contain more than one face, so it is unrecorded
    # which face the rating, gender and ethnicity labels describe. The
    # maintainer reviewed all 24 on 2026-07-28 and judged the intended subject
    # unambiguous in every case but these two. The rest are kept deliberately,
    # including the only three-face image
    # (male/hispanic/betzy-arosemena-Mx15HMZGQzY-unsplash.jpg).
    "male/hispanic/michele-seghieri-9cRe2YMORtc-unsplash.jpg": "Finding 19: two faces, intended subject ambiguous (maintainer review)",
    "female/indian/pexels-shubham-sharma-2912695.jpg": "Finding 19: two faces, intended subject ambiguous (maintainer review)",
    # Surfaced by the SCRFD pass (Finding 22), which detects second faces the
    # 2021 MTCNN run missed. Two girls side by side at near-identical scale --
    # no signal in the image says which one the rating describes.
    "female/indian/eyes-5938203_1920.jpg": "Finding 22: two faces of equal prominence, intended subject ambiguous (maintainer review)",
}

# Finding 8 (docs/DATASET_AUDIT.md): 8 images are filed under two conflicting
# gender/ethnicity folders. 7 of these are the *same photo* (byte-identical)
# filed twice -- content-addressing collapses the duplication automatically,
# but not the ambiguity about which folder's label was right. Resolved by
# the maintainer's manual visual review, 2026-07-26, keyed on the shared
# filename (not path, since the whole point is the path/folder disagrees).
# The 8th collision-report entry, shivam-singh-2_X6NMP-E_U-unsplash.jpg, is
# deliberately absent here: it's two *different* photos (different SHA-256)
# that happen to share a filename, not a real label conflict -- both of its
# folder labels are already correct for their own photo and need no
# resolution.
LABEL_COLLISION_RESOLUTIONS: dict[str, tuple[str, str]] = {
    "huu-chung-dang-lP02hkcp7H0-unsplash.jpg": ("female", "asian"),
    "julian-florez-l5rmMuK8070-unsplash.jpg": ("female", "hispanic"),
    "kunal-goswami-YHSohAq-PuI-unsplash.jpg": ("female", "indian"),
    "pexels-anna-shvets-4971982.jpg": ("male", "caucasian"),
    "pexels-moh-mckenzie-3597035.jpg": ("male", "black"),
    "raamin-ka-4lQmQ_DBbNc-unsplash.jpg": ("female", "mideastern"),
    "tobi-oshinnaike-Z7MKNGFnbOw-unsplash.jpg": ("male", "black"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-copy", required=True, help="Path to the copied legacy repo"
    )
    parser.add_argument(
        "--v2", required=True, help="Path to the existing data/mebeauty_v2/"
    )
    parser.add_argument(
        "--images-parquet",
        required=True,
        help="reports/legacy_audit/canonical_dataset/images.parquet",
    )
    parser.add_argument("--output", required=True, help="Directory for the v3 dataset")
    parser.add_argument(
        "--near-duplicates",
        default=None,
        help="Optional: reports/legacy_audit/near_duplicate_images.json from "
        "find_near_duplicate_images.py, run against a previous build's images/. "
        "Two-pass: build once, run that script, then re-run this with the report "
        "to fold the flag into metadata.parquet.",
    )
    return parser.parse_args()


def apply_near_duplicates(
    metadata_df: pd.DataFrame, report_path: str | None
) -> pd.DataFrame:
    metadata_df = metadata_df.copy()
    metadata_df["has_near_duplicate"] = False
    metadata_df["near_duplicate_image_ids"] = ""
    if report_path is None:
        return metadata_df

    report = json.loads(
        Path(report_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    partners: dict[str, set[str]] = defaultdict(set)
    for pair in report["pairs"]:
        partners[pair["image_id_a"]].add(pair["image_id_b"])
        partners[pair["image_id_b"]].add(pair["image_id_a"])

    indexed = metadata_df.set_index("image_id", drop=False)
    for image_id, others in partners.items():
        if image_id in indexed.index:
            indexed.loc[image_id, "has_near_duplicate"] = True
            indexed.loc[image_id, "near_duplicate_image_ids"] = ", ".join(
                sorted(others)
            )
    return indexed.reset_index(drop=True)


def build_metadata(images_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """One row per unique SHA-256. Returns (metadata_df, legacy_path -> image_id)."""
    path_to_image_id: dict[str, str] = dict(
        zip(images_df["image"], images_df["sha256"], strict=True)
    )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in images_df.to_dict("records"):
        groups[row["sha256"]].append(row)

    rows = []
    for image_id, members in groups.items():
        members_sorted = sorted(members, key=lambda r: r["image"])
        primary = members_sorted[0]
        others = members_sorted[1:]

        label_pairs = {(m["gender"], m["ethnicity"]) for m in members_sorted}
        has_collision = len(label_pairs) > 1

        filename = primary["image"].rsplit("/", 1)[-1]
        resolution = LABEL_COLLISION_RESOLUTIONS.get(filename)
        label_resolved = has_collision and resolution is not None
        gender, ethnicity = (
            resolution if label_resolved else (primary["gender"], primary["ethnicity"])
        )
        alternatives = sorted(
            f"{g}/{e}" for g, e in label_pairs if (g, e) != (gender, ethnicity)
        )

        provenance_source = primary
        if primary["inferred_platform"] == "unknown":
            better = next(
                (m for m in members_sorted if m["inferred_platform"] != "unknown"), None
            )
            if better is not None:
                provenance_source = better

        rows.append(
            {
                "image_id": image_id,
                # Required by Hugging Face's ImageFolder loader (load_dataset("imagefolder", ...)):
                # a metadata file needs a `file_name` column giving the path to the image file,
                # relative to the directory containing this metadata file. metadata.parquet is
                # written into images/ (a sibling of the image files, not the dataset root) --
                # verified directly that this matters: with metadata.parquet at the dataset root
                # alongside ratings/aggregate/{train,val,test}.parquet, ImageFolder's split-pattern
                # auto-detection gets confused by those filenames and produces empty phantom
                # splits instead of the real data. Bare filename, no "images/" prefix, since the
                # two are siblings.
                "file_name": f"{image_id}{primary['extension']}",
                "extension": primary["extension"],
                "legacy_filename": primary["image"].rsplit("/", 1)[-1],
                "legacy_path": primary["image"],
                "other_legacy_paths": ", ".join(m["image"] for m in others),
                "gender": gender,
                "ethnicity": ethnicity,
                "has_label_collision": has_collision,
                "label_collision_resolved": label_resolved,
                "label_collision_alternatives": ", ".join(alternatives),
                "size_bytes": primary["size_bytes"],
                "inferred_platform": provenance_source["inferred_platform"],
                "inferred_photo_id": provenance_source["inferred_photo_id"],
                "inferred_source_url": provenance_source["inferred_source_url"],
                "provenance_confidence": provenance_source["provenance_confidence"],
            }
        )

    metadata_df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    return metadata_df, path_to_image_id


#: JPEG quality for the two legacy PNGs, converted so every shipped image is
#: a `.jpg`. High enough that the re-encode is visually lossless.
PNG_TO_JPEG_QUALITY = 95


def convert_pngs_to_jpeg(
    metadata_df: pd.DataFrame, legacy_root: Path, output_dir: Path
) -> dict[str, str]:
    """Re-encode the legacy PNGs as JPEG, and re-derive their `image_id`.

    Two of the 2,467 images are PNG; the maintainer asked for a single format
    throughout. The conversion is not a rename: `image_id` is the SHA-256 of
    the file's bytes, so changing the bytes must change the id, or the
    dataset's content-addressing invariant silently becomes false.

    This lives in the build rather than being applied to the output directory
    once, because a one-off edit is undone by the next rebuild -- the copy
    step would simply restore the PNG from the legacy snapshot.

    Two real costs, accepted deliberately:

    - one PNG carries an alpha channel (values 254-255, imperceptible but
      present). JPEG cannot store alpha, so it is flattened onto white.
    - a lossless source is re-encoded lossily. At quality 95 this is
      invisible, but it is not reversible.

    Returns old id -> new id, so every table keyed by `image_id` can follow.
    """
    from io import BytesIO

    from PIL import Image

    images_root = legacy_root / "original_images"
    remap: dict[str, str] = {}
    for row in metadata_df[metadata_df["extension"] == ".png"].itertuples():
        with Image.open(images_root / row.legacy_path) as image:
            if image.mode in {"RGBA", "LA", "P"}:
                flattened = Image.new("RGB", image.size, (255, 255, 255))
                converted = image.convert("RGBA")
                flattened.paste(converted, mask=converted.split()[-1])
                image = flattened
            else:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=PNG_TO_JPEG_QUALITY, subsampling=0)

        payload = buffer.getvalue()
        new_id = hashlib.sha256(payload).hexdigest()
        remap[row.image_id] = new_id
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / f"{new_id}.jpg").write_bytes(payload)
        print(f"  Converted {row.legacy_path} -> .jpg ({len(payload) / 1024:.0f} KB)")

    if remap:
        mask = metadata_df["image_id"].isin(remap)
        metadata_df.loc[mask, "size_bytes"] = [
            (output_dir / "images" / f"{remap[i]}.jpg").stat().st_size
            for i in metadata_df.loc[mask, "image_id"]
        ]
        metadata_df.loc[mask, "image_id"] = metadata_df.loc[mask, "image_id"].map(remap)
        metadata_df.loc[mask, "extension"] = ".jpg"
        metadata_df.loc[mask, "file_name"] = metadata_df.loc[mask, "image_id"] + ".jpg"
    return remap


def copy_images(metadata_df: pd.DataFrame, legacy_root: Path, output_dir: Path) -> None:
    images_root = legacy_root / "original_images"
    dest_dir = output_dir / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    for row in metadata_df.itertuples():
        source = images_root / row.legacy_path
        dest = dest_dir / f"{row.image_id}{row.extension}"
        expected.add(dest.name)
        # Converted PNGs are already written by convert_pngs_to_jpeg; their
        # legacy source no longer corresponds to the shipped bytes.
        if not dest.exists() and source.suffix.lower() == row.extension:
            shutil.copy2(source, dest)

    # Rebuilding into an existing directory must also *remove* images that are
    # no longer in the metadata. Without this, growing EXCLUDED_LEGACY_PATHS
    # leaves the excluded file physically present and still loadable by
    # ImageFolder -- which for a public-figure or consent removal means the
    # image is not actually withdrawn.
    orphans = [
        path
        for path in dest_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.name not in expected
    ]
    for path in orphans:
        print(f"  Removing orphaned image no longer in metadata: {path.name}")
        path.unlink()


# The three sizes that dominate the legacy tree (2,472 of 2,495 images).
# Each is a preprocessing batch signature -- see Finding 17.
_UNIFORM_CROP_SIZES = {(400, 400), (500, 500), (600, 600)}


def add_dimensions(metadata_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Record each image's actual pixel dimensions in the metadata.

    Finding 17: the legacy `original_images/` tree is 99.1% pre-cropped,
    downsampled face images (400x400, 500x500, 600x600) rather than source
    photographs, and the filenames actively mislead about this -- a file named
    `male-4572748_1920.jpg` is stored at 400x400. Shipping the real dimensions
    means a consumer can see the heterogeneity and stratify or filter on it
    instead of trusting the name.

    Deliberately *not* resampling to a uniform size: upscaling a 400x400 crop
    to 600x600 invents detail that was never captured, and downscaling
    discards it. Consumers resize at training time anyway; the honest move is
    to hand them accurate numbers, not a uniform-looking set.
    """
    from PIL import Image  # optional dependency; see this script's docstring

    widths, heights = [], []
    for row in metadata_df.itertuples():
        with Image.open(output_dir / "images" / row.file_name) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)

    metadata_df = metadata_df.copy()
    metadata_df["width"] = widths
    metadata_df["height"] = heights
    metadata_df["megapixels"] = [
        round(w * h / 1_000_000, 4) for w, h in zip(widths, heights, strict=True)
    ]

    # The uniform sizes are preprocessing batch signatures, not coincidence:
    # 600x600 in particular maps *exactly* onto the images with no recoverable
    # provenance (75/75, and no image with provenance is 600x600), which is
    # why it is worth carrying as an explicit column rather than leaving
    # consumers to rediscover it from width/height.
    metadata_df["crop_batch"] = [
        f"uniform_{w}"
        if (w, h) in _UNIFORM_CROP_SIZES
        else ("full_image" if w * h > 1_000_000 else "other")
        for w, h in zip(widths, heights, strict=True)
    ]
    # False for the ~99% that are already-cropped faces. Named for what it
    # asserts about *this file*, not about some other file's availability:
    # a consumer wants to know "am I holding a crop or a photograph?".
    metadata_df["is_preprocessed_crop"] = [
        batch != "full_image" for batch in metadata_df["crop_batch"]
    ]
    return metadata_df


def build_landmarks(
    v2_dir: Path, path_to_image_id: dict[str, str], output_dir: Path
) -> int:
    """Write landmarks as a native list<float> column, not a comma-joined string.

    The legacy/v2 CSV stores landmarks as "x1,y1,x2,y2,...,x68,y68" text,
    which every consumer has to re-parse. Parquet/Arrow support nested list
    columns directly (the same pattern Hugging Face's own metadata format
    uses for structured fields like bounding boxes), so store it as one.
    """
    rows = list(csv.DictReader((v2_dir / "landmarks.csv").open(encoding="utf-8")))
    seen: set[str] = set()
    out_rows = []
    for row in rows:
        image_id = path_to_image_id.get(row["image"])
        if image_id is None or image_id in seen:
            continue
        seen.add(image_id)
        values = [float(v) for v in row["landmarks"].split(",") if v.strip() != ""]
        out_rows.append({"image_id": image_id, "landmarks": values})
    pd.DataFrame(out_rows).sort_values("image_id").to_parquet(
        output_dir / "landmarks.parquet", index=False
    )
    return len(out_rows)


def build_ratings(
    v2_dir: Path, path_to_image_id: dict[str, str], output_dir: Path
) -> dict[str, int]:
    ratings_out = output_dir / "ratings" / "aggregate"
    ratings_out.mkdir(parents=True, exist_ok=True)
    counts = {}
    unmapped: list[str] = []
    for split in ["train", "val", "test"]:
        rows = []
        for row in csv.DictReader(
            (v2_dir / "ratings" / f"{split}.csv").open(encoding="utf-8")
        ):
            image_id = path_to_image_id.get(row["image"])
            if image_id is None:
                unmapped.append(row["image"])
                continue
            rows.append({"image_id": image_id, "score": float(row["score"])})
        df = pd.DataFrame(rows)
        df.to_parquet(ratings_out / f"{split}.parquet", index=False)
        counts[split] = len(df)
    if unmapped:
        print(
            f"  WARNING: {len(unmapped)} rating rows had no matching image_id: {unmapped}"
        )
    return counts


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_copy).expanduser().resolve()
    v2_dir = Path(args.v2).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_parquet(args.images_parquet)
    print(f"Source: {len(images_df)} legacy-path rows")

    excluded_mask = images_df["image"].isin(EXCLUDED_LEGACY_PATHS)
    if excluded_mask.any():
        for path in images_df.loc[excluded_mask, "image"]:
            print(f"  Excluding {path}: {EXCLUDED_LEGACY_PATHS[path]}")
        images_df = images_df.loc[~excluded_mask].reset_index(drop=True)

    metadata_df, path_to_image_id = build_metadata(images_df)
    # Re-encode the legacy PNGs before anything downstream reads an image_id:
    # the id is the file hash, so converting the bytes must change it, and
    # every join key has to follow in the same pass.
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    png_remap = convert_pngs_to_jpeg(metadata_df, legacy_root, output_dir)
    if png_remap:
        path_to_image_id = {
            path: png_remap.get(image_id, image_id)
            for path, image_id in path_to_image_id.items()
        }
    metadata_df = apply_near_duplicates(metadata_df, args.near_duplicates)
    if args.near_duplicates:
        print(
            f"  {metadata_df['has_near_duplicate'].sum()} images flagged as near-duplicates"
        )

    print("Copying images/ (flat, content-addressed) ...")
    copy_images(metadata_df, legacy_root, output_dir)
    metadata_df = add_dimensions(metadata_df, output_dir)
    size_counts = (
        metadata_df.groupby(["width", "height"]).size().sort_values(ascending=False)
    )
    print(f"  dimensions recorded; most common: {dict(list(size_counts.items())[:3])}")
    print(
        f"  {int((metadata_df['megapixels'] > 1.0).sum())} images over 1MP "
        f"(the rest are pre-cropped faces -- Finding 17)"
    )

    # Written inside images/, not the dataset root -- see the comment on
    # `file_name` in build_metadata for why (verified: it matters).
    metadata_df.to_parquet(output_dir / "images" / "metadata.parquet", index=False)
    print(
        f"images/metadata.parquet: {len(metadata_df)} unique images (content-deduped from {len(images_df)})"
    )
    unresolved_collisions = (
        metadata_df["has_label_collision"] & ~metadata_df["label_collision_resolved"]
    ).sum()
    print(
        f"  {metadata_df['has_label_collision'].sum()} carry a label collision "
        f"({metadata_df['label_collision_resolved'].sum()} resolved, "
        f"{unresolved_collisions} still need a human choice)"
    )

    landmark_count = build_landmarks(v2_dir, path_to_image_id, output_dir)
    print(f"landmarks.parquet: {landmark_count} rows")

    rating_counts = build_ratings(v2_dir, path_to_image_id, output_dir)
    print(f"ratings/aggregate/: {rating_counts}")

    summary = {
        "unique_images": len(metadata_df),
        "label_collisions_total": int(metadata_df["has_label_collision"].sum()),
        "label_collisions_resolved": int(metadata_df["label_collision_resolved"].sum()),
        "label_collisions_remaining": int(unresolved_collisions),
        "landmark_rows": landmark_count,
        "rating_counts": rating_counts,
    }
    (output_dir / "COVERAGE.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    reproduce_dir = output_dir / "reproduce_legacy_baseline"
    reproduce_dir.mkdir(exist_ok=True)
    (reproduce_dir / "README.md").write_text(
        "See docs/REPRODUCE_LEGACY_BASELINE.md in the source repository for how to "
        "regenerate MTCNN/OpenCV crops, FaceNet-512 embeddings, and the geometric-ratio "
        "features from images/ + landmarks.parquet -- deliberately not shipped as data.\n",
        encoding="utf-8",
    )

    readme = f"""# MEBeauty v3 (local, unreleased)

Flat, content-addressed images with labels only in metadata -- see
`docs/RESTRUCTURE_PROPOSAL.md` for why. **Local only** -- not uploaded, not
licensed, not committed to git.

## Coverage

```json
{json.dumps(summary, indent=2)}
```

## Layout

- `images/<image_id>.<ext>` + `images/metadata.parquet` -- {len(metadata_df)} unique
  images, keyed by SHA-256, with metadata as a *sibling* of the image
  files (not at the dataset root -- verified that matters: a sibling
  `ratings/aggregate/{{train,val,test}}.parquet` elsewhere in this tree
  confuses Hugging Face's `ImageFolder` split auto-detection otherwise).
  `load_dataset("imagefolder", data_dir="images")` works directly.
  metadata.parquet has image_id, `file_name` (the ImageFolder-required
  link to the image file), legacy_filename/legacy_path (provenance),
  gender, ethnicity, has_label_collision + label_collision_alternatives
  (still a human decision where true), inferred source provenance.
  Content-duplicate images (same photo, different legacy filename/label)
  collapse to one file automatically.
- `landmarks.parquet` -- image_id, landmarks (native `list<float>`, 136
  values = 68 points, not a string -- queryable/usable without re-parsing).
- `ratings/aggregate/{{train,val,test}}.parquet` -- canonical split, keyed
  by image_id. Run `scripts/data/enrich_label_provenance.py` afterwards to
  add the per-label rater support columns (`n_ratings`, `score_std`,
  `score_mean`, `score_delta`, `diverges_from_score_mean`); this build writes
  `image_id`/`score` only. See Finding 20.
- `ratings/distributions.parquet` -- per-image soft labels over the 1-10
  scale, for label distribution learning (run
  `scripts/data/build_rating_distributions.py` after the by-rater table).
- `ratings/by_rater/ratings_by_rater.parquet` -- individual pseudonymized
  rater scores (run `scripts/data/build_ratings_by_rater.py` separately;
  see `reconciliation_report.json` alongside it for source selection and
  what was dropped/why).
- `reproduce_legacy_baseline/` -- pointer to `docs/REPRODUCE_LEGACY_BASELINE.md`;
  crops/embeddings/geometric-features are deliberately not shipped as data.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
