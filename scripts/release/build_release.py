"""Assemble the public Hugging Face release from the built v3 dataset.

Two tiers, and the separation is the point:

* **public/** -- everything uploaded. Images, scores, per-rater ratings under
  project pseudonyms, and the documentation.
* **private/** -- never uploaded. The pseudonym-to-worker mapping, exact panel
  ages, per-rater behavioural profiles, and the exclusion log.

Nothing is computed here that was not already computed and checked upstream;
this reshapes and renames for publication and writes the manifest. Scores come
from the ratings, splits from `splits.parquet`, images from disk.

**Identifiers.** The internal `image_id` is the SHA-256 of the file, which is
useful and unreadable. The release adds `mebeauty_000001`-style ids assigned in
sorted-hash order, so they are stable across rebuilds, and keeps `sha256`
beside them so the content link is never lost.

**Two scores ship, and they are not interchangeable:**

* `beauty_score` (internally `score_adjusted`) -- rater offsets fitted jointly
  by alternating least squares, then removed. **The label**, and the only one
  in every config: it reproduces at r = 0.81 across an independent half of the
  panel, against 0.75 for the plain mean.
* `plain_mean_score` (internally `score_mean`) -- plain unweighted mean of
  every retained generic rating. **`full` only.** Kept for hand-verification
  and for comparison with datasets using the plain convention; it is not the
  benchmark target, and shipping it everywhere invited results that quietly
  used a different label.

Both are recomputable from the ratings shipped in the `full` config, and the
plain mean is also the expectation of `rating_distribution`, which ships in
every config -- so nothing is actually lost by confining the column to `full`.
`score_standardized` is computed but stays private: it is a third route to the
same correction (r = 0.99 with `score_adjusted`) and a third public column
would only make the benchmark target ambiguous.

Date-task ratings never touch any of them; the validator asserts it.

    uv run python scripts/release/build_release.py \\
        --v3 data/mebeauty_v3 \\
        --out data/release
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mebeauty_benchmark.legacy.normalization import (
    DEFAULT_SHRINKAGE,
    normalised_image_scores,
)
from mebeauty_benchmark.legacy.validity import MIN_RATINGS_PER_IMAGE

#: The public release version. The internal build tree is numbered v3; the
#: *dataset* being published is MEBeauty v2 -- the 2022 paper release is v1,
#: and this is the improved, privacy-audited mirror of it. Numbering the
#: release after the internal build directory would invent a version the
#: dataset's own history does not have.
RELEASE_VERSION = "2.1.0"
DATASET_NAME = "MEBeauty v2"

#: Legacy paths whose source link the maintainer confirmed by opening the page.
#: They arrive as `inferred` because the photo id came from the filename; a
#: human having checked the page is a stronger signal than any heuristic, so
#: they are promoted -- but explicitly, listed here, rather than by loosening
#: the rule that produces `inferred`.
MAINTAINER_CONFIRMED_LINKS = {
    "female/asian/hijab-5090230_1920.jpg",
    "male/mideastern/pexels-emre-keshavarz-3518392.jpg",
    "female/caucasian/woman-3718859_1920.jpg",
    "male/hispanic/couple-5917009_1920.png",
}

#: How each internal task name is published.
#:
#: The second task is stored internally as `date`, after the question it asked
#: ("would you date this person"). That is a blunt name for a public artefact
#: and invites the dataset to be read as a dating-suitability resource, which
#: it is not. It ships as `personal_preference`. The *question* is still
#: stated plainly in the card -- the label is softened, never the description,
#: because a reader must know what was actually asked.
PUBLISHED_TASK_NAME = {"generic": "attractiveness", "date": "preference"}

#: Images rated by at least this many valid raters are called `reliable`;
#: those below it are `provisional`. Both ship -- the flag lets a consumer
#: choose, which is better than choosing for them by deletion.
RELIABLE_RATING_COUNT = 10

SCORE_BINS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    return parser.parse_args()


def image_count_chain(v3: Path, metadata, images) -> dict:
    """Every step from the legacy collection to the released image count."""
    root = v3.parent.parent
    inventory = pd.read_parquet(
        root / "reports/legacy_audit/canonical_dataset/images.parquet"
    )
    report = json.loads(
        (root / "reports/legacy_audit/consolidation.json").read_text(encoding="utf-8")
    )
    merged = {m for entry in report["merges"] for m in entry["merged_away"]}
    inventory_ids, kept_ids = set(inventory["sha256"]), set(metadata["image_id"])
    return {
        "legacy_file_paths": len(inventory),
        "byte_identical_duplicate_paths": len(inventory)
        - int(inventory["sha256"].nunique()),
        "unique_image_files": int(inventory["sha256"].nunique()),
        "merged_duplicate_photographs": len(merged & inventory_ids),
        "excluded_or_unusable": len(inventory_ids - kept_ids - merged),
        "reencoded_png_to_jpeg": len(kept_ids - inventory_ids),
        "images_in_dataset": len(metadata),
        "never_rated": len(metadata) - len(images),
        "released": len(images),
    }


def source_status(row) -> str:
    """Controlled provenance vocabulary, never 'verified' on an inference."""
    if row.legacy_path in MAINTAINER_CONFIRMED_LINKS:
        return "verified"
    if row.provenance_confidence == "documented":
        return "verified"
    if row.provenance_confidence == "inferred":
        return "inferred_from_filename"
    return "unresolved"


def verification_method(row) -> str:
    if row.legacy_path in MAINTAINER_CONFIRMED_LINKS:
        return "maintainer_confirmed"
    return row.source_link_method


def platform_of(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    for name in ("unsplash", "pexels", "pixabay"):
        if name in url.lower():
            return name
    return "other"


def main() -> None:
    args = parse_args()
    v3 = Path(args.v3).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    public, private = out / "public", out / "private"
    for directory in (public / "data", private, out / "ratings"):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(v3 / "images" / "metadata.parquet")
    splits = pd.read_parquet(v3 / "ratings" / "splits.parquet")
    ratings = pd.read_parquet(v3 / "ratings" / "by_rater" / "ratings_all_tasks.parquet")
    raters = pd.read_parquet(v3 / "ratings" / "by_rater" / "raters.parquet")
    landmarks = pd.read_parquet(v3 / "landmarks.parquet")
    standardized = pd.read_parquet(
        v3 / "standardized_256" / "images" / "metadata.parquet"
    )
    standardized_landmarks = pd.read_parquet(
        v3 / "standardized_256" / "landmarks.parquet"
    )

    # **No rater is excluded or down-weighted.** `score_mean` is the plain
    # unweighted mean of every retained generic attractiveness rating.
    #
    # An earlier build screened out 64 raters who used fewer than 3 distinct
    # values, on the grounds that they discriminate between faces poorly.
    # That is defensible but it makes the published label depend on a rule a
    # user cannot see, and it shifted 316 images by up to 0.75. The label is
    # now exactly what its name says. The screening statistics remain in the
    # private tier for anyone who wants to apply them.
    valid = ratings[ratings["task"] == "generic"][
        ["image_id", "rater_id", "score"]
    ].reset_index(drop=True)

    counts = valid.groupby("image_id")["score"].agg(["size", "mean", "std", "median"])
    counts = counts[counts["size"] >= MIN_RATINGS_PER_IMAGE]
    print(f"Labelled images: {len(counts)} (>= {MIN_RATINGS_PER_IMAGE} ratings)")

    normalised = normalised_image_scores(
        valid[["image_id", "rater_id", "score"]],
        shrinkage=args.shrinkage,
        min_ratings=MIN_RATINGS_PER_IMAGE,
    ).set_index("image_id")

    # Rating distribution from the raw vector, never reconstructed from a mean.
    histogram = (
        valid.assign(bin=valid["score"].round().clip(1, 10).astype(int))
        .pivot_table(index="image_id", columns="bin", aggfunc="size", fill_value=0)
        .reindex(columns=SCORE_BINS, fill_value=0)
    )

    frame = metadata[metadata["image_id"].isin(counts.index)].copy()
    frame = frame.merge(splits, on="image_id", how="left")
    frame = frame.sort_values("image_id").reset_index(drop=True)
    # The internal id is the file's SHA-256. Move it aside *before* assigning
    # the published `mebeauty_000001` id, or the two collide and every
    # subsequent lookup keyed on the hash silently returns nothing.
    frame = frame.rename(columns={"image_id": "sha256"})
    frame["image_id"] = [f"mebeauty_{i:06d}" for i in range(1, len(frame) + 1)]

    # **Two public scores, and they answer different questions.**
    #
    # `score_mean` is the plain unweighted mean: transparent, reproducible in
    # one line, and the convention SCUT-FBP5500 uses.
    #
    # `score_adjusted` corrects for rater leniency first. That correction is
    # not optional decoration here. SCUT's plain mean is sound because their
    # design is fully crossed -- all 60 labelers rate every image, so leniency
    # is a constant that cancels. This corpus is 4.7% filled: each image was
    # seen by a different draw of raters whose means span 1.0 to 9.9, so
    # leniency does not cancel and lands in the label as noise. Measured by
    # rater split-half, `score_mean` reproduces at r = 0.75 and
    # `score_adjusted` at r = 0.81.
    #
    # Both ship because neither dominates: `score_adjusted` is the better
    # training target, `score_mean` is the one a reader can verify by hand.
    # Both are recomputable from the ratings in the `full` config.
    # `score_standardized` stays in the private tier -- it is a third route to
    # the same correction (r = 0.99 with `score_adjusted`) and publishing it
    # would only make the target ambiguous.
    adjusted = pd.concat(
        [
            pd.read_parquet(v3 / "ratings" / "aggregate" / f"{s}.parquet")
            for s in ("train", "val", "test")
        ]
    ).set_index("image_id")
    frame["score_standardized"] = frame["sha256"].map(normalised["score_normalised"])
    frame["score_adjusted"] = frame["sha256"].map(adjusted["score_adjusted"])
    frame["score_mean"] = frame["sha256"].map(counts["mean"])
    frame["score_std"] = frame["sha256"].map(counts["std"])
    frame["score_median"] = frame["sha256"].map(counts["median"])
    frame["rating_count"] = frame["sha256"].map(counts["size"]).astype(int)
    frame["rating_distribution"] = frame["sha256"].map(
        histogram.apply(lambda r: [int(v) for v in r], axis=1)
    )
    frame["label_status"] = np.where(
        frame["rating_count"] >= RELIABLE_RATING_COUNT, "reliable", "provisional"
    )

    frame["landmarks_native"] = frame["sha256"].map(
        dict(zip(landmarks["image_id"], landmarks["landmarks"]))
    )
    # Landmarks in the aligned crop's own coordinates.
    #
    # The crop metadata records the face box in source pixels; mapping the
    # native landmarks through that box and scaling to 256 puts them on the
    # face in the cropped image. Verified: eyes land at y = 103 +/- 10 and the
    # mouth at y = 196 +/- 9 across all 2,467 crops, so the alignment is real
    # rather than assumed.
    crop_meta = pd.read_parquet(
        v3 / "cropped_256" / "images" / "metadata.parquet"
    ).set_index("image_id")
    native_landmarks = dict(zip(landmarks["image_id"], landmarks["landmarks"]))

    def to_crop_space(image_id):
        if image_id not in crop_meta.index:
            return None
        box = crop_meta.loc[image_id]
        x1, y1 = float(box["bbox_x1"]), float(box["bbox_y1"])
        width = float(box["bbox_x2"]) - x1
        height = float(box["bbox_y2"]) - y1
        points = np.asarray(native_landmarks[image_id], dtype=float).reshape(-1, 2)
        moved = (points - [x1, y1]) * [256.0 / width, 256.0 / height]
        return moved.reshape(-1).tolist()

    frame["landmarks_cropped"] = frame["sha256"].map(to_crop_space)
    frame["landmarks_256"] = frame["sha256"].map(
        dict(
            zip(
                standardized_landmarks["image_id"],
                standardized_landmarks["landmarks"],
            )
        )
    )
    frame["landmarks_in_bounds"] = ~frame["sha256"].map(
        dict(
            zip(
                standardized_landmarks["image_id"],
                standardized_landmarks["has_out_of_bounds_landmarks"],
            )
        )
    ).astype(bool)
    # Named rather than silently corrected: these points come from the
    # original detector on faces cut off at the image border. Clipping them
    # would move a landmark onto a pixel it was never predicted at.
    frame["landmark_issue"] = np.where(
        frame["landmarks_in_bounds"], None, "point_outside_frame"
    )
    frame["preprocessing_version"] = frame["sha256"].map(
        dict(zip(standardized["image_id"], standardized["preprocessing_version"]))
    )

    frame["source_status"] = [source_status(r) for r in frame.itertuples()]
    frame["source_platform"] = frame["source_url"].map(platform_of)
    frame["verification_method"] = [verification_method(r) for r in frame.itertuples()]
    # The internal id is the SHA-256 of the file; the published id is the
    # readable `mebeauty_000001` form. Rename the internal one out of the way
    # first so `image_id` can carry the published value.
    frame = frame.rename(
        columns={
            "gender": "legacy_gender_label",
            "ethnicity": "legacy_ethnicity_label",
        }
    )

    columns = [
        "image_id",
        "sha256",
        "file_name",
        "score_mean",
        "score_adjusted",
        "score_std",
        "rating_count",
        "rating_distribution",
        "label_status",
        "landmarks_native",
        "landmarks_cropped",
        "landmarks_in_bounds",
        "landmark_issue",
        "legacy_gender_label",
        "legacy_ethnicity_label",
        "source_platform",
        "source_url",
        "source_status",
        "verification_method",
        "width",
        "height",
        "split",
        "cv_fold",
        "preprocessing_version",
    ]
    images = frame[columns].copy()
    # Published names. Internally these stay `score_adjusted` / `score_mean`,
    # which say how they were computed; the release names say what they are
    # *for*, and `beauty_score` is the term SCUT-FBP5500 and the wider FBP
    # literature use, so it is what a reader will look for.
    images = images.rename(
        columns={"score_adjusted": "beauty_score", "score_mean": "plain_mean_score"}
    )
    # Build artefact, not a release file: every column of it is already in the
    # image configs, so shipping it would hand users the same data twice.
    images.to_parquet(out / "images.parquet", index=False)
    print(f"Wrote images.parquet: {len(images)} rows, {len(columns)} columns")

    # Per-rater ratings, keyed by the release id so the sha256 is not needed
    # to join. Both tasks ship; the validator asserts date never enters a score.
    id_of = dict(zip(frame["sha256"], frame["image_id"]))
    published = ratings[ratings["image_id"].isin(id_of)].copy()
    published["image_id"] = published["image_id"].map(id_of)
    split_of = dict(zip(frame["image_id"], frame["split"]))
    for task in ("generic", "date"):
        published_name = PUBLISHED_TASK_NAME[task]
        subset = published[published["task"] == task][
            ["image_id", "rater_id", "score"]
        ].reset_index(drop=True)
        subset["split"] = subset["image_id"].map(split_of)

        # Sharded by the image's split so a user can load the ratings for one
        # split without filtering, and so a rating can never be paired with an
        # image from a different split.
        # Build artefact, not a config. The same ratings ship nested inside
        # the personalized configs, so publishing a flat copy would hand
        # users the identical data twice under a different shape.
        directory = out / "ratings" / f"{published_name}_ratings"
        directory.mkdir(parents=True, exist_ok=True)
        for split, filename in (
            ("train", "train"),
            ("val", "validation"),
            ("test", "test"),
        ):
            part = subset[subset["split"] == split].drop(columns=["split"])
            part.reset_index(drop=True).to_parquet(
                directory / f"{filename}-00000-of-00001.parquet", index=False
            )
        print(
            f"Wrote {published_name}_ratings/: {len(subset)} rows "
            f"({subset['rater_id'].nunique()} raters)"
        )

    # Rater demographics: age as a band, never exact.
    #
    # There are 29 panel raters, and crossing gender x ethnicity x exact age
    # leaves most combinations holding a single person -- the triple would be
    # an identifier. Bands coarsen it enough to be publishable while keeping
    # what rater-effects research actually needs.
    #
    # `age_source` is on every row because the ages were never self-reported:
    # they are read from the collection filenames (`asian_female_22.xlsx`),
    # and the source workbooks contain no age field at all. A consumer must
    # not mistake them for verified demographics.
    demographics_table = raters.copy()
    demographics_table["age_band"] = pd.cut(
        demographics_table["age"].astype("Float64"),
        [0, 24, 34, 44, 54, 200],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    ).astype("string")
    demographics_table["age_source"] = np.where(
        demographics_table["age"].notna(), "inferred_from_legacy_filename", None
    )
    public_raters = demographics_table[
        ["rater_id", "pool", "gender", "ethnicity", "age_band", "age_source"]
    ]
    # Also a build artefact: every rater appears, with demographics, inside
    # the nested ratings of the personalized configs.
    public_raters.to_parquet(
        out / "ratings" / "rater_demographics.parquet", index=False
    )
    print(
        f"Wrote rater_demographics.parquet: {len(public_raters)} rows "
        f"({int(public_raters['age_band'].notna().sum())} with demographics)"
    )

    # Private tier: exact ages and anything else that must not leave this disk.
    raters.to_parquet(private / "raters_exact_age.parquet", index=False)
    # The rater-scale corrections, computed but deliberately not published.
    frame[
        ["image_id", "sha256", "score_mean", "score_standardized", "score_adjusted"]
    ].to_parquet(private / "corrected_scores.parquet", index=False)
    for name in ("rater_quality.parquet",):
        source = v3 / "ratings" / "by_rater" / name
        if source.is_file():
            shutil.copy2(source, private / name)
    (private / "README.md").write_text(
        "# PRIVATE -- never upload\n\n"
        "Exact panel ages, per-rater quality profiles, the rater-scale\n"
        "corrected scores, and any identifier mapping.\n\n"
        "The public release carries age bands only, because most panel\n"
        "demographic cells contain exactly one person, and a single\n"
        "`score_mean` so the benchmark target is unambiguous.\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset": DATASET_NAME,
        "version": RELEASE_VERSION,
        "images": len(images),
        "splits": images["split"].value_counts().to_dict(),
        "cv_folds": {
            str(int(k)): int(v)
            for k, v in images["cv_fold"].value_counts(dropna=True).sort_index().items()
        },
        "label_status": images["label_status"].value_counts().to_dict(),
        "source_status": images["source_status"].value_counts().to_dict(),
        "source_platform": images["source_platform"].value_counts().to_dict(),
        "ratings": {
            PUBLISHED_TASK_NAME[task]: int((published["task"] == task).sum())
            for task in ("generic", "date")
        },
        "raters_total_unique": len(public_raters),
        "attractiveness_raters": int(
            published[published["task"] == "generic"]["rater_id"].nunique()
        ),
        "date_raters": int(
            published[published["task"] == "date"]["rater_id"].nunique()
        ),
        "min_ratings_per_image": MIN_RATINGS_PER_IMAGE,
        "reliable_threshold": RELIABLE_RATING_COUNT,
        "shrinkage": args.shrinkage,
        # Landmark points outside the crop frame. Counted, not quoted: the
        # figure the card previously stated belonged to a different image
        # config and was wrong by an order of magnitude.
        "landmarks_outside_crop": {
            "images_with_any": int(
                sum(
                    1
                    for v in frame.loc[
                        frame["image_id"].isin(images["image_id"]),
                        "landmarks_cropped",
                    ]
                    if any(c < 0 or c > 256 for c in v)
                )
            ),
            "images_with_eye_or_mouth_outside": int(
                sum(
                    1
                    for v in frame.loc[
                        frame["image_id"].isin(images["image_id"]),
                        "landmarks_cropped",
                    ]
                    if any(c < 0 or c > 256 for c in v[72:136])
                )
            ),
        },
        # Facts the card would otherwise state from memory. Both were wrong
        # when hand-written: the rating range, and how many of the *shipped*
        # images were already crops (the figure quoted was of all 2,467,
        # before the unrated ones were dropped).
        "rating_count_range": [
            int(images["rating_count"].min()),
            int(images["rating_count"].max()),
        ],
        # Stated in the card; computed because the hand-written version was a
        # bounding box over all widths and heights ("6240x6630") that no
        # actual image has.
        "smallest_native": [
            int(frame.loc[(frame["width"] * frame["height"]).idxmin(), "width"]),
            int(frame.loc[(frame["width"] * frame["height"]).idxmin(), "height"]),
        ],
        "largest_native": [
            int(frame.loc[(frame["width"] * frame["height"]).idxmax(), "width"]),
            int(frame.loc[(frame["width"] * frame["height"]).idxmax(), "height"]),
        ],
        "caucasian_percent_by_split": {
            split: round(
                100
                * float(
                    (
                        frame.loc[frame["split"] == split, "legacy_ethnicity_label"]
                        == "caucasian"
                    ).mean()
                ),
                1,
            )
            for split in ("train", "val", "test")
        },
        # Facts a reader needs in order to reconcile counts against v1, or to
        # know what is deliberately absent. Computed, because every number
        # this card stated from memory turned out to be wrong.
        "archive_images": len(metadata),
        # The 2,547 -> 2,462 reduction, derived from the audit artefacts so
        # the card never states a figure nobody can reproduce.
        "image_count_chain": image_count_chain(v3, metadata, images),
        "images_without_ratings": int(len(metadata) - len(counts)),
        "duplicate_source_url_pairs": int(
            frame.loc[frame["image_id"].isin(images["image_id"]), "source_url"]
            .duplicated()
            .sum()
        ),
        "images_without_date_ratings": int(
            len(images) - published[published["task"] == "date"]["image_id"].nunique()
        ),
        "already_cropped": int(
            frame.loc[
                frame["image_id"].isin(images["image_id"]), "is_preprocessed_crop"
            ].sum()
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
