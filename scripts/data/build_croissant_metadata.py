"""Generate and validate Croissant (mlcommons.org/croissant) metadata for
`data/mebeauty_v3/`, locally -- no Hugging Face upload involved.

The original project plan's stated preference (see `docs/RESTRUCTURE_PROPOSAL.md`
and earlier planning) was to let Hugging Face auto-generate Croissant
metadata on upload rather than hand-write `croissant.json`. That auto-
generation only happens *after* a dataset is pushed to the Hub, which is
blocked on the licensing decision (`docs/DATASET_AUDIT.md`, Open section).
This generates an equivalent `croissant.json` locally with the reference
`mlcroissant` library (Google's own implementation, used to validate
Croissant files across the ecosystem) so the metadata layer can be
prepared and validated now, ahead of that decision -- and re-generated
from HF directly later if preferred.

    uv run --with mlcroissant python scripts/data/build_croissant_metadata.py \\
        --v3 data/mebeauty_v3 --output data/mebeauty_v3/croissant.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlcroissant as mlc

#: Per-label rater support added by `enrich_label_provenance.py` (Finding 20).
#: Describes the canonical `score`; never replaces it.
LABEL_PROVENANCE_FIELDS = [
    (
        "n_ratings",
        mlc.DataType.INTEGER,
        (
            "Generic ratings backing this image in the per-rater layer. Null "
            "where the image has no generic rating at all."
        ),
    ),
    (
        "score_std",
        mlc.DataType.FLOAT,
        "Standard deviation of those ratings -- how much the raters disagreed.",
    ),
    (
        "score_mean",
        mlc.DataType.FLOAT,
        (
            "SCUT-FBP5500-style label: the plain unweighted mean of every "
            "generic rating, no rater excluded. Reproducible in one line from "
            "ratings_by_rater, and equal to the mean of the shipped "
            "distribution. Use this when reproducibility matters; use `score` "
            "for comparability with the published paper."
        ),
    ),
    (
        "score_delta",
        mlc.DataType.FLOAT,
        (
            "score - score_mean. Systematically nonzero because `score` is "
            "consensus-filtered and `score_mean` is not; not an error term."
        ),
    ),
    (
        "diverges_from_score_mean",
        mlc.DataType.BOOL,
        (
            "True where abs(score_delta) > 0.5, i.e. the two labels differ "
            "enough to change how the image ranks. Flags where the choice "
            "between them matters, not where either is wrong."
        ),
    ),
]


#: Soft-label columns in `ratings/distributions.parquet`, built by
#: `build_rating_distributions.py`.
DISTRIBUTION_FIELDS = [
    ("image_id", mlc.DataType.TEXT, "Content-addressed image identifier."),
    (
        "rating_type",
        mlc.DataType.TEXT,
        (
            "'generic' or 'date' -- two different questions, never pooled "
            "into one distribution."
        ),
    ),
    (
        "n_ratings",
        mlc.DataType.INTEGER,
        "Ratings backing this distribution (minimum 9).",
    ),
    (
        "counts",
        mlc.DataType.INTEGER,
        (
            "Raw rater counts per score, as a 10-element list for scores "
            "1..10. Lossless: any other statistic can be re-derived from it."
        ),
    ),
    (
        "probabilities",
        mlc.DataType.FLOAT,
        (
            "counts normalized to sum to 1 -- the soft label consumed "
            "directly by label distribution learning."
        ),
    ),
    ("mean", mlc.DataType.FLOAT, "Unfiltered mean of the ratings."),
    ("median", mlc.DataType.FLOAT, "Median of the ratings."),
    (
        "std",
        mlc.DataType.FLOAT,
        "Sample standard deviation; null for a single rating.",
    ),
    (
        "entropy_bits",
        mlc.DataType.FLOAT,
        (
            "Shannon entropy of the distribution, 0 (unanimous) to 3.322 "
            "(evenly split). A direct measure of how contested the label is."
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="Path to data/mebeauty_v3")
    parser.add_argument("--output", required=True, help="Output croissant.json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    referenced_files = [
        "images/metadata.parquet",
        "landmarks.parquet",
        "ratings/aggregate/train.parquet",
        "ratings/aggregate/val.parquet",
        "ratings/aggregate/test.parquet",
        "ratings/distributions.parquet",
        "ratings/by_rater/ratings_by_rater.parquet",
        "standardized_256/images/metadata.parquet",
        "standardized_256/landmarks.parquet",
    ]
    for relative in referenced_files:
        if not (v3_dir / relative).is_file():
            raise FileNotFoundError(
                f"croissant.json would reference a file that doesn't exist: {relative}"
            )
    print(
        f"Confirmed all {len(referenced_files)} referenced files exist under {v3_dir}"
    )

    images_fileset = mlc.FileSet(
        id="images",
        name="images",
        description="Face images, flat and content-addressed (SHA-256 filename).",
        contained_in=[],
        encoding_formats=["image/jpeg", "image/png"],
        includes=["images/*.jpg", "images/*.png"],
    )
    metadata_file = mlc.FileObject(
        id="metadata-parquet",
        name="metadata.parquet",
        description="Per-image metadata: labels, provenance, duplicate/collision flags.",
        content_url="images/metadata.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )
    landmarks_file = mlc.FileObject(
        id="landmarks-parquet",
        name="landmarks.parquet",
        description="68-point facial landmarks per image_id (native list<float>, 136 values).",
        content_url="landmarks.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )
    ratings_files = [
        mlc.FileObject(
            id=f"ratings-{split}-parquet",
            name=f"ratings/{split}.parquet",
            description=f"Canonical {split} split: image_id, attractiveness score (1-10).",
            content_url=f"ratings/aggregate/{split}.parquet",
            encoding_formats=["application/vnd.apache.parquet"],
            sha256="unset-local-file",
        )
        for split in ["train", "val", "test"]
    ]
    distributions_file = mlc.FileObject(
        id="ratings-distributions-parquet",
        name="ratings/distributions.parquet",
        description=(
            "Per-image rating distribution (soft label) over the 1-10 scale, "
            "one row per (image_id, rating_type). Unfiltered: every rater "
            "contributes. Supports label distribution learning."
        ),
        content_url="ratings/distributions.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )
    by_rater_file = mlc.FileObject(
        id="ratings-by-rater-parquet",
        name="ratings/by_rater/ratings_by_rater.parquet",
        description=(
            "Individual pseudonymized rater scores. Carries `rating_type`: the "
            "collection ran two distinct tasks, `generic` attractiveness "
            "(mean 6.00) and `date` attractiveness (mean 5.01). The canonical "
            "aggregate corresponds to `generic` -- see DATASET_AUDIT.md Finding 16."
        ),
        content_url="ratings/by_rater/ratings_by_rater.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )

    # The second image configuration (docs/DATASET_AUDIT.md, Finding 17): the
    # native files are heterogeneous face crops, so a uniform 256x256 variant
    # ships alongside rather than replacing them. Described here so a Croissant
    # consumer sees both, not just the default.
    standardized_fileset = mlc.FileSet(
        id="standardized-256-images",
        name="standardized_256_images",
        description="Uniform 256x256 RGB variant: aspect-preserving Lanczos, centre-padded.",
        contained_in=[],
        encoding_formats=["image/jpeg"],
        includes=["standardized_256/images/*.jpg"],
    )
    standardized_metadata_file = mlc.FileObject(
        id="standardized-256-metadata-parquet",
        name="standardized_256/metadata.parquet",
        description=(
            "Per-image transform record for standardized_256: resize_scale, "
            "pad_left/top/right/bottom, original and output dimensions, "
            "preprocessing_version, and both native and standardized checksums."
        ),
        content_url="standardized_256/images/metadata.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )
    standardized_landmarks_file = mlc.FileObject(
        id="standardized-256-landmarks-parquet",
        name="standardized_256/landmarks.parquet",
        description=(
            "68-point landmarks in standardized_256 pixel space "
            "(native coordinates * resize_scale + padding offset)."
        ),
        content_url="standardized_256/landmarks.parquet",
        encoding_formats=["application/vnd.apache.parquet"],
        sha256="unset-local-file",
    )

    images_record_set = mlc.RecordSet(
        id="images-metadata",
        name="images",
        description="One record per unique image.",
        fields=[
            mlc.Field(
                id="images-metadata/image_id",
                name="image_id",
                description="SHA-256 of the image file; also its filename stem.",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="metadata-parquet",
                    extract=mlc.Extract(column="image_id"),
                ),
            ),
            mlc.Field(
                id="images-metadata/image",
                name="image",
                description="The image content.",
                data_types=[mlc.DataType.IMAGE_OBJECT],
                source=mlc.Source(
                    file_set="images",
                    extract=mlc.Extract(file_property=mlc.FileProperty.content),
                ),
                # Declares the join between the "images" FileSet and
                # metadata-parquet's rows: the file's own name must equal
                # metadata.parquet's `file_name` column. Without this,
                # mlcroissant rejects the RecordSet as an undeclared join
                # between two sources (confirmed: it does, this isn't
                # defensive over-specification).
                references=mlc.Source(
                    file_object="metadata-parquet",
                    extract=mlc.Extract(column="file_name"),
                ),
            ),
            mlc.Field(
                id="images-metadata/gender",
                name="gender",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="metadata-parquet", extract=mlc.Extract(column="gender")
                ),
            ),
            mlc.Field(
                id="images-metadata/ethnicity",
                name="ethnicity",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="metadata-parquet",
                    extract=mlc.Extract(column="ethnicity"),
                ),
            ),
            mlc.Field(
                id="images-metadata/has_label_collision",
                name="has_label_collision",
                description="True if this image's original label placement was ambiguous (see Finding 8).",
                data_types=[mlc.DataType.BOOL],
                source=mlc.Source(
                    file_object="metadata-parquet",
                    extract=mlc.Extract(column="has_label_collision"),
                ),
            ),
        ],
    )
    ratings_record_sets = [
        mlc.RecordSet(
            id=f"ratings-{split}",
            name=f"ratings_{split}",
            description=f"Canonical {split} ratings.",
            fields=[
                mlc.Field(
                    id=f"ratings-{split}/image_id",
                    name="image_id",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object=f"ratings-{split}-parquet",
                        extract=mlc.Extract(column="image_id"),
                    ),
                ),
                mlc.Field(
                    id=f"ratings-{split}/score",
                    name="score",
                    description=(
                        "Canonical attractiveness label, inherited from the legacy "
                        "release. NOT recomputable from ratings_by_rater -- it is the "
                        "output of a 2021 rater-cleaning pipeline whose intermediate "
                        "inputs are lost (Finding 20). Use this, not score_mean."
                    ),
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object=f"ratings-{split}-parquet",
                        extract=mlc.Extract(column="score"),
                    ),
                ),
                *[
                    mlc.Field(
                        id=f"ratings-{split}/{column}",
                        name=column,
                        description=description,
                        data_types=[data_type],
                        source=mlc.Source(
                            file_object=f"ratings-{split}-parquet",
                            extract=mlc.Extract(column=column),
                        ),
                    )
                    for column, data_type, description in LABEL_PROVENANCE_FIELDS
                ],
            ],
        )
        for split in ["train", "val", "test"]
    ]

    distributions_record_set = mlc.RecordSet(
        id="ratings-distributions",
        name="ratings_distributions",
        description=(
            "Per-image rating distribution (soft label), one row per "
            "(image_id, rating_type). Built from every rater with none "
            "excluded, unlike the canonical score."
        ),
        fields=[
            mlc.Field(
                id=f"ratings-distributions/{column}",
                name=column,
                description=description,
                data_types=[data_type],
                source=mlc.Source(
                    file_object="ratings-distributions-parquet",
                    extract=mlc.Extract(column=column),
                ),
            )
            for column, data_type, description in DISTRIBUTION_FIELDS
        ],
    )

    by_rater_record_set = mlc.RecordSet(
        id="ratings-by-rater",
        name="ratings_by_rater",
        description="One row per (image, rater, rating task).",
        fields=[
            mlc.Field(
                id="ratings-by-rater/image_id",
                name="image_id",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="ratings-by-rater-parquet",
                    extract=mlc.Extract(column="image_id"),
                ),
            ),
            mlc.Field(
                id="ratings-by-rater/rater_id",
                name="rater_id",
                description="Stable pseudonym; raw MTurk Worker IDs are never released.",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="ratings-by-rater-parquet",
                    extract=mlc.Extract(column="rater_id"),
                ),
            ),
            mlc.Field(
                id="ratings-by-rater/rating_type",
                name="rating_type",
                description=(
                    "`generic` or `date` -- two different questions, about a "
                    "point apart in mean. Filter to `generic` for the per-rater "
                    "equivalent of the canonical aggregate."
                ),
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="ratings-by-rater-parquet",
                    extract=mlc.Extract(column="rating_type"),
                ),
            ),
            mlc.Field(
                id="ratings-by-rater/score",
                name="score",
                data_types=[mlc.DataType.FLOAT],
                source=mlc.Source(
                    file_object="ratings-by-rater-parquet",
                    extract=mlc.Extract(column="score"),
                ),
            ),
        ],
    )

    standardized_record_set = mlc.RecordSet(
        id="standardized-256",
        name="standardized_256",
        description="One record per image in the uniform 256x256 configuration.",
        fields=[
            mlc.Field(
                id="standardized-256/image_id",
                name="image_id",
                description="Same id as the native configuration -- the two join on this.",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="standardized-256-metadata-parquet",
                    extract=mlc.Extract(column="image_id"),
                ),
            ),
            mlc.Field(
                id="standardized-256/image",
                name="image",
                description="The 256x256 RGB image content.",
                data_types=[mlc.DataType.IMAGE_OBJECT],
                source=mlc.Source(
                    file_set="standardized-256-images",
                    extract=mlc.Extract(file_property=mlc.FileProperty.content),
                ),
                references=mlc.Source(
                    file_object="standardized-256-metadata-parquet",
                    extract=mlc.Extract(column="file_name"),
                ),
            ),
            mlc.Field(
                id="standardized-256/resize_scale",
                name="resize_scale",
                description="Native-to-standardized scale factor; landmarks move by this.",
                data_types=[mlc.DataType.FLOAT],
                source=mlc.Source(
                    file_object="standardized-256-metadata-parquet",
                    extract=mlc.Extract(column="resize_scale"),
                ),
            ),
            mlc.Field(
                id="standardized-256/preprocessing_version",
                name="preprocessing_version",
                description="Pins target size, resampler, JPEG settings and Pillow version.",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="standardized-256-metadata-parquet",
                    extract=mlc.Extract(column="preprocessing_version"),
                ),
            ),
        ],
    )

    metadata = mlc.Metadata(
        name="MEBeauty",
        description=(
            "Multi-ethnic facial beauty dataset with attractiveness ratings. "
            "DRAFT/CANDIDATE: license and redistribution rights not yet resolved -- "
            "see docs/DATASET_AUDIT.md. This Croissant file describes the local "
            "data/mebeauty_v3/ structure and has not been published anywhere."
        ),
        cite_as=(
            "@article{lebedeva2022mebeauty, title={MEBeauty: a multi-ethnic facial beauty "
            "dataset in-the-wild}, author={Lebedeva, Irina and Guo, Yi and Ying, Fangli}, "
            "journal={Neural Computing and Applications}, volume={34}, number={17}, "
            "pages={14169--14183}, year={2022}, doi={10.1007/s00521-021-06535-0}}"
        ),
        url="https://github.com/dr-irina-lebedeva/MEBeauty-Benchmark",
        # No date_published: genuinely not published anywhere, so omitting is
        # accurate; inventing a date would not be. `version` reflects that
        # this is the first draft structure, not a claim about a real release.
        version="0.1.0-draft",
        # Deliberately NOT a real license (e.g. not "CC0" or any real SPDX/URL) --
        # this project's standing rule is to never invent a license (docs/DATASET_AUDIT.md).
        # An obviously-fake placeholder string only for local schema validation.
        license=["LICENSE-NOT-YET-DECIDED-SEE-DATASET-AUDIT-MD"],
        distribution=[
            images_fileset,
            metadata_file,
            landmarks_file,
            *ratings_files,
            distributions_file,
            by_rater_file,
            standardized_fileset,
            standardized_metadata_file,
            standardized_landmarks_file,
        ],
        record_sets=[
            images_record_set,
            *ratings_record_sets,
            distributions_record_set,
            by_rater_record_set,
            standardized_record_set,
        ],
    )

    # Round-trip through mlcroissant's own parser: this is the validation step.
    # ReadFromCroissant raises/reports issues if the generated JSON-LD is malformed.
    jsonld = metadata.to_json()
    issues = mlc.Dataset(jsonld=jsonld).metadata.issues
    print(f"Validation errors: {len(issues.errors)}")
    for error in issues.errors:
        print(f"  ERROR: {error}")
    print(f"Validation warnings: {len(issues.warnings)}")
    for warning in issues.warnings:
        print(f"  WARNING: {warning}")

    output_path.write_text(json.dumps(jsonld, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    print(
        "\nNOTE: the 'license' field above is the literal string "
        "'LICENSE-NOT-YET-DECIDED-SEE-DATASET-AUDIT-MD' -- not a real license, deliberately "
        "not CC0/MIT/any real value, so it can never be mistaken for an actual decision. "
        "Replace it once docs/DATASET_AUDIT.md's licensing question is resolved."
    )


if __name__ == "__main__":
    main()
