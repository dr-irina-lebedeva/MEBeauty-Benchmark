"""Build the five published configurations.

Each is a *view* of the same 2,462 images, shaped for one job. A user should be
able to pick a config by naming their task, not by reading a schema.

| Config | For |
|---|---|
| `fbp` | training a beauty predictor. Drop-in shape for SCUT-FBP5500 code. |
| `fbp_extended` | the same plus demographics and the native image, for fairness and subgroup work |
| `personalized_fbp` | modelling *individual* raters rather than the consensus |
| `personalized_date` | the same for the second task |
| `full` | everything, for anyone who would otherwise have to join configs |

**Ratings are nested, not long.** In the personalized configs each image row
carries a list of `{rater_id, score, rater_gender, rater_ethnicity,
rater_age_band}`. Long format -- one row per (image, rater) -- would repeat
the image bytes 28 times on average, turning a 58 MB config into several
gigabytes for no extra information.

**Rater demographics exist only for the 29-member in-house panel.** MTurk
supplied none, so those fields are null on roughly 99% of ratings. They are
carried anyway because the panel is exactly who a rater-effects study needs.

**`fbp` deliberately omits gender and ethnicity.** It is the minimal
benchmark surface: a model trained on it cannot condition on demographics
even accidentally. `fbp_extended` exists for when you want them.

**One label everywhere.** `beauty_score` removes rater leniency before
averaging and is the target in every config. The plain unweighted average
ships as `plain_mean_score` in `full` only -- it is a reference point, not a
benchmark target, and offering both in every config invited papers to report
against different labels without saying so. The two correlate 0.97 but differ
by up to 1.2 on individual images.

Nothing is lost by confining it: the plain mean is the expectation of
`rating_distribution`, which ships in every config.

    uv run python scripts/release/build_configs.py \\
        --v3 data/mebeauty_v3 --release data/release/public
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SPLIT_FILENAMES = {"train": "train", "val": "validation", "test": "test"}
SHARD_ROWS = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--release", required=True)
    return parser.parse_args()


def features_for(config: str):
    from datasets import Features, Image, Sequence, Value

    base = {
        "image": Image(),
        "image_id": Value("string"),
        "beauty_score": Value("float64"),
        "rating_distribution": Sequence(Value("int32")),
        "landmarks": Sequence(Value("float64")),
        # Both split protocols live in the minimal config, so `fbp` alone
        # supports either without stepping up to `fbp_extended`.
        "split": Value("string"),
        "cv_fold": Value("int32"),
    }
    if config == "fbp":
        return Features(base)

    extended = {
        **base,
        "image_native": Image(),
        "landmarks_native": Sequence(Value("float64")),
        "gender": Value("string"),
        "ethnicity": Value("string"),
    }
    if config == "fbp_extended":
        return Features(extended)

    # `[{...}]`, not `Sequence({...})`. The latter is HF's "struct of arrays"
    # layout and expects a dict of parallel lists; the list form gives the
    # natural list-of-records a caller expects from `row["ratings"]`.
    ratings = [
        {
            "rater_id": Value("string"),
            "score": Value("float32"),
            "rater_gender": Value("string"),
            "rater_ethnicity": Value("string"),
            "rater_age_band": Value("string"),
        }
    ]
    if config in ("personalized_fbp", "personalized_date"):
        return Features({**extended, "ratings": ratings})
    # `full` is the only config carrying the label-support and provenance
    # fields. They are reference material rather than training signal, and
    # repeating them across four configs duplicated bytes for no use.
    return Features(
        {
            **extended,
            "plain_mean_score": Value("float64"),
            "score_std": Value("float64"),
            "rating_count": Value("int32"),
            "source_url": Value("string"),
            "attractiveness_ratings": ratings,
            "date_ratings": ratings,
        }
    )


def nested_ratings(
    ratings: pd.DataFrame, raters: pd.DataFrame
) -> dict[str, list[dict]]:
    """image_id -> list of per-rater records, demographics joined on."""
    joined = ratings.merge(raters, on="rater_id", how="left")
    out: dict[str, list[dict]] = {}
    for image_id, frame in joined.groupby("image_id"):
        out[image_id] = [
            {
                "rater_id": row.rater_id,
                "score": float(row.score),
                "rater_gender": None if pd.isna(row.gender) else row.gender,
                "rater_ethnicity": None if pd.isna(row.ethnicity) else row.ethnicity,
                "rater_age_band": None if pd.isna(row.age_band) else str(row.age_band),
            }
            for row in frame.itertuples()
        ]
    return out


def build(
    config: str,
    images: pd.DataFrame,
    crop_dir: Path,
    native_dir: Path,
    out: Path,
    attractiveness: dict | None,
    date: dict | None,
) -> None:
    from datasets import Dataset

    out.mkdir(parents=True, exist_ok=True)
    features = features_for(config)
    wanted = set(features)
    total = 0

    for split, filename in SPLIT_FILENAMES.items():
        subset = images[images["split"] == split].reset_index(drop=True)
        shards = max(1, (len(subset) + SHARD_ROWS - 1) // SHARD_ROWS)
        for index in range(shards):
            chunk = subset.iloc[index * SHARD_ROWS : (index + 1) * SHARD_ROWS]
            rows = []
            for row in chunk.itertuples():
                record = {
                    "image": {
                        "bytes": (crop_dir / f"{row.sha256}.jpg").read_bytes(),
                        "path": f"{row.image_id}.jpg",
                    },
                    "image_id": row.image_id,
                    "beauty_score": float(row.beauty_score),
                    "rating_distribution": [int(v) for v in row.rating_distribution],
                    "landmarks": [float(v) for v in row.landmarks_cropped],
                    "split": row.split,
                    "cv_fold": int(row.cv_fold),
                }
                if "image_native" in wanted:
                    record |= {
                        "image_native": {
                            "bytes": (native_dir / f"{row.sha256}.jpg").read_bytes(),
                            "path": f"{row.image_id}_native.jpg",
                        },
                        "landmarks_native": [float(v) for v in row.landmarks_native],
                        "gender": row.legacy_gender_label,
                        "ethnicity": row.legacy_ethnicity_label,
                    }
                if "ratings" in wanted:
                    source = attractiveness if config == "personalized_fbp" else date
                    record["ratings"] = source.get(row.image_id, [])
                if "score_std" in wanted:
                    record |= {
                        "plain_mean_score": float(row.plain_mean_score),
                        "score_std": None
                        if pd.isna(row.score_std)
                        else float(row.score_std),
                        "rating_count": int(row.rating_count),
                        "source_url": row.source_url,
                    }
                if "attractiveness_ratings" in wanted:
                    record["attractiveness_ratings"] = attractiveness.get(
                        row.image_id, []
                    )
                    record["date_ratings"] = date.get(row.image_id, [])
                rows.append(record)

            dataset = Dataset.from_list(rows, features=features)
            target = out / f"{filename}-{index:05d}-of-{shards:05d}.parquet"
            dataset.to_parquet(target)
            total += target.stat().st_size
    print(f"  {config:20s} {total / 1e6:7.1f} MB  ({len(features)} columns)")


def main() -> None:
    args = parse_args()
    v3 = Path(args.v3).expanduser().resolve()
    release = Path(args.release).expanduser().resolve()
    images = pd.read_parquet(release.parent / "images.parquet")

    raters = pd.read_parquet(release.parent / "ratings" / "rater_demographics.parquet")
    attractiveness = nested_ratings(
        pd.concat(
            [
                pd.read_parquet(p)
                for p in sorted(
                    (release.parent / "ratings" / "attractiveness_ratings").glob(
                        "*.parquet"
                    )
                )
            ]
        ),
        raters,
    )
    date = nested_ratings(
        pd.concat(
            [
                pd.read_parquet(p)
                for p in sorted(
                    (release.parent / "ratings" / "preference_ratings").glob(
                        "*.parquet"
                    )
                )
            ]
        ),
        raters,
    )

    for config in (
        "fbp",
        "fbp_extended",
        "personalized_fbp",
        "personalized_date",
        "full",
    ):
        build(
            config,
            images,
            v3 / "cropped_256" / "images",
            v3 / "images",
            release / "data" / config,
            attractiveness,
            date,
        )


if __name__ == "__main__":
    main()
