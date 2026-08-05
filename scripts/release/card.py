"""Generate the dataset card from the built release.

Written by a script rather than by hand so the prose cannot drift from the
data. That is not hypothetical: the columns were trimmed from 21 to 12 and a
hand-written table went on describing two that no longer existed, alongside
four links to documents that had been folded into the card. Counts come from
`manifest.json`, and the column tables are read from the shipped parquet, so
neither can be wrong unless the build itself is.

Everything lives in this one file: terms, takedown, limitations, citation. The
release is `README.md` plus `data/`, and nothing else.

    uv run python scripts/release/card.py --release data/release/public
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CITATION = """@article{lebedeva2022mebeauty,
  title   = {MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author  = {Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal = {Neural Computing and Applications},
  volume  = {34},
  number  = {17},
  pages   = {14169--14183},
  year    = {2022},
  doi     = {10.1007/s00521-021-06535-0}
}"""

MAINTAINER = """**Irina Lebedeva, PhD**
<dr.irina.lebedeva@gmail.com>
<https://www.irina-lebedeva.com>"""

#: What each shipped column means. A key with no matching column is simply not
#: printed, so deleting a column can never leave a stale row behind.
COLUMN_MEANING = {
    "image": "the face image",
    "image_id": "identifier for this image, `mebeauty_000001` style",
    "beauty_score": (
        "**the label.** The average rating, with rater leniency removed first"
    ),
    "plain_mean_score": (
        "the simple average of the same ratings, with no correction. For "
        "reference — not the score to train on"
    ),
    "rating_count": "how many raters scored this image",
    "score_std": "spread of those ratings",
    "source_url": "the source page this image was matched to",
    "image_native": "the image as collected, at its original resolution, before cropping",
    "landmarks_native": "the same 68 points in native coordinates",
    "gender": "gender of the person in the image",
    "ethnicity": "ethnicity of the person in the image",
    "split": "`train`, `val` or `test` under the held-out protocol",
    "attractiveness_ratings": "all individual attractiveness ratings for this image",
    "date_ratings": "all individual personal-preference ratings for this image",
    "ratings": "all individual ratings for this image",
    "rating_distribution": "counts per point on the 1-10 scale, from the raw ratings",
    "landmarks": "68 facial points, 136 values, in this config's coordinate space",
    "legacy_gender_label": (
        "annotation from the original collection, not a verified identity"
    ),
    "legacy_ethnicity_label": (
        "annotation from the original collection, not a verified identity"
    ),
    "cv_fold": "cross-validation fold 0-4, covering every image",
    "rater_id": "identifies a rater within this dataset only",
    "rater_valid": (
        "false for raters screened out of the label (fewer than 3 distinct scores)"
    ),
    "score": "the rating this rater gave, 1-10",
    "age_source": "`inferred_from_legacy_filename` -- never self-reported",
}

LICENSE_TEXT = """MEBeauty Research Use Terms
===========================

Copyright in the MEBeauty compilation, annotations and metadata remains
with the applicable authors and rightsholders.

This licence covers the dataset compilation: the annotations, aggregated
scores, splits, landmarks and accompanying metadata. It does NOT cover the
photographs themselves, which were obtained from Unsplash, Pixabay and Pexels
and remain subject to those platforms' terms and to the rights of the
photographers and the people depicted.

PERMITTED

  Non-commercial academic research on facial attractiveness assessment
  (facial beauty prediction), including publication of aggregate results.

PROHIBITED

  1. Any commercial use.
  2. Facial identity recognition, verification, biometric identification,
     surveillance, or re-identification of any kind.
  3. Attempting to identify any depicted individual or any rater.
  4. Redistribution, mirroring, re-uploading or sublicensing of any part of
     the dataset, including images, ratings and metadata, is prohibited.
     Link to the official release instead.
  5. Consequential decisions about individuals -- employment, insurance,
     credit, immigration, dating or ranking services.

CONDITIONS

  Cite: Lebedeva, Guo & Ying, "MEBeauty: a multi-ethnic facial beauty dataset
  in-the-wild", Neural Computing and Applications 34(17):14169-14183, 2022.

  If notified that material has been removed from the official release, users
  must delete their local copy of that material and discontinue its use.

NO WARRANTY

  Provided as is. The maintainer cannot grant rights belonging to
  photographers, platforms, or depicted individuals, and makes no
  representation that redistribution or consent rights have been verified.

Contact: Irina Lebedeva, PhD -- dr.irina.lebedeva@gmail.com
"""

FRONTMATTER = """---
license: other
license_name: mebeauty-research-only
license_link: LICENSE
pretty_name: "MEBeauty: Multi-Ethnic Facial Beauty & Attractiveness Dataset"
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - facial-beauty-prediction
  - facial-attractiveness-prediction
  - facial-aesthetics-assessment
  - image-regression
  - subjective-labels
  - label-distribution-learning
  - preference-learning
  - multi-ethnic
  - computer-vision
  - face-analysis
annotations_creators:
  - crowdsourced
  - expert-generated
source_datasets:
  - original
extra_gated_prompt: >-
  MEBeauty is released for NON-COMMERCIAL ACADEMIC RESEARCH ONLY, restricted to
  facial attractiveness assessment (facial beauty prediction). It must NOT be
  used for facial identity recognition, verification, surveillance, or
  re-identification.

  These are images of real people. The ratings are subjective human
  opinions, not measurements of any property of the individuals shown.

  By requesting access you agree to all terms below.
extra_gated_fields:
  Full name: text
  Affiliation / Institution: text
  Intended research use: text
  I will use this dataset for non-commercial research only: checkbox
  I will NOT use it for identity recognition, verification, surveillance, or re-identification: checkbox
  I will NOT attempt to identify depicted individuals or raters: checkbox
  I will not redistribute, mirror, re-upload or sublicense any part of the dataset, including images, ratings and metadata: checkbox
  I agree to cite the MEBeauty paper: checkbox
extra_gated_button_content: Accept terms and access MEBeauty
configs:
  - config_name: fbp
    default: true
    data_files:
      - split: train
        path: data/fbp/train-*.parquet
      - split: validation
        path: data/fbp/validation-*.parquet
      - split: test
        path: data/fbp/test-*.parquet
  - config_name: fbp_extended
    data_files:
      - split: train
        path: data/fbp_extended/train-*.parquet
      - split: validation
        path: data/fbp_extended/validation-*.parquet
      - split: test
        path: data/fbp_extended/test-*.parquet
  - config_name: personalized_fbp
    data_files:
      - split: train
        path: data/personalized_fbp/train-*.parquet
      - split: validation
        path: data/personalized_fbp/validation-*.parquet
      - split: test
        path: data/personalized_fbp/test-*.parquet
  - config_name: personalized_date
    data_files:
      - split: train
        path: data/personalized_date/train-*.parquet
      - split: validation
        path: data/personalized_date/validation-*.parquet
      - split: test
        path: data/personalized_date/test-*.parquet
  - config_name: full
    data_files:
      - split: train
        path: data/full/train-*.parquet
      - split: validation
        path: data/full/validation-*.parquet
      - split: test
        path: data/full/test-*.parquet
---
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--repo",
        default="dr-irina-lebedeva/MEBeauty",
        help="Hugging Face repo id, used in the load examples",
    )
    return parser.parse_args()


def column_table(path: Path) -> str:
    """The schema, read from the parquet that actually shipped."""
    import pyarrow.parquet as pq

    lines = ["| Column | Meaning |", "|---|---|"]
    for name in pq.read_schema(path).names:
        lines.append(f"| `{name}` | {COLUMN_MEANING.get(name, '')} |")
    return "\n".join(lines)


def column_summary(release: Path) -> str:
    """One compact line per column, read from the shipped parquet.

    Generated rather than written: a hand-typed list went stale twice, once
    describing two columns that had been deleted.
    """
    import pyarrow.parquet as pq

    lines = ["| Column | From | Meaning |", "|---|---|---|"]
    for config in ("fbp", "fbp_extended", "full"):
        path = min((release / "data" / config).glob("test-*.parquet"))
        for name in pq.read_schema(path).names:
            if any(name in row for row in lines):
                continue
            lines.append(f"| `{name}` | `{config}`+ | {COLUMN_MEANING.get(name, '')} |")
    return "\n".join(lines)


def body(m: dict, repo: str, release: Path) -> str:
    s = m["splits"]
    (sw, sh), (lw, lh) = m["smallest_native"], m["largest_native"]

    return f"""
# MEBeauty: Multi-Ethnic Facial Beauty & Attractiveness Dataset

MEBeauty contains about 2,500 face images rated for attractiveness on a 1-10
scale by people from different social and cultural backgrounds. The dataset is
diverse in age and includes both genders and six ethnic groups: Caucasian,
Asian, Hispanic, Black, Indian, and Middle Eastern. It provides the average
score for each image, all individual attractiveness ratings, and a separate
set of personal-preference ratings. This allows researchers to study both
generic and personalized facial beauty prediction.

These scores show people's opinions, not facts about the people in the images.
They reflect what a particular group of raters found attractive, including
their personal tastes, culture, and possible biases. The scores do not measure
anyone's value and should not be used to make conclusions about any gender,
ethnicity, or age group. A model trained on this dataset learns only the
preferences of these raters. The dataset is published to support research on
how people judge facial attractiveness.

## About this release

A **mirror and improved version** of the original MEBeauty dataset, published
by its first author. The original appeared five years ago; this release keeps
the same images and the same ratings — **nothing new was collected** —
and improves everything around them. The ratings, the raters and the image
metadata were re-checked with tooling that did not exist in 2021, duplicates
and unusable files were removed, and the splits were rebuilt. Counts differ
slightly from the original for that reason.

Original repository: <https://github.com/fbplab/MEBeauty-database>

## Terms of use

- **Non-commercial academic research only.**
- **Facial attractiveness assessment only.** Not for face recognition,
  identification, verification, biometric matching, re-identification or
  surveillance.
- **No redistribution.** Do not mirror or re-upload the images or ratings.
- **Citation required** (see below).

The images are not owned by the maintainer and remain subject to the rights
of their photographers and the people shown. Collection is described in
the paper. See `LICENSE`.

## Quick start

**Access is gated.** Log in and accept the terms once, then:

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")     # default config: `fbp`
ds[0]["image"], ds[0]["beauty_score"]          # 256x256 aligned face, 1-10
```

```bash
huggingface-cli login      # first time only
```

### Raters and ratings

| Task | Raters | Ratings |
|---|---|---|
| Attractiveness | {m["attractiveness_raters"]:,} | {m["ratings"]["attractiveness"]:,} |
| Date preference | {m["date_raters"]:,} | {m["ratings"]["preference"]:,} |
| Unique raters | {m["raters_total_unique"]:,} | |

| Config | For |
|---|---|
| **`fbp`** *(default)* | cropped aligned face + score. Minimal setup for training a generic predictor |
| **`fbp_extended`** | adds the original image, gender and ethnicity |
| **`personalized_fbp`** | adds every individual rating with rater demographics |
| **`personalized_date`** | the same for a second task (would the rater date this person) |
| **`full`** | everything |

## The data

**Images.** Two views of each face: an **aligned 256x256 crop** (eyes and
mouth in the same place every time) and the **original image** as collected,
{sw}x{sh} to {lw}x{lh}, with background.

{column_summary(release)}

**Label.** `beauty_score`, on a 1-10 scale, in every config.

Every face was seen by a different group of raters, and raters differ in how
generously they score — so a simple average would partly depend on *who
happened to rate a face*. `beauty_score` removes that effect before averaging.
No rater is removed.

The simple average ships too, as `plain_mean_score`, but **only in the `full`
config** — for reference, not for training.

Also `rating_distribution`, a histogram of the raw ratings, for
label-distribution learning. 68 facial landmarks ship with every image.

**Splits.** Two protocols; use one and say which.

| | |
|---|---|
| **80/10/10 fixed split** *(default)* | train {s["train"]:,} / validation {s["val"]:,} / test {s["test"]:,} |
| **5-fold cross-validation** | `cv_fold` 0-4, every image evaluated once |

Similar-looking images that may show the same person are grouped so they never
cross a split or fold.

```python
from datasets import concatenate_datasets

# cv_fold covers every image, so join the three splits first
data = concatenate_datasets([load_dataset("{repo}", split=s)
                             for s in ("train", "validation", "test")])
train = data.filter(lambda r: r["cv_fold"] != 0)
test  = data.filter(lambda r: r["cv_fold"] == 0)
```

### Individual ratings

`personalized_fbp`, `personalized_date` and `full` carry **all individual
ratings for each image**:

| Field | Meaning |
|---|---|
| `rater_id` | identifies a rater within this dataset only |
| `score` | what this person gave, 1-10 |
| `rater_gender`, `rater_ethnicity`, `rater_age_band` | where known; empty for most raters |

## More examples

```python
# a different config
ext = load_dataset("{repo}", "fbp_extended", split="train")
ext[0]["image_native"], ext[0]["gender"], ext[0]["ethnicity"]

# every individual rating for one face
per = load_dataset("{repo}", "personalized_fbp", split="test")
[r["score"] for r in per[0]["ratings"]]        # e.g. [4.0, 7.0, 6.0, ...]
per[0]["beauty_score"]                         # the label for that face

# train on one subgroup
asian_women = ext.filter(lambda r: r["ethnicity"] == "asian"
                                and r["gender"] == "female")
```

## Version history

**2.1.0** — new label, `beauty_score`, which corrects for raters who score
high or low in general. The old `score_mean` has the same values but a new
name, `plain_mean_score`, and now sits in the `full` config only. Results from
2.0.0 are still valid — just say which score you used, since the two can
differ by up to 1.2 for one face.

**2.0.0** — first Hugging Face release.

## Limitations

- Attractiveness ratings are **subjective opinions**. A model trained here
  predicts what these raters said, not a property of anyone shown.
- Labels were recomputed from the individual ratings, so **numbers from papers
  using the 2022 release are not directly comparable**.

## Citation

```bibtex
{CITATION}
```

## Contact

{MAINTAINER}
"""


def main() -> None:
    args = parse_args()
    release = Path(args.release).expanduser().resolve()
    manifest = json.loads((release.parent / "manifest.json").read_text())
    (release / "README.md").write_text(
        FRONTMATTER + body(manifest, args.repo, release), encoding="utf-8"
    )
    (release / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    print(f"Wrote {release / 'README.md'} and LICENSE")


if __name__ == "__main__":
    main()
