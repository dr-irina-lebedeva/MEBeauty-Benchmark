# Datasheet for MEBeauty

Following Gebru et al., ["Datasheets for Datasets"](https://arxiv.org/abs/1803.09010).
Status: **draft, unreleased**. Answers reflect the state after the audit in
[`docs/DATASET_AUDIT.md`](DATASET_AUDIT.md); anything not verifiable from
that audit is marked "Unknown" rather than assumed.

## Motivation

**For what purpose was the dataset created?**
To study facial attractiveness/beauty prediction across ethnicities, in
unconstrained real-world conditions, and to support general and personalized
attractiveness prediction research (Lebedeva, Guo & Ying, 2022).

**Who created the dataset and on whose behalf?**
Irina Lebedeva, Yi Guo, and Fangli Ying, as part of the published paper. This
repository (a re-audit and re-packaging effort) is maintained by Irina
Lebedeva independently of any employer or institution.

**Who funded the creation of the dataset?**
Unknown — not stated in the paper or the legacy repository.

## Composition

**What do the instances represent?**
Photographs of human faces, each with one or more crowd-sourced
attractiveness ratings on a 1–10 scale, plus per-image ethnicity and gender
labels.

**How many instances are there?**
2,547 original images (see Finding 1). 2,511 rated instances in the
canonical `benchmark-v1` split after deduplication (Finding 6). Two other
legacy split generations exist with different (unreconciled) counts.

**Does the dataset contain all possible instances or a sample?**
A sample, collected from public stock-photo platforms (Unsplash, Pexels,
Pixabay, inferred — not confirmed) plus an unknown number from other
sources (117 images have no resolvable provenance signal at all — Finding
11).

**What data does each instance consist of?**
A JPEG or PNG image (3 of 2,547 are PNG), an ethnicity label, a gender
label, and one or more numeric attractiveness ratings from different raters.
Derived artifacts also exist for most images: an MTCNN face crop, an OpenCV
face crop, a FaceNet-512 embedding, and 68-point facial landmarks — coverage
is not 100% (Finding 3, Finding 4).

**Is there a label or target associated with each instance?**
Yes — the attractiveness score is the prediction target. Ethnicity and
gender folder placement are secondary labels (8 images have conflicting
placement — Finding 8).

**Is any information missing from individual instances?**
Yes. 145 images lack a usable MTCNN crop/embedding, 196 lack an OpenCV crop,
88 lack landmarks (all silent pipeline failures, no recorded reason — see
Finding 3/4). 117 images have no resolvable source-provenance signal.

**Are relationships between instances made explicit?**
Content-duplicate relationships now are: `images.parquet`'s
`content_duplicate_group` column groups the 49 pairs of byte-identical
images (Finding 7), which the original release did not surface at all.

**Are there recommended data splits?**
A canonical `benchmark-v1` split (train/val/test, 1,766/224/521) is provided
in `reports/legacy_audit/canonical_splits/`, derived from the legacy "2022"
split generation with leakage removed. It is **not confirmed** to match the
original paper's reported train/val/test protocol — see Finding 6 and the
Open items in `docs/DATASET_AUDIT.md`.

**Are there any errors, sources of noise, or redundancies?**
Yes, itemized in `docs/DATASET_AUDIT.md`: 49 duplicate-image groups (Finding
7), 8 cross-label conflicts (Finding 8), 1 orphaned derived artifact with no
source image (Finding 3), and previously-undetected train/test leakage now
fixed in the canonical split (Finding 6).

**Is the dataset self-contained, or does it link to external resources?**
Self-contained for images and ratings. `images.parquet`'s inferred
provenance columns point at external platforms (Unsplash/Pexels/Pixabay),
but these are unverified guesses from filenames, not confirmed links
(Finding 11).

**Does the dataset contain data that might be considered confidential?**
No confidential data by design, but see the biometric-data question below.

**Does the dataset contain data that might be offensive, insulting,
threatening, or anxiety-inducing?**
Not identified in this audit; not systematically checked either.

**Does the dataset relate to people?**
Yes — every image depicts an identifiable human face, and ratings were
produced by human raters.

**Does the dataset identify any subpopulations?**
Yes, by ethnicity (6 categories) and gender (2 categories), as folder
labels assigned by the original collectors. Assignment methodology
(self-reported vs. inferred) is unknown.

**Is it possible to identify individuals, either directly or indirectly,
from the dataset?**
The photographed individuals are potentially identifiable from the images
themselves (faces are not blurred or altered). Whether the face embeddings
and landmarks constitute special-category biometric data under GDPR depends
on processing and intended use and has not had a legal review (flagged
explicitly as open in `docs/DATASET_AUDIT.md`). Raters are pseudonymized
(Finding 9); the real Worker ID mapping exists only in a local, uncommitted
file.

**Does the dataset contain data that might be considered sensitive?**
Potentially yes — see the biometric-data point above. No determination has
been made.

## Collection process

**How was the data associated with each instance acquired?**
Unknown in full detail. Images appear to have been sourced from stock-photo
platforms (inferred from filenames, Finding 11) and possibly other
unrecorded sources for the 117 images with no resolvable provenance.
Ratings were collected via Amazon Mechanical Turk (Finding 9).

**What mechanisms or procedures were used to collect the data?**
Unknown beyond what's in the published paper; not reconstructable from the
legacy repository's code.

**If the dataset is a sample, what was the sampling strategy?**
Unknown.

**Who was involved in data collection, and how were they compensated?**
Raters were compensated MTurk workers (rate unknown). Image collection was
done by the original authors; further detail unknown.

**Over what timeframe was the data collected?**
Unknown precisely. The legacy repository's most recent 2022 split
generation and the paper's 2022 publication date bound it loosely.

**Were any ethical review processes conducted?**
Unknown — not documented in the legacy repository.

## Preprocessing / cleaning / labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes, extensively, and largely undocumented in the original release:
- Face detection/cropping via DeepFace (MTCNN and OpenCV backends),
  with silent failures on ~6–8% of images (Finding 3).
- Facial landmark extraction (68-point), also with silent failures on
  ~3.5% of images (Finding 4).
- A geometric feature vector derived from landmarks, saved via
  `str(numpy_array)` and corrupted (100% truncated) in the legacy release —
  regenerated from scratch for this audit (Finding 5).
- FaceNet-512 embeddings extracted from the MTCNN crops.
- Rater Worker IDs pseudonymized for this audit (Finding 9); not
  pseudonymized in the original legacy release.

**Was the "raw" data saved in addition to the preprocessed data?**
Yes — `original_images/` is the untouched source; all derived artifacts sit
alongside it. This audit's tooling (`scripts/data/`) never modifies the
legacy source, only a local snapshot copy of it.

**Is the software used to preprocess/clean/label the instances available?**
Yes — `scripts/data/*.py` in this repository, all reproducible against a
checksummed local snapshot (`docs/DATASET_AUDIT.md`, "Reproducing this
document").

## Uses

**Has the dataset been used for any tasks already?**
Yes — the original paper's facial-beauty-prediction benchmarks (CNN and
transfer-learning baselines; Finding 10).

**Is there a repository that links to papers or systems using the dataset?**
Not yet built. `README.md` tracks this project's own use.

**Is there anything about the composition or collection that might impact
future uses?**
Yes: the silent preprocessing failures (Findings 3–4), duplicate images
(Finding 7), previously-unfixed split leakage (Finding 6), and unverified
provenance/rights (Finding 11) should all be accounted for before using this
data for a new benchmark claim. Ratings reflect one specific, unverified
rater population and should not be treated as a demographically neutral
ground truth.

**Are there tasks for which the dataset should not be used?**
Not evaluated. Given faces are identifiable and biometric classification is
unresolved, uses involving identification, verification, or any
consequential decision-making about the depicted individuals should not
proceed without a legal review.

## Distribution

**Will the dataset be distributed to third parties?**
Not yet. No Hugging Face upload, gating, or public release has occurred —
see `docs/DATASET_AUDIT.md`'s Open items.

**How will the dataset be distributed?**
Planned: a gated Hugging Face dataset repository, pending the licensing and
rights review.

**Will the dataset be subject to copyright or other IP restrictions?**
Undetermined — this is precisely the open licensing question in
`docs/DATASET_AUDIT.md`.

**Are there IP-based or other restrictions on the data?**
Source images are inferred (not confirmed) to originate from
stock-photo platforms with their own licensing terms; these have not been
verified.

## Maintenance

**Who maintains the dataset?**
Irina Lebedeva (see `README.md`).

**How can the maintainer be contacted?**
See `README.md`'s Maintainer section.

**Will the dataset be updated?**
Yes, as this audit and modernization effort continues — see
`docs/PROJECT_STATUS.md` for current phase.

**Will older versions continue to be supported/hosted/maintained?**
Planned: the historical paper protocol and any superseded split generations
will be kept as separate, clearly labeled protocols rather than overwritten
— see `docs/DATASET_AUDIT.md`, Finding 6.

**Is there a mechanism for others to contribute?**
Via this repository's issue tracker/pull requests once public (see
`README.md`).
