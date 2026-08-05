# Known problems — `data/mebeauty_v3/`, 2026-07-28

Findings from a fresh audit of the built dataset, not of the documentation.
Every number here was measured against the files on disk on this date; none is
carried over from an earlier write-up.

State at time of audit: **2,467 images**, 2,460 labelled (train 1,719 / val
218 / test 523), 68,944 ratings from 593 raters, `make check` 96/96 passing.

> **Update, later on 2026-07-28.** The labelling policy changed after this
> audit (see "Labelling policy change" at the end). The dataset is now
> **2,467 images, 2,018 labelled** (train 1,399 / val 185 / test 434) from
> 64,624 ratings by 522 raters. **D1 and D4 are fixed.** Every other finding
> is unaffected: they concern the images, not the labels.

---

## Blocking release

### B1. Unsplash's terms prohibit this use — 1,321 images (54%)

Unsplash Terms & Conditions, Section 8 (Prohibited Conduct):

> "Use the Images in connection with any machine learning and/or artificial
> intelligence datasets (e.g., training any machine learning and/or artificial
> intelligence models), or for technologies designed or intended for the
> identification of natural persons."

MEBeauty is a machine learning dataset containing those images. Confirmed
verbatim from three independent sources.

Two mitigating points, neither resolved:

- The *second* clause (identification of natural persons) does **not** apply —
  attractiveness prediction is not identification.
- MEBeauty was collected in 2020–21 and **the date this clause was added is
  unverified** (archive.org was unreachable from the audit environment). If it
  postdates collection, its application to images already downloaded is a
  genuine question.

Unsplash operates <https://unsplash.com/data> for licensed ML use. Asking
directly is the cheapest route to an answer.

### B2. No platform licence grants rights over the people depicted

All three sources explicitly exclude this: Pixabay notes third-party
**privacy rights** may require separate consent; Unsplash excludes
**recognisable people**; Pexels requires identifiable people not appear "in a
bad light".

A stock licence covers the photographer's copyright, never the subject's
consent. For a dataset where every face carries a public attractiveness
score, **this affects all 2,467 images, not only the Unsplash ones**, and no
platform licence has ever addressed it.

### B3. Documentation does not describe the current dataset

`docs/DATASET_CARD.md` still describes the pre-2026-07-28 state: old split
sizes, no source links, no rater screening, and none of the columns added
since. A reader would form a materially wrong picture of what ships.

---

## Data defects

### D1. `cropped_256` metadata is stale — 2 phantom rows

`cropped_256/images/metadata.parquet` has **2,469 rows against 2,467 crop
files**. Two rows reference `image_id`s that no longer exist, left from an
intermediate build; the table was not regenerated after the image count was
corrected. Joining on it yields two rows with no file behind them.

**Fixed 2026-07-28.** The cause was not `build_face_crops.py` but the
PNG-to-JPEG conversion: it changed two `image_id`s, and the remap added the
new rows without retiring the old ones. Re-running the crop detector for two
images would have been the wrong fix. `scripts/data/refresh_coverage.py` now
reconciles every derived config against the images actually on disk, and
fails loudly if an image has no metadata row.

### D2. Two entries are the same photograph — found only via source links

`male/mideastern/118.jpg` (600×600) and
`male/mideastern/beard-1866984_1920.jpg` (400×400) both resolve to
`pixabay.com/photos/baard-guy-gelukkig-man-snor-1866984/`.

Two different crops of one source photograph. Byte hashing missed it (not
identical), perceptual hashing missed it (different framing), and background
correlation missed it (different crops). **Only the source URL reveals it** —
a check that became possible only when the documented links arrived.

Both copies are in `train`, so this is redundancy, not split leakage. But
**matching by source URL has never been run systematically**; this single case
surfaced incidentally.

### D3. Quality flags exist but are unreachable from the main metadata

Information that consumers need is computed and then left where they will not
look:

| Flag | Affected | Where it lives now |
|---|---|---|
| landmarks outside the image frame | **101 images** | nowhere — computable, never stored |
| crop >20% invented pixels | **220 crops** | `cropped_256/` metadata only |
| crop from a multi-face image | **17 crops** | `cropped_256/` metadata only |
| face count per image | 25 multi-face, 4 no-face | `reports/` only, which does not ship |

`docs/DATASET_AUDIT.md` Finding 17 states `has_out_of_bounds_landmarks` flags
these in metadata. **It does not exist in metadata.** That claim is false as
written.

### D4. Label reliability varies about 15x

Ratings per image range from **7 to 103** (median 27); **443 images (18%) have
fewer than 10**. `ci95` and `n_ratings` ship, so this is visible — but a
benchmark whose test labels sometimes rest on seven opinions should say so
explicitly rather than leave it to be discovered.

**Fixed 2026-07-28** by the policy change below: an image now needs 10 ratings
from valid raters to carry a label at all. 449 images lost their label and
left the splits. The remaining range is 10–103 (median 29), so reliability
still varies about 10x — visible in `n_ratings` and `ci95`, and no longer
extending down to seven opinions.

### D5. Redundant provenance columns invite the wrong choice

`metadata.parquet` carries both the authoritative `source_url` (documented,
from the maintainer's own collection record) and the superseded
`inferred_source_url` / `inferred_photo_id` / `inferred_platform` (filename
guesses, ~1,050 of which were malformed). Nothing marks which to prefer, and
the wrong one is easy to pick.

### D6. Four links remain unverified

Four images carry `provenance_confidence = "inferred"` — the photo id comes
from the filename, but the page was never confirmed to open:

```
female/asian/hijab-5090230_1920.jpg
male/mideastern/pexels-emre-keshavarz-3518392.jpg
female/caucasian/woman-3718859_1920.jpg
male/hispanic/couple-5917009_1920.png
```

Correctly labelled, not hidden. Noted so the count is not mistaken for 2,467
verified links — **no link in this dataset has ever been machine-confirmed to
resolve**; all three platforms block automated checks.

---

## Open, not defects

### O1. Public-figure screening is incomplete by construction

Three public figures were found and removed. A perceptual screen ranked all
three inside the top 19%, and a visual pass covered ranks 1–672. **1,795
images have never been looked at.** Contact sheets covering all 2,467 exist at
`reports/legacy_audit/review_sheets/` (62 sheets, highest risk first) and the
review has not been started.

No claim that the dataset is free of public figures is supportable.

### O2. Subgroup imbalance, inherited

Caucasian: 537 images. Every other ethnicity: 140–150. Subgroup score means
span **5.30 (mideastern male) to 6.64 (mideastern female)**.

Inherited from the original collection, not introduced here, and every
ethnicity x gender cell has at least 24 test images — so subgroup evaluation
is possible. But it bounds what fairness claims the dataset can support.

### O3. Identity leakage is closed only above a validated threshold

All 46 cross-split identity leaks were closed and re-verified at **0**. The
threshold is face similarity 0.7, chosen because below it the model began
grouping different people who merely look alike. Pairs between 0.5 and 0.7
were deliberately left alone. The honest claim is "no identity leakage above
similarity 0.7", never "no identity leakage".

---

## Verified healthy

Checked this date and found clean:

- **No nulls and no empty strings** anywhere in `metadata.parquet`
- **Landmarks 100% present**, no NaN, no all-zero rows, correct 136-value shape
- **No orphans** — every file in all three image configurations has a metadata
  row, and every row has a file
- **All 6 excluded images genuinely absent**, verified by name
- **No duplicate `image_id` or `legacy_path`**
- **Filenames still equal the SHA-256 of their contents** (sampled)
- **`score_raw_mean` equals the shipped distribution mean to 0.00e+00**
  (`score` deliberately does not — it normalises each rater first)
- **Splits balanced** — score means 6.00 / 5.91 / 5.98 after the policy
  change, female share 0.53 / 0.51 / 0.52
- **All three image configurations agree at 2,467 files**
- **Croissant validates with 0 errors**; `load_dataset()` returns 2,467 for
  both configurations

---

## Labelling policy change — 2026-07-28

Applied after the audit above, at the maintainer's direction. Three rules
changed at once, and they interact, so they are recorded together.

### What changed

| | Before | After |
|---|---|---|
| Rater dropped for low volume | fewer than 10 ratings | **no volume rule** |
| Rater dropped for no variation | ≤2 distinct scores, and ≥10 ratings | **<3 distinct scores, at any volume** |
| Image needs to be labelled | 1 rating | **10 ratings from valid raters** |
| Averaging | plain mean | **per-rater normalisation, then mean** |

### What it cost and bought

- **106 raters returned** (407 ratings) that the volume rule had excluded.
- **64 raters removed** (341 ratings, 0.5%) for using fewer than 3 distinct
  scores — including light raters the old rule never examined.
- Net: **64,624 ratings from 522 raters** over **2,018 images**.
- **449 images lost their label** (18%) and left the splits: train
  1,719 → 1,399, val 218 → 185, test 523 → 434. They keep their pixels and
  metadata; only the label is gone.

### Corrections made on 2026-07-29

Three changes after the policy first shipped, all found by checking rather
than by review:

**1. The soft label and the point label disagreed.** `score` is a mean of
*normalised* ratings, but `distributions.parquet` counts the *raw* integer
scores, so its expectation is `score_raw_mean`. A distribution-learning method
trains on the distribution and is then scored against `score` — measured, that
cost the LDL entry **~0.09 MAE (≈16%)** for correctly hitting the target it
was given.

Fixed at the root: clipping now happens **per rating** rather than after
averaging, so `score` is exactly the mean of a mean-preserving soft-binned
distribution over normalised ratings. `distributions_normalised.parquet` ships
alongside, the benchmark selects the soft label matching the point label, and
the protocol **verifies expectation == label on every load**. That check
immediately caught a second bug: the two builders were normalising over
different rating sets (0.09 drift). Both now agree to 2×10⁻¹⁵.

**2. The shrinkage constant was never validated.** `k = 10` was chosen by
analogy to the 10-rating image threshold. Held-out rating prediction (5 folds,
`scripts/data/validate_shrinkage.py`) says **k = 5**, and the default changed.

But the honest headline is that **the choice barely matters**: everything from
k = 0 to k = 20 sits inside 0.1% of the best. It is not a tuned knob and
should not be presented as one. What the measurement *does* rule out is heavy
shrinkage — k ≥ 50 is clearly worse, because it drags every rater onto the
global scale and erases the differences normalisation exists to model.

The comparison also understates shrinkage's value, and does so honestly: to
score k = 0 at all, raters whose z is undefined (one rating, zero spread) must
be dropped — exactly the rows shrinkage exists to handle. Shrinkage is in the
pipeline because the policy keeps light raters, not because this curve proves
it.

**3. Labels shifted slightly.** With k = 5 and per-rating clipping,
normalisation now moves a label by 0.234 on average (was 0.243) and 1.195 at
most. Agreement with the independent affine model is unchanged at r = 0.976.

### Why normalisation needed shrinkage

Keeping light raters and z-scoring them are directly in tension: the returning
raters have a **median of 3 ratings and a minimum of 1**, and a rater with one
rating has no standard deviation to divide by. `legacy/normalization.py`
therefore shrinks each rater's mean and variance toward the global values in
proportion to their evidence, so a rater with 200 ratings keeps their own
statistics and a rater with 2 is barely adjusted rather than wildly adjusted.

### Effect on the labels

Normalisation moves a label by **0.243 on average and 1.133 at most**;
`score` correlates with the plain mean at **r = 0.973** (Spearman 0.973). One
image's normalised score fell below 1 (to 0.48) and was clipped to the scale
floor.

Both the unnormalised mean (`score_raw_mean`) and the fully unscreened mean
(`score_all_raters`) ship beside `score`, so every step is reversible.

### Independent confirmation

`score` standardises each rater on their own. `score_adjusted` fits per-rater
offsets and scales **jointly** by alternating least squares — a different
method for the same correction. They agree at **r = 0.976** (Spearman 0.975,
mean absolute difference 0.234). Two unrelated routes landing in the same
place is evidence the correction reflects the ratings rather than either
method's assumptions.

### Open caveat this raised

The assignment-confounding diagnostic moved from r = 0.090 (p = 0.11) to
**r = 0.120 (p = 0.033)** on the screened ratings. Normalisation assumes a
rater's mean reflects their generosity; a nominally significant correlation
between rater means and the leave-one-rater-out quality of their assigned
images means a small part of it instead reflects **which images they were
shown**. At r = 0.12 that is about 1.4% of the variance in rater means, so
`score` is very slightly over-corrected. It is small, it is now measured
every build, and it is not resolved.
