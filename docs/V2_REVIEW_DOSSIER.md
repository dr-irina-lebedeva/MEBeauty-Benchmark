# MEBeauty v2 — review dossier

**For external review before publication.** Prepared 2026-08-02.
Maintainer: Irina Lebedeva, PhD · dr.irina.lebedeva@gmail.com

v1 = the dataset accompanying Lebedeva, Guo & Ying, *Neural Computing and
Applications* 34(17):14169–14183, 2022. v2 is an improved, privacy-audited
re-release of the same collection by its original first author. **No new
images were collected and no new ratings were gathered.**

---

## 1. What changed, v1 → v2

| | v1 (2022) | v2 |
|---|---|---|
| Distribution | GitHub + cloud zips | Hugging Face, gated, parquet |
| Loading | manual download, custom code | `load_dataset(...)`, one line |
| Ratings released | aggregate score only | **every individual rating** (141,399) |
| Rater information | none | pseudonymous ids + demographics for the panel |
| Second task | not released | released, renamed `personal_preference` |
| Provenance | none | **source URL for all 2,462 images, all verified** |
| Splits | generated without a seed, not reproducible | fixed, seeded, leakage-checked, + 5 CV folds |
| Duplicate photographs | present | 25 merged |
| Landmarks | separate `.pts` files | embedded arrays, native and 256×256 |
| Label definition | one number, method undocumented | 3 aggregates, each defined and reproducible |
| Terms | "research only", unenforced | gated acceptance, 6 checkboxes |

---

## 2. Problems found and fixed

These were discovered by auditing the v1 material. Each was verified, not
assumed.

**Label reproducibility.** v1's score could not be recomputed from any
released file — its cleaning pipeline's inputs no longer exist. v2's `score`
is recomputed from the individual ratings and is reproducible in one line.
*Consequence: v2 labels differ from v1. Published v1 numbers are not
comparable.*

**Identity leakage across splits.** 35 identity groups spanned splits in v1,
inflating any test score. Closed, and re-verified against the rebuilt splits
at two levels of evidence:

| Evidence | Groups/pairs present | Crossing a split |
|---|---|---|
| Confirmed same-person (background-correlation test) | 224 pairs | **0** |
| ArcFace matches at 0.5, unconfirmed (weaker) | 156 groups | **0** |

Groups are formed from confirmed same-photograph pairs, confirmed same-person
pairs, shared source URLs and perceptual near-duplicates. The second row is
the stricter test: even matches nobody confirmed do not cross.

**Duplicate photographs.** 25 pairs were the same photograph stored twice
under different filenames. Byte hashing missed them (different encodings);
perceptual hashing missed them (different crops). Found by aligning on
landmarks and correlating the background. Merged, ratings pooled.

**No provenance.** v1 shipped no source information. v2 has a verified source
URL for every image, so licence and takedown questions are answerable
per-image.

**Rater privacy.** v1 material contained raw MTurk Worker IDs. v2 contains
**zero** worker IDs, hashes, HIT ids, emails, usernames, IPs or local paths —
asserted by an automated validator on every build, not by inspection.

**Rating-count threshold was excluding minorities.** Requiring ≥10 ratings
dropped 443 images that were **27% Asian, 18% Indian, 17% Black** against
11/10/11% among those kept — it made a multi-ethnic dataset measurably less
so. Threshold set to 8 (the data has a clean gap: 435 images sit at exactly
9). Ethnic imbalance improved 4.21× → 3.35×. The excluded images' mean score
(5.973) is statistically indistinguishable from the rest (5.981).

**Two images had silently lost their entire rating history** (26 and 27
ratings) through an alternate-filename lookup bug. Recovered.

---

## 3. Current state

### Structure

```
README.md                                  card, terms, limitations (all inline)
data/
  standardized_256/  train|validation|test  RGB 256×256, aspect preserved, centre-padded
  native/            train|validation|test  legacy face crops, 400×400 – 5304×6630
  generic_ratings.parquet                   68,868 rows
  personal_preference_ratings.parquet       72,531 rows
  raters.parquet                               860 rows
```

12 files, 179 MB. Five Hugging Face configs. Nothing else ships; the
pseudonym mapping and rater quality profiles stay in a private tier.

### Statistics

| | |
|---|---|
| Images | **2,462** (train 1,962 / validation 250 / test 250, + 5 CV folds) |
| Ratings | **141,399** — 68,868 attractiveness, 72,531 personal-preference |
| Raters | **860** — 831 crowd, 29 in-house panel |
| Ratings per image | 8 / **28** / 92 (min / median / max) |
| Ratings per rater | 1 / **51** / 1,127 |
| Score | mean 5.98, sd 1.30, range 1.72–9.03 (scale 1–10) |
| Mean within-image disagreement | sd **2.06** |
| Mean within-rater spread | sd 1.88 |
| Ethnicity | caucasian 964, asian 342, hispanic 291, black 289, indian 289, mideastern 287 |
| Gender | female 1,303, male 1,159 |
| Source | Unsplash 1,388, Pixabay 764, Pexels 310 |

### Inter-rater reliability

Computed from the shipped per-rater ratings (generic task, valid raters:
68,527 ratings, 529 raters, 2,462 images).

| Statistic | Value | Reading |
|---|---|---|
| ICC(1), single rater | **0.19** | two random raters agree weakly |
| ICC(1,k), k ≈ 28 | **0.87** | the *published mean* is highly reliable |
| Split-half r | 0.70 | two disjoint halves of the raters |
| Spearman–Brown corrected | **0.82** | reliability of a full-length mean |

This is the expected shape for a subjective judgement: individuals disagree,
the aggregate is stable. It also sets the ceiling any model can reach — a
predictor cannot correlate with the mean better than the mean correlates with
itself.

### Labels

Three aggregates ship; none is derivable from the others.

| Field | Correction applied |
|---|---|
| `score_mean` | none — plain unweighted mean |
| `score` *(canonical)* | each rater standardised to a common scale first |
| `score_adjusted` | rater offsets fitted jointly rather than independently |

`score` and `score_adjusted` agree at **r = 0.97** — two independent methods
reaching the same correction. `rating_distribution` is a histogram counted
from the raw ratings, never reconstructed from a mean, so label-distribution
learning is supported directly.

### Verification

An automated validator runs **36 checks** before any upload: zero rater
identifiers, no rater under 18, `score_mean` reproducible from the shipped
ratings to 0.00e+00, the second task provably absent from every score, every
one of 4,924 images decoding, all 256×256 images exactly 256×256, no
duplicate group crossing a split, no dropped column documented in the card,
and all counts reconciling.

---

## 4. Known limitations, disclosed in the card

- **Ethnic distribution is uneven** — Caucasian ~39%, others 11–14%.
- **Demographic labels on images are legacy annotations**, not verified
  identities, not self-reported.
- **Panel rater ages are inferred from filenames** and unverified; the source
  workbooks contain no age field.
- **101 images have landmark points outside the frame**, inherited from the
  original detector on faces cut off at the border. Carried through
  faithfully rather than silently clipped.
- **Identity screening is not exhaustive** — filenames, provenance and
  limited manual review, with no automated face recognition.
- **Ratings are subjective** and reflect the raters who gave them.
- **v2 labels differ from v1**; v1 numbers are not comparable.

---

## 5. Open matters a reviewer should weigh

Listed because they are unresolved, not because they are hidden.

1. **Platform terms.** 1,388 images (56%) come from Unsplash, whose terms
   restrict use in machine-learning datasets. The maintainer's position is
   that this is a 5-year-old published dataset being re-released, not a new
   collection, so republication creates no new exposure. A reviewer may take
   a different view.

2. **Consent of depicted persons.** Platform licences cover the
   photographers' rights, not the subjects'. Nobody photographed consented to
   being rated for attractiveness. Inherited from v1 and unchanged.

3. **Panel re-identification.** 29 in-house raters with exact demographics;
   most gender × ethnicity × age cells hold a single person. No personal
   identifier ships, access is gated, and terms forbid re-identification —
   but the residual risk is real and was accepted deliberately to preserve
   rater-effects research value.

4. **1,790 of the 2,462 images were never visually screened for public
   figures.** A celebrity-classifier ranking covered every image, but human
   review covered only ranks 1–672 — the most likely candidates. The
   remainder are unreviewed.

5. **`score` is slightly over-corrected.** The correction assumes a rater's
   mean reflects their generosity; measured, part of it instead reflects
   which images they saw (r = 0.120, p = 0.033, ≈1.4% of variance).
   `score_mean` ships uncorrected for anyone who prefers it.

6. **Two panel ages are published as 18** where the source filename reads 17,
   at the maintainer's direction. No under-18 age appears in the release; the
   original filenames remain in the private archive.

---

## 6. Questions for reviewers

1. Is the Unsplash position defensible for a re-release, or should those
   1,388 images be withheld or replaced with a fetch-by-URL variant?
2. Is publishing exact panel demographics (n=29) acceptable behind gating, or
   should they be coarsened despite the loss of research value?
3. Should `personal_preference` ship at all in a first release?
4. Are three parallel score definitions helpful, or should one be canonical
   and the others moved to a supplement?
5. Is the ≥8 rating threshold the right trade between coverage and label
   reliability, given the ethnic-balance argument?
6. Does anything in §5 rise to a blocker rather than a disclosure?

---

## 7. Where improvement is still possible

- **No implementation of the label pipeline has been checked against an
  external reference.** Everything is internally consistent and verified, but
  a second pair of eyes on the normalisation would strengthen it.
- **Public-figure screening should be completed** across all images.
- **No baseline results ship with the dataset.** Deliberate — benchmark work
  is separate — but users have no reference point for what a reasonable score
  is on these splits.
- **A DOI** (Zenodo or the Hub's own) would make v2 citable distinctly
  from v1.
