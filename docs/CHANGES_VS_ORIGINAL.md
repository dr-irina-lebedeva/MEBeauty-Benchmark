# What changed vs. the original MEBeauty release

A complete, reproducible account of every difference between this
repository's dataset and the original MEBeauty release, for anyone who has
used the original and needs to know what moved.

**Original**: <https://github.com/fbplab/MEBeauty-database>, pinned at commit
`7b849562ee92d99d34d56afa2ebe85e5075f9b20`. The original repository is
treated as read-only; nothing here modifies it.

**Paper**: Lebedeva, Guo & Ying, *"MEBeauty: a multi-ethnic facial beauty
dataset in-the-wild"*, Neural Computing and Applications 34(17):14169–14183,
2022. [doi:10.1007/s00521-021-06535-0](https://doi.org/10.1007/s00521-021-06535-0)

Every claim below is reproducible from scripts in this repository; commands
are collected in [`DATASET_AUDIT.md`](DATASET_AUDIT.md#reproducing-this-document),
and the finding numbers referenced throughout point into that document.

> **Status**: this describes a local candidate build. Nothing has been
> published to Hugging Face or anywhere else, and licensing is unresolved —
> see [Still open](#still-open) before using any of it.

---

## Summary

| | Original | This repository |
|---|---|---|
| Images | 2,547 files | **2,495** unique (49 byte-identical duplicate groups collapsed; 3 removed, see below) |
| MTCNN crops | 2,403 | **2,547** (100%) |
| OpenCV crops | 2,351 | **2,539** (99.7%) |
| FaceNet-512 embeddings | 2,403 | **2,547** (100%) |
| Landmarks | 2,459 rows / 2,445 unique images | **2,547** (100%) |
| Geometric features | 2,459 rows, **all truncated** | **2,547** regenerated |
| Canonical split | 2,511 rated rows, leaks across splits | **2,490** rows, deduped and existence-checked |
| Per-rater scores | partly-redundant spreadsheets in 4 identifier conventions, raw Worker IDs | **1 table**, 141,736 rows, 860 raters incl. the never-released in-house panel, pseudonymized, `generic`/`date` tasks separated |
| Labels | Encoded in folder paths | In metadata; images content-addressed |

The single most consequential finding: **the original `FaceNet_512_features`
covered only 8 of 151 `male/indian` images (5.3%)**, versus ~100% for every
other subgroup. Anyone who trained on the original embeddings was training
on a set missing ~95% of one ethnicity/gender group.

---

## 1. Silent failures recovered

The original `face_crop_align.py` wraps face detection in a bare
`except: print(...)`, so images where detection failed were skipped without
error and never appear in the derived artifacts. The same pattern affected
landmarks. The result is a set of gaps that look like the data simply
"doesn't have" those rows.

### 1.1 The `male/indian` embedding gap (Finding 3)

MTCNN failed on **143 of 151 `male/indian` images (94.7%)**, against roughly
0% elsewhere. This is an isolated pipeline/environment failure, not a
property of the images — the independent OpenCV backend processed the same
folder normally, and all 143 were recovered and visually spot-checked as
genuine faces.

| Subgroup | Original embedding coverage | Now |
|---|---|---|
| `male/indian` | 8 / 151 (5.3%) | 151 / 151 (100%) |
| all others | ~100% | 100% |

**If you used the original `FaceNet_512_features`, your subgroup analyses
are affected.** This is the one change that can alter published conclusions
rather than just tidying the data.

### 1.2 Remaining crop and landmark gaps (Findings 3, 4)

- **145 missing MTCNN crops** → all recovered (100%).
- **197 missing OpenCV crops** → 189 recovered (96%). The remaining 8 have
  no face detectable by that detector family even under expanded tuning;
  candidates were cross-validated against MTCNN (IoU ≥ 0.3) and 8 of 57
  initial hits were rejected as false positives — one was an ear.
- **102 missing landmark rows** (not the 88 that `2,547 − 2,459` suggests —
  14 rows in `landmarks.csv` duplicate an image already covered) → all
  recovered (100%).

### 1.3 Geometric features were entirely unusable (Finding 5)

Every one of the 2,459 rows in the original `geometric_features.csv` is
**truncated** — the values were written via a string repr that elides the
middle of each vector with `...`. Not 98%, not "some rows": all of them.
Regenerated from landmarks for all 2,547 images.

Two recovered images have geometric ratios that are mathematically undefined
(`NaN`, not missing) — both extreme profile poses where two landmark points
collapse onto the same pixel. These are reported as `NaN`, not silently
zero-filled or dropped.

---

## 2. Corrections

### 2.1 Split leakage and unjoinable ratings (Finding 6)

The 2022-generation split leaked **42 rows** across train/val/test: 3 by
identical filename, and 39 more by identical image *content* under different
filenames — invisible to any filename-based check.

Separately, **33 of 2,511 canonical rating rows referenced an image that no
longer existed at its rated path**:

- **4** from a parser bug — split lines quote paths containing spaces
  (`"foo (1).jpg" 5.5`), and the quotes were not stripped, silently breaking
  path normalization.
- **7** from images later refiled into a different ethnicity/gender folder —
  remapped, rating kept.
- **22** from images removed from the dataset since rating — dropped, as
  there is nothing to join them to.

Every remaining row is now verified to join to a real image file.

### 2.2 A dedup bug this repository introduced and fixed

Worth recording because it is the kind of error the restructure is designed
to prevent: the deduplication logic originally keyed on *bare filename*,
which conflated `male/indian/shivam-singh-2_X6NMP-E_U-unsplash.jpg` with
`male/mideastern/shivam-singh-2_X6NMP-E_U-unsplash.jpg` — **two different
photographs** that happen to share a filename (confirmed by SHA-256). Fixed
to key on the full normalized path.

### 2.3 Conflicting labels (Finding 8)

8 filenames appeared under two different `gender/ethnicity` folders at once.
Seven were the same photo filed twice under conflicting labels; each was
visually reviewed and assigned a single correct label:

| File | Chosen | Rejected |
|---|---|---|
| `huu-chung-dang-lP02hkcp7H0-unsplash.jpg` | `female/asian` | `female/indian` |
| `julian-florez-l5rmMuK8070-unsplash.jpg` | `female/hispanic` | `female/mideastern` |
| `kunal-goswami-YHSohAq-PuI-unsplash.jpg` | `female/indian` | `female/mideastern` |
| `pexels-anna-shvets-4971982.jpg` | `male/caucasian` | `female/caucasian` |
| `pexels-moh-mckenzie-3597035.jpg` | `male/black` | `female/black` |
| `raamin-ka-4lQmQ_DBbNc-unsplash.jpg` | `female/mideastern` | `female/hispanic` |
| `tobi-oshinnaike-Z7MKNGFnbOw-unsplash.jpg` | `male/black` | `male/caucasian` |

The 8th (`shivam-singh-...`) was never a label conflict — see 2.2.
`metadata.parquet` retains `has_label_collision` so the history stays
visible, plus `label_collision_resolved`.

### 2.4 Duplicate images (Findings 7, 12)

- **49 groups of byte-identical images (98 files)** stored under different
  filenames. Content-addressing collapses these automatically; 18 of the
  pairs had previously leaked across split boundaries.
- **1 near-duplicate pair** that exact hashing cannot catch — the same
  photograph re-saved under two stock-platform names. Found by perceptual
  hashing (threshold validated against the dataset's own distribution: the
  true match sits at Hamming distance 16, the next-closest candidate at 58).
  Both are in `train`, so this is redundancy rather than leakage. Flagged
  via `has_near_duplicate`, **not** removed.

### 2.5 Truncated filenames cost 5 rated images (Finding 15)

Five images are stored under a filename whose stem was cut to exactly 32
characters, while the score files kept the full name — four end mid-word in
a partial `-unsplash` suffix (`-uns`, `-unsp`, `-unspl`, `-unspla`). Because
ratings were matched on the full name, all five were counted as "image no
longer exists" and dropped: **five genuinely rated images excluded for a
filename bug.** Now recovered, but only where unambiguous — one case
(`jonathan-borba-...-unspl.jpg` vs `...-unsplash.jpg`) is **two different
photographs**, and a naive match would have assigned one photo's score to
the other. Canonical train: 1,746 → **1,751**.

### 2.6 Two rating tasks were pooled into one score column (Finding 16)

**The most consequential defect found**, and one this repository introduced
rather than inherited. The collection ran two distinct rating tasks over the
same images — *generic* attractiveness and *date* attractiveness, whose
means sit about a point apart (currently 68,974 ratings at 5.86 and 72,762
at 4.80; the counts quoted below are the pre-Finding-21 figures that were
current when this defect was found). The per-rater
reconciliation discarded the source label and wrote a single
undifferentiated `score`, so:

- Two different questions were merged into one label with no way to separate
  them.
- Deduplication keyed on `(image_id, rater)` treated a rater's *generic* and
  *date* answers about the same image as duplicates and dropped one —
  **7,555 real ratings destroyed**, with which one survived depending on
  source load order.

The canonical aggregate corresponds to the **generic** task (bias −0.06,
correlation 0.966), so the pooled table was offset +0.51 from the very
labels it was meant to explain. Fixed: `ratings_by_rater.parquet` now
carries `rating_type`, dedup keys on
`(image_id, rater_id, rating_type)`, and rater-quality consensus is computed
within a task. **Filter `rating_type == "generic"` for the per-rater
equivalent of the canonical label.**

### 2.7 Source-URL inference bug

Inferred Unsplash URLs were being built by splitting the filename on its
*last* hyphen — but Unsplash IDs are a fixed 11 characters and may contain
hyphens internally. **195 of 1,325 Unsplash-pattern filenames (14.7%)** had a
truncated, wrong URL (e.g. `.../photos/PuI` instead of
`.../photos/YHSohAq-PuI`). Fixed, regression-tested, and
`image_provenance.csv` regenerated (317 rows changed).

---

## 3. Removals

**Three images of named, recognizable public figures** were found actively
rated in the canonical train split and have been excluded (Finding 13):
Michelle Obama, Deepika Padukone, Aditi Rao Hydari.

This is a content/consent issue rather than a data-quality bug. Rating a
named individual's likeness for attractiveness and distributing the result
as research data raises publicity-rights and consent questions that a stock
photo licence does not address.

**The screening is known to be incomplete.** The first pass used a filename
regex matching exactly two hyphenated words before a numeric ID. A second,
broader pass immediately found a third case (a three-word name) that the
first pass *structurally could not* have caught. Treat "3 found" as a floor,
not a ceiling — a reliable screen requires visual/perceptual identity
matching, not filename patterns. Two of the three images' source URLs no
longer resolve on Pixabay, which is at least consistent with the platform
having removed them too.

---

## 4. Structural changes

Images are now **flat and content-addressed**: `images/<sha256>.<ext>`, with
`gender`/`ethnicity` living only in `metadata.parquet`.

The original `{gender}/{ethnicity}/<file>` layout was the direct mechanism
behind the two largest classes of bug found in this audit — a photo could be
physically filed under two labels at once (2.3), and relabelling required
moving a file, which happened inconsistently (2.1). Content-addressing makes
duplicates collapse automatically, makes relabelling a metadata edit, and
removes filename-parsing as a failure mode.

```
data/mebeauty_v3/
├── images/
│   ├── <sha256>.jpg              2,495 unique images
│   └── metadata.parquet          labels, provenance, duplicate/collision flags
├── landmarks.parquet             68-point landmarks, native list<float>
├── ratings/
│   ├── aggregate/{train,val,test}.parquet    1,751 / 222 / 517
│   ├── distributions.parquet                 4,972 soft labels over the 1-10 scale
│   └── by_rater/
│       ├── ratings_by_rater.parquet          141,736 rows, 860 raters, generic+date
│       └── rater_quality.parquet             860 raters, quality statistics
├── croissant.json                schema-validated ML metadata
└── COVERAGE.json
```

`load_dataset("imagefolder", data_dir="data/mebeauty_v3/images")` works
directly — verified against the real directory. (`metadata.parquet` must sit
*inside* `images/`; at the dataset root, Hugging Face's split auto-detection
mistakes the sibling `ratings/aggregate/{train,val,test}.parquet` filenames
for split definitions and yields empty phantom splits.)

Derived pixel artifacts (crops, embeddings) are deliberately **not** shipped —
they are regenerable from images + landmarks, and shipping them fixes a
preprocessing choice that should stay the consumer's. See
[`REPRODUCE_LEGACY_BASELINE.md`](REPRODUCE_LEGACY_BASELINE.md).

---

## 5. New artifacts

- **Per-rater ratings** (`ratings/by_rater/ratings_by_rater.parquet`) —
  141,736 individual scores across 2,487 images and 860 raters, reconciled
  from overlapping spreadsheets in four different identifier conventions. Raw
  MTurk Worker IDs are pseudonymized to stable `rater_XXXX` identifiers; the
  mapping is kept local-only and never committed.
  **Includes the in-house rater panel (29 members, `panel_XXXX`), which no
  previous release contained** — their columns are demographic codes rather
  than Worker IDs, so an earlier prefix filter dropped them silently
  (Finding 21). An earlier version of this document described
  `date_scores_all.xlsx` as having "corrupted column headers"; that was
  wrong — those headers are panel raters. Both `date_scores_all*.xlsx` remain
  excluded, now as superseded rather than broken, since the panel's own
  per-rater files are the authoritative source. Redundant supersets and the
  two `private_date_{female,male}.xlsx` merges are also excluded to avoid
  double-counting.
- **Rating distributions** (`ratings/distributions.parquet`) — 4,972 soft
  labels, one per (image, task), with raw counts and normalized probabilities
  over the 1-10 scale. Supports label distribution learning, which the
  original release's single mean score could not.
- **A reproducible point label** (`score_mean`, in each aggregate split) —
  the plain unweighted mean of every generic rating, equal to the mean of the
  shipped distribution. The original `score` is retained unchanged but cannot
  be recomputed from any released data (Finding 20).
- **Per-rater quality statistics** (`rater_quality.parquet`) — see §6.
- **Provenance** (`reports/legacy_audit/image_provenance.csv`) — inferred
  source platform and photo ID per image, so a future consent or takedown
  request resolves to a single photograph. **Inferred from filenames only,
  never verified against the live platforms**; 101 images (4.0%) have no
  source link at all, and every automated recovery route has been tried and
  failed (no filename signal, no embedded metadata, no perceptual match
  within the validated threshold) — reverse image search by hand is the only
  remaining option. See `reports/legacy_audit/unknown_provenance_recovery.json`.
- **The other two split generations** reconciled to the same rigour as the
  canonical one (`canonical_splits_{original,crop}/`).
- **Croissant metadata**, schema-validated with `mlcroissant`.
- **Datasheet, dataset card, citation file** — `DATASHEET.md`,
  `DATASET_CARD.md`, `CITATION.cff`.

---

## 6. Rater quality: measured, deliberately not filtered

Per-rater quality statistics now ship, including a **discrimination spread**:
how much higher a rater scores images the rest of the pool ranked in its top
consensus third versus its bottom third, computed leave-one-out so no rater
helps define the yardstick they are measured against.

**No rater is excluded or down-weighted. The canonical score remains the
unweighted mean of every rating**, which keeps it reproducible and
comparable to the original release. Filtering is left to the consumer, for
three reasons documented in full under Finding 14:

1. **No threshold is principled** — across 557 raters with ≥30 ratings the
   spread is a smooth continuum from −8.44 to +7.96 (median +2.38) with no
   gap. Cutoffs that all sound defensible remove between 5.3% and 18.2% of
   ratings.
2. **The impact is not marginal** — excluding spread ≤0 moves 576 images'
   mean score by >0.25; at ≤1.0 it moves 1,119 images (~45% of the dataset).
3. **The rule is circular** — filtering by agreement-with-consensus defines
   "good rater" as "agrees with the majority" and deletes exactly the
   minority-viewpoint variation this dataset exists to study.

---

## 7. What did *not* change

- **No image pixels were modified.** Images are byte-identical copies of the
  originals, renamed to their own SHA-256.
- **No rating value was altered.** Ratings were re-joined, deduplicated and
  existence-checked; none were edited.
- **The original repository is untouched**, and the legacy snapshot under
  `data/legacy_snapshot/` is a checksummed copy retained for reproduction.
- **The canonical aggregate remains a plain unweighted mean.**

---

## 8. Known discrepancies not resolved here

- **The paper's abstract states 2,550 images (1,250 men / 1,300 women).**
  This snapshot has 2,547 (1,196 / 1,351). The underlying image set has
  changed since publication in a way that cannot be reconstructed from the
  repository. The most likely reading of "2,550" is a pre-deduplication
  split total rather than a file count (Finding 2), but this is inference.
- **Which split generation produced the published results is unknown.**
  Three exist (`train.txt`, `train_2022.txt`, `train_crop.csv`). The
  "original" generation has 252 relabelled rows out of 2,514 (10%) versus
  0.3% for "2022", suggesting "2022" is a later cleaned revision — suggestive,
  not proof. Resolving it needs the paper's full methodology section.
- **Reported baselines were not reproduced.** No claim is made about
  reproducing the paper's numbers, and none should be inferred.
- Crop-regeneration scripts reproduce *this repository's* recovered crops
  exactly (0.0 pixel difference across all 2,547), but **not** the original
  legacy crops bit-for-bit (mean difference 43.9; 0 of 2,401 within 1.0) —
  methodologically equivalent, not pixel-identical. Full detail in
  [`REPRODUCE_LEGACY_BASELINE.md`](REPRODUCE_LEGACY_BASELINE.md).

---

## Still open

Not resolved, and not resolvable without decisions or input beyond this audit:

- **Licensing.** No licence is selected. The intended model is a gated
  release under a custom research-use agreement rather than an open licence,
  but the agreement text has not had legal review. **Nothing here is
  licensed for redistribution.**
- **Redistribution rights** per source platform are inferred from filenames,
  never verified.
- **GDPR / biometric classification.** Face embeddings and landmarks may
  constitute biometric data depending on processing and use. No legal review
  has happened; no data controller is named. Embeddings are *not* lower-risk
  than the photographs — they encode stable, matchable facial characteristics.
- **Identity screening** is filename-based and demonstrably incomplete (§3).
- **Score disparities by subgroup** are documented as descriptive fact (raw
  unadjusted means: female 6.58 vs. male 5.56; by ethnicity 5.41–6.46) but
  not explained. Whether this reflects rater bias, sample composition, or
  something else is a research question this audit cannot settle.

---

## Reproducing

Every number above comes from a script in `scripts/data/`, run against a
checksummed copy of the pinned legacy commit. The full command sequence is
in [`DATASET_AUDIT.md`](DATASET_AUDIT.md#reproducing-this-document); reports
are under `reports/legacy_audit/`.

```
make check    # lint, format, tests
```
