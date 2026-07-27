# Legacy Dataset Audit

Reproducible findings from running the `scripts/data/*.py` tooling against a
checksummed snapshot of the legacy repository. This document supersedes the
scattered "Finding N" references in individual script docstrings — those
referred to an earlier, unwritten findings list (only Findings 1, 2, 5, 6 and
10 were ever named in code, and 3/4/7/8/9 could not be located anywhere in
either repo). Everything below is renumbered from scratch and each number is
backed by a command you can re-run.

## Source

- Legacy repository: `https://github.com/fbplab/MEBeauty-database.git`
- Commit: `7b849562ee92d99d34d56afa2ebe85e5075f9b20` (see `docs/LEGACY_SOURCE.md`)
- Verified: `git -C ~/Research/MEBeauty-Legacy rev-parse HEAD` matches exactly.
- Snapshot: a checksummed, file-for-file copy was made with
  `scripts/data/copy_legacy_snapshot.py` to `data/legacy_snapshot/`
  (gitignored, 9,808 files, 242,073,620 bytes). All commands below ran
  against that copy; the legacy repository itself was never written to.
  Manifest and provenance: `data/legacy_snapshot.manifest.csv`,
  `data/legacy_snapshot.provenance.json`.

## Finding 1 — `original_images/` totals 2,547 files, 3 of them PNG

```
find original_images -type f | wc -l          # 2547
find original_images -type f ! -iname '*.jpg'  # 3 PNG files:
  male/hispanic/couple-5917009_1920.png
  male/caucasian/0.png
  female/caucasian/f3.png
```

Reproduced in `reports/legacy_audit/inventory/summary.json`
(`top_level_file_counts.original_images: 2547`).

## Finding 2 — the legacy README's "2550 images" is not the raw file count, but likely the pre-dedup 2022 split total

`README.md` states "2550 images." That does not match Finding 1 (2,547 raw
files). It does match the 2022 split generation almost exactly:

```
wc -l scores/train_2022.txt scores/val_2022.txt scores/test_2022.txt
# 1787 + 230 + 536 = 2553 raw rows
```

2,553 raw rows minus the 3 filename-duplicate rows fixed in Finding 6 below
equals **2,550** — exactly the README's number. This is the most likely
explanation (rated split rows, not distinct source images) but is not
confirmed against the README's actual authoring process, so treat it as a
strong hypothesis, not a settled fact. Note this hypothesis only accounts
for the filename-level fix: the fully canonical split in Finding 6, which
also resolves relabeled/missing images and removes content-duplicate
leakage, is smaller still (2,488) — the README's number almost certainly
predates any awareness of same-photo, different-filename duplicates, let
alone images that were later reclassified or removed entirely.

**External corroboration, and a new mismatch.** The published paper's own
abstract (not just the legacy README) states the dataset is "2550 in-the-wild
facial photos... 1250 men and 1300 women" [Lebedeva, Guo & Ying, 2022,
via ResearchGate abstract]. That confirms 2,550 is a real, paper-stated
figure, not a README typo — but the stated **gender split does not match
this snapshot at all**: `original_images/` currently has 1,196 male and
1,351 female files (2,547 total), not 1,250/1,300. Both the total (2,547 vs
2,550) and the per-gender counts (off by 54 and 51 respectively, in
opposite directions) disagree with the paper. This means the dataset
composition has changed since publication (images added, removed, or
reclassified between genders) — the pre-dedup-2022-split-total hypothesis
above may still explain where "2550" as a *number* came from, but it can no
longer be read as "this snapshot, lightly deduped": the underlying image
set itself is not the one the paper described. Which images were added or
removed, and when, is not reconstructable from this repository.

## Finding 3 — FaceNet/crop counts diverge from Finding 1 because face detection failures were silently dropped

```
find cropped_images/images_crop_align_mtcnn -type f | wc -l    # 2403
find cropped_images/images_crop_align_opencv -type f | wc -l   # 2351
find FaceNet_512_features -type f -iname '*.csv' | wc -l       # 2403
```

MTCNN-backend crop count (2,403) matches the FaceNet-512 feature count
(2,403) exactly — every image that got a face crop also got an embedding.
But one of those 2,403 crops, `female/hispanic/islander-mtcnn-kwHBTNUQIqs-unsplash.jpg`
(plus its FaceNet feature), has **no matching file anywhere in
`original_images/`** — an orphaned derived artifact from a source image that
was since renamed or removed. It has no OpenCV crop, no landmarks row, and
never appears in any split file, so it was never rated. Net effect: 145 of
2,547 originals (5.7%) have no MTCNN crop (144 by simple arithmetic, plus one
more because the 2,403rd crop doesn't correspond to a current original);
196 (7.7%) have no OpenCV crop.

**The 5.7% aggregate MTCNN failure rate hides a severe, single-subgroup
failure, not a random one.** Broken down by folder:

| Folder | Total | MTCNN missing | OpenCV missing |
|---|---|---|---|
| `male/indian` | 151 | **143 (94.7%)** | 13 (8.6%) |
| every other folder (11 folders, 2,396 images combined) | 2,396 | 1 (0.04%) | 183 (7.6%) |

**143 of the 144 real MTCNN failures (99.3%) are in one folder.** Every
other ethnicity/gender combination has an MTCNN failure rate at or near
zero. Critically, the *independent* OpenCV backend processed `male/indian`
at a completely unremarkable rate (8.6%, in line with every other folder) —
proof the images themselves are not unusually hard to detect faces in; only
the MTCNN backend's run over that one folder failed almost completely.
`face_crop_align.py` uses `os.walk`, which processes every file in one
directory before moving to the next, consistent with a transient
model/environment failure (e.g. GPU memory exhaustion, a corrupted model
load) that happened to coincide with the `male/indian` batch specifically —
not a per-file cause, since the code has no folder-specific logic at all.
The original run's logs no longer exist, so the exact mechanism can't be
confirmed, only the effect: **anyone who used the original
`FaceNet_512_features` for any downstream task had only 8 of 151
male/Indian embeddings (5.3% coverage) versus ~100% for every other
subgroup** — a severe, undocumented representation gap in a dataset whose
stated purpose is multi-ethnic fairness, caused by an infrastructure bug,
not a genuine data limitation. Recovery (below) confirms this: every one of
the 143 recovered crops was visually spot-checked and shows a normal,
correctly-centered face, no different in quality from any other subgroup.

Two more angles were checked to narrow the mechanism down further, both
negative (i.e. they rule things out rather than explain it): the 8 images
that *did* succeed are scattered across the folder's full alphabetical
range (positions 9, 18, 38, 64, 104, 115, 126, 128 of 151) rather than
clustered at the start or end, arguing against a simple "crashed early,
never recovered" story; and their file size/dimensions/format (mostly
500×500 or 400×400 RGB JPEGs) are indistinguishable from a random sample of
the failures, ruling out corrupted or unusually-formatted source files as
the cause. This is consistent with an intermittent environmental failure
(e.g. GPU memory pressure, a flaky model backend) rather than anything
about specific files — but confirming that precisely would need the
original 2021-era run's logs, which don't exist. The recovery above already
fixes the actual data gap; this paragraph documents the limit of what's
forensically reconstructable about *why* it happened.

Root cause, `face_crop_align.py:42-48`:

```python
try:
    detected_face = DeepFace.detectFace(img_path=fullname, detector_backend=method)
    cv2.imwrite(...)
except:
    print("The file ", ..., " has no face")
```

The bare `except` swallows every detection failure, prints to stdout, and
never records a persistent failure list or raises. There is no artifact
anywhere in the legacy repo listing which 144 (MTCNN) or 196 (OpenCV) images
failed — only the count is inferable, by set difference against
`original_images/`.

**Recovered.** `scripts/data/recover_missing_crops.py` re-ran detection on
exactly the missing images using maintained detector implementations
(`facenet_pytorch.MTCNN`, and `cv2`'s Haar cascade for the OpenCV backend),
logging every outcome instead of swallowing it:

| Backend | Attempted | Recovered | Genuinely no face detected |
|---|---|---|---|
| MTCNN (`facenet_pytorch`) | 145 | **145 (100%)** | 0 |
| OpenCV (Haar cascade, first pass) | 197 | 140 (71%) | 57 |
| OpenCV (expanded tuning, verified) | 57 | **49 (86%)** | 8 |
| **OpenCV total** | 197 | **189 (96%)** | 8 |

MTCNN recovered every single previously-missing image — the legacy
failures were an artifact of the buggy wrapper, not genuinely undetectable
faces. FaceNet-512 embeddings were also computed for all 145 recovered
MTCNN crops (`facenet_pytorch.InceptionResnetV1`, `vggface2` weights).

The 57 initial OpenCV misses got a second pass: multiple Haar cascades
(`default`/`alt`/`alt2`/`alt_tree`), histogram equalization, and a wider
scaleFactor/minNeighbors sweep. That alone "recovered" all 57 — a red flag,
since loosening a Haar cascade's parameters is known to inflate false
positives (confirmed by inspection: one candidate box was centered on an
ear, not a face). Every candidate was therefore cross-validated against an
independent detector (`facenet_pytorch.MTCNN`, run directly on the full
image) via IoU ≥ 0.3 between the two detections; candidates below that
threshold were rejected rather than kept. Spot-checking rejected candidates
found a blank white patch and an off-target crop dominated by a non-face
object — the filter is doing real work, not just discarding at random. 49
of 57 survived verification; the remaining 8 stay genuinely unrecovered
under this method family. Full detail (including the false-positive box
coordinates and IoU scores): `reports/legacy_audit/crop_recovery_report.json`.

Recovered artifacts are written to `data/recovered_crops/`
(gitignored — derived image/embedding data, not committed) and are kept
**separate from the legacy `cropped_images/`/`FaceNet_512_features/`
directories** so it is never ambiguous which pipeline produced which file.
Full per-image outcomes: `reports/legacy_audit/crop_recovery_report.json`.

## Finding 4 — `landmarks.csv` has the same silent-failure pattern — 102 images, not 88

```
python3 -c "import csv; print(sum(1 for _ in csv.reader(open('landmarks.csv')))-1)"  # 2459
```

2,459 rows looks like only 88 images (2,547 − 2,459) lack a landmark row,
but that arithmetic assumes every row is a distinct image. It isn't: 14
rows in `landmarks.csv` are duplicates of an image already covered
elsewhere in the file (contributing no new coverage), leaving only 2,445
unique images landmarked. The real gap is **102 images (4.0%)**, with no
recorded reason for any of them — the same class of bug as Finding 3, in a
different pipeline stage. (Unlike Finding 3's orphaned crop, every path in
`landmarks.csv` does match a current `original_images/` file — no orphans
here.)

**Recovered — 102/102 (100%).** `scripts/data/recover_missing_landmarks.py`
re-ran 68-point landmark detection on all 102 missing images using
`face_alignment` (PyTorch-based; standard ibug/300-W 68-point ordering,
positionally compatible with the dlib numbering
`mebeauty_benchmark.legacy.geometry` assumes). Two issues surfaced and were
fixed during the run, both informative in their own right:

- **One image, `male/indian/128.jpg`, is 5304×6630px** (35 megapixels) —
  far larger than the rest of the dataset — and its full-resolution
  inference was killed by the OS (OOM). Fixed by downscaling any image
  over 1600px on its long side before detection and rescaling the
  resulting landmark coordinates back to original-image pixel space.
- **One image, `female/caucasian/f3.png`, has an alpha channel** (RGBA),
  which crashed the detector's internal tensor shapes. Fixed by explicitly
  converting to RGB before detection — the same class of issue as the 3
  PNGs noted in Finding 1, now confirmed to actually matter downstream.

Recovered landmarks and their derived geometric features (11,628-dim,
computed with the exact same formula as Finding 5) are written to
`data/recovered_landmarks/` (gitignored), kept separate from the legacy
`landmarks.csv`. Full per-image outcomes:
`reports/legacy_audit/landmark_recovery_report.json`.

## Finding 5 — `geometric_features.csv` is 100% truncated, not partially

```
python3 -c "
import csv
rows = list(csv.reader(open('geometric_features.csv')))[1:]
print(sum(1 for r in rows if '...' in ','.join(r)), '/', len(rows))
"
# 2459 / 2459  (100%)
```

Every row is truncated, not "98%" as originally estimated. Root cause: the
legacy notebook wrote each row's feature vector with `str(numpy_array)`,
which elides the middle of any sufficiently long array with `...`. The
feature vectors are otherwise fully recoverable because `landmarks.csv`
(the input) is intact. `scripts/data/regenerate_geometric_features.py`
re-derives them from `landmarks.csv` using the exact formula in
`get_landmarks_geom.features.ipynb` (`C(19,4) × 3 = 11,628` dimensions per
image) and wrote **2,459 vectors, 0 parse failures**, matching Finding 4's
count exactly, to `data/geometric_features/geometric_features.npz`
(gitignored — derived dataset artifact, not committed).

**A second, smaller data-quality issue surfaced when merging this with the
recovered landmarks (Finding 4) into `data/mebeauty_v2/`:** the ratio
formula divides by the distance between two landmark points, which is
mathematically undefined (`inf`) if those two points coincide. This
happens for 2 of the 102 recovered images —
`female/asian/pexels-raydar-341970.jpg` (70 of 11,628 features) and
`male/asian/yang-deng-C4bZ_42MAYA-unsplash.jpg` (133 of 11,628 features) —
both confirmed by direct inspection to be extreme profile-view photos,
where the standard frontal 68-point scheme collapses distinct points (e.g.
the far eye corner) onto the same pixel as a visible landmark. Not present
anywhere in the legacy dlib-derived landmarks — a `face_alignment`-specific
limitation on this pose type, not a bug introduced by downscaling or the
RGB-conversion fix. `assemble_v2_dataset.py` converts these `inf` values to
`NaN` (the correct "undefined", not "missing", marker) rather than leaving
a silent `inf` to break a future user's gradient computation, and lists the
affected images in `data/mebeauty_v2/COVERAGE.json`'s
`geometric_feature_rows_with_undefined_ratios`. All 2,547 FaceNet-512
embeddings (legacy + recovered) were also checked directly: correct
dimensionality, no NaN/Inf, no zero-norm vectors.

## Finding 6 — the 2022 split generation leaks 42 rows total: 3 by filename, 39 by image content

`scripts/data/build_canonical_splits.py`'s docstring originally assumed "one
basename shared between train and val." Running it against the real data
found two distinct kinds of leakage, from two independent checks:

**Filename dedup** — the same basename appearing twice, anywhere:

```json
"removed_for_filename_leakage": [
  {"basename": "asian-girl-4819726_1920.jpg", "dropped_from_split": "train"},
  {"basename": "asian-girl-4819726_1920.jpg", "dropped_from_split": "train"},
  {"basename": "asian-girl-4819726_1920.jpg", "dropped_from_split": "val"}
]
```

That file is duplicated **twice within `train_2022.txt` itself**, plus once
more in `val_2022.txt`. 3 rows removed.

**Content-hash dedup (new)** — a filename check cannot see the same photo
saved twice under two *different* names. Hashing every file under
`original_images/` (see Finding 7) and cross-referencing against the split
files found this affects the splits directly: **18 of the 49 duplicate
image pairs straddle train/val/test**, e.g. `female/asian/30.jpg` (test) is
byte-identical to `female/asian/asian-girl-4819726_1920.jpg` (train) — the
same photo rated as two different test items under the old split. Extending
`build_canonical_splits.py` with a second, content-hash-based dedup pass
(`dedupe_by_content_hash` in `mebeauty_benchmark.legacy.splits`) removed 39
rows total (18 cross-split leaks plus 21 same-split redundant duplicates,
collapsed under the same "one row per distinct image" rule the filename
dedup already applied).

**A third check was added after the fact, and it mattered more than either
of the other two.** Building `data/mebeauty_v2/` and cross-checking every
canonical rating row against the actual merged crop files found **33 of
2,511 rows didn't join to any image at all** — the two dedup passes above
only ever operate on whatever path string is already in a row; neither
checks that the path still points at a real file. Two distinct causes:

- **4 rows use a quoted path** (`"foo (1).jpg" 5.5`, because the filename
  itself contains a space) that `parse_split_lines` never stripped — the
  leading `"` silently broke every prefix-normalization check downstream,
  so the row parsed without error but could never be joined to a file.
  Fixed in `parse_split_lines` itself.
- **29 rows reference an image that isn't at its rated path any more.**
  Cross-referencing by basename against the current `original_images/`
  tree splits this into two real outcomes: **7 images still exist, just
  under a different ethnicity/gender folder** — reclassified sometime
  after the rating was recorded (one, `imad-clicks-2_qmEnz7bQ4-unsplash.jpg`,
  changed gender too: `male/mideastern` → `female/mideastern`). **22 images
  no longer exist anywhere** in this snapshot — removed since the rating
  was made, with nothing left to join the rating to.

`mebeauty_benchmark.legacy.splits.resolve_to_existing_images` now runs
before both dedup passes: it remaps the 7 relabeled rows to their current
path (keeping the rating, correcting the path) and drops the 22 (now 23 —
one of the 4 quote-parsing fixes also turned out to resolve to nothing)
that have no current file. Rows resolving to more than one current
candidate (possible for the label-collision basenames — Finding 8) are
dropped rather than guessed at; none occurred in this split generation.

Canonical, fully-resolved-and-deduped output
(`reports/legacy_audit/canonical_splits/`): **train 1,749, val 222, test
517 (2,488 total)**, down from the 2,553 raw rows via existence resolution
(23 dropped, 7 relabeled) then the two dedup passes (3 exact-path, 39
content-hash). Every one of these 2,488 rows was verified to join to an
actual crop file in `data/mebeauty_v2/` — zero missing, checked directly,
not assumed. Full trail for all three passes:
`reports/legacy_audit/split_dedup_report.json`. This is one candidate
protocol (`benchmark-v1` in the wider plan); the historical paper protocol
(whichever split generation actually produced the published results) has
not yet been identified and is out of scope for this pass — see Open
items.

There are two other split generations in the legacy repo. Running the same
two dedup checks against them shows the leakage is not specific to the 2022
generation — it's a property of the underlying duplicate images (Finding 7),
present at a comparable rate in all three.

**A real bug was found and fixed while doing this.** The filename-dedup
pass originally keyed on the bare basename (`os.path.basename`), not the
full normalized path. Cross-referencing Finding 7's content-duplicate
groups against Finding 8's label collisions found that 7 of those 8
basename collisions are the same photo filed twice (a real duplicate,
correctly caught either way) — but one,
`shivam-singh-2_X6NMP-E_U-unsplash.jpg` (`male/indian/` vs.
`male/mideastern/`), is **two different photos that happen to share an
identical filename** (confirmed by SHA-256: different content). A
basename-only key cannot tell these apart from a real duplicate. It never
corrupted the persisted `benchmark-v1` canonical split (that image is only
rated once in the 2022 generation), but it did silently drop 4 legitimate,
distinct rows from the "original" generation reconciliation below in an
earlier version of this table — the same score
(`5.894444444444445`, suspiciously identical to 15 decimal places, and
itself worth treating as a red flag about how that generation's scores
were assigned) had been copy-pasted onto both filenames, and only the
full-path fix caught that these are not interchangeable. `dedupe_across_splits`
now keys on the full normalized path
(`mebeauty_benchmark.legacy.splits.dedupe_across_splits`); cross-folder
same-*content* duplicates (the other 7 cases) are still caught, correctly,
by the separate content-hash pass.

**Updated: all three generations now have real, built, verified output
files** (`scripts/data/build_canonical_splits.py --generation
{2022,original,crop}`), not just the informal reconciliation counts this
table originally reported — every number below is a rebuild, not an
estimate:

| Split generation | Raw rows | Relabeled | Dropped: gone | Dropped: ambiguous | Exact-path dups | Content dups | Final | Leak rate |
|---|---|---|---|---|---|---|---|---|
| `train_2022`/`val_2022`/`test_2022` | 2,553 | 7 | 23 | 0 | 3 | 39 | 2,488 | 2.5% |
| `train`/`val`/`test` (original) | 2,514 | **252** | 18 | 2 | 1 | 43 | 2,450 | 2.5% |
| `train_crop`/`test_crop` (no val) | 2,056 | 0 | 0 | 0 | 0 | 35 | 2,021 | 1.7% |

Output files: `reports/legacy_audit/canonical_splits_{2022,original,crop}/`
and `split_dedup_report_{2022,original,crop}.json`. (The `2022` generation
is additionally what feeds `data/mebeauty_v3/`, where Finding 13's 2
exclusions bring it down further to 1,747/222/517.)

**The "original" generation has 252 relabeled rows — 10% of it — versus 7
(0.3%) in the 2022 generation.** That gap is a strong, previously-invisible
clue about chronology: it's consistent with "2022" being a cleaned-up
revision produced *after* a large ethnicity/gender relabeling effort, with
"original" predating that cleanup (matching the plain reading of the
filenames themselves — "original" vs. a dated "2022" revision). The `crop`
generation has zero relabels, suggesting it may be a narrower, later export
that only ever included already-stable classifications. None of this
confirms which generation (if any) matches the *published paper's* numbers
— that still needs the paper's full text, not just its abstract (Finding
2) — but it does make "2022 is the most-maintained, most-corrected
generation" a more evidenced claim than before, not just "the one with the
most rows."

## Finding 7 — 49 groups of byte-identical images exist under different filenames, 98 files total

`scripts/data/find_duplicate_images.py` hashes every file under
`original_images/` and groups exact byte-for-byte duplicates, independent of
the splits. Result: **49 duplicate pairs (98 files, 49 redundant copies)** —
about 2% of the dataset is a second copy of an image already present under
another name. Pattern: a numeric legacy filename (`30.jpg`) and a
platform-slug filename (`asian-girl-4819726_1920.jpg`) turn out to be the
same photo. One pair (`shivam-singh-DOmEd6CYP5M-unsplash.jpg` and its
`(1)` copy) is a duplicate at the source but has *no* MTCNN crop for either
copy (both independently failed face detection — see Finding 3), so it
doesn't affect the splits either way. Full list:
`reports/legacy_audit/duplicate_images.json`. 18 of these 49 pairs are the
train/val/test leakage fixed in Finding 6; the remaining 31 are same-split
or not-in-any-split redundant copies, left in place except where Finding 6's
canonical-split dedup already removes them.

## Finding 8 — 8 images are filed under conflicting ethnicity/gender labels

`scripts/data/find_label_collisions.py` against `original_images/` found 8
filenames present in two label folders simultaneously — 6 are ethnicity
conflicts, 2 are **gender** conflicts. Cross-referencing each pair's
SHA-256 (Finding 7's data) distinguishes two genuinely different situations
that look identical from the filename alone:

| File | Conflicting folders | Same photo (SHA-256)? |
|---|---|---|
| `pexels-anna-shvets-4971982.jpg` | `female/caucasian` vs `male/caucasian` | yes |
| `pexels-moh-mckenzie-3597035.jpg` | `female/black` vs `male/black` | yes |
| `huu-chung-dang-...unsplash.jpg` | `female/asian` vs `female/indian` | yes |
| `julian-florez-...unsplash.jpg` | `female/hispanic` vs `female/mideastern` | yes |
| `kunal-goswami-...unsplash.jpg` | `female/indian` vs `female/mideastern` | yes |
| `raamin-ka-...unsplash.jpg` | `female/hispanic` vs `female/mideastern` | yes |
| `tobi-oshinnaike-...unsplash.jpg` | `male/black` vs `male/caucasian` | yes |
| `shivam-singh-2_X6NMP-E_U-unsplash.jpg` | `male/indian` vs `male/mideastern` | **no** |

7 of the 8 are unambiguously **the same photo filed under two labels** — a
duplication error, not a judgment call about which label fits the image
(the image is identical either way; only its label placement is
duplicated). The 8th, `shivam-singh-2_X6NMP-E_U-unsplash.jpg`, is a
coincidental filename collision between **two different photos** — see
Finding 6 above for why that one mattered for a dedup bug, not a labeling
one.

None of this determines *which* folder holds the "correct" label for the
7 true duplicates — that's still a judgment call for the maintainer, not
inferred here. Full paths: `reports/legacy_audit/label_collisions.json`.

**Resolved, 2026-07-26**: the maintainer reviewed all 7 and chose the
correct label for each. Baked into `LABEL_COLLISION_RESOLUTIONS` in
`scripts/data/build_v3_dataset.py`, keyed by filename (not path, since the
path is exactly what disagreed):

| File | Chosen label | Rejected label |
|---|---|---|
| `huu-chung-dang-lP02hkcp7H0-unsplash.jpg` | `female/asian` | `female/indian` |
| `julian-florez-l5rmMuK8070-unsplash.jpg` | `female/hispanic` | `female/mideastern` |
| `kunal-goswami-YHSohAq-PuI-unsplash.jpg` | `female/indian` | `female/mideastern` |
| `pexels-anna-shvets-4971982.jpg` | `male/caucasian` | `female/caucasian` |
| `pexels-moh-mckenzie-3597035.jpg` | `male/black` | `female/black` |
| `raamin-ka-4lQmQ_DBbNc-unsplash.jpg` | `female/mideastern` | `female/hispanic` |
| `tobi-oshinnaike-Z7MKNGFnbOw-unsplash.jpg` | `male/black` | `male/caucasian` |

`data/mebeauty_v3/images/metadata.parquet`'s `gender`/`ethnicity` columns
now reflect the chosen label for all 7; `has_label_collision` stays `true`
(the conflict happened and is worth knowing about) and a new
`label_collision_resolved` column records that it's been decided, not
just flagged. Rebuilt end-to-end and reverified: `COVERAGE.json` now
reports `"label_collisions_total": 7, "label_collisions_resolved": 7,
"label_collisions_remaining": 0`. The 8th entry
(`shivam-singh-2_X6NMP-E_U-unsplash.jpg`) is correctly absent from this
table — it was never a real collision (two different photos, different
SHA-256), so it needed no decision.

Along the way, a real bug surfaced and was fixed while pulling source
links for this review: `infer_provenance()`'s Unsplash-ID extraction split
on the *last* hyphen in the filename, truncating the id whenever the id
itself contained one (Unsplash ids are a fixed 11 characters and can
contain `-`/`_`). Confirmed against the full corpus: 195 of 1,325
Unsplash-pattern files (14.7%) had a truncated, wrong inferred URL (e.g.
`kunal-goswami-YHSohAq-PuI-unsplash.jpg` inferred as
`unsplash.com/photos/PuI` instead of the real
`unsplash.com/photos/YHSohAq-PuI`). Fixed in
`src/mebeauty_benchmark/legacy/provenance.py` (take the last 11 characters
of the filename body, not the last hyphen-delimited segment), covered by a
new regression test, and `reports/legacy_audit/image_provenance.csv`
regenerated (317 rows changed). This does not touch the redistribution-
rights question itself — inferred URLs are still unverified guesses, now
just correctly-formed ones.

## Finding 9 — rater identity: 831 unique MTurk Worker IDs, all pseudonymized and remapped consistently

`scripts/data/pseudonymize_raters.py` scanned all 6 `*_scores_all*.xlsx`
workbooks (wide-format, worker IDs as column headers) plus 37 individual
files under `scores/public_generic/` and `scores/public_date/`
(long-format, worker IDs as a `rater` column value) — 43 files, 831 unique
worker IDs total, each mapped to a stable `rater_XXXX` pseudonym across every
file. The script asserts no unmapped worker ID remains in any output file
before exiting. Outputs:

- Pseudonymized workbooks: `data/pseudonymized_scores/` (gitignored — still
  contains individual rating data, not committed even de-identified)
- Real mapping: `data/rater_mapping.LOCAL_ONLY.csv` (gitignored, never to be
  committed or uploaded)
- Counts-only summary (no worker IDs): `reports/legacy_audit/pseudonymization_report.json`

## Finding 10 — the legacy repo does provide baselines; what it lacks is a reproducible benchmark table

The published MEBeauty paper evaluates multiple CNNs and layer-wise
transfer-learning approaches — "no baselines" would be inaccurate. What is
actually missing from the legacy repo: exact splits tied to reported numbers,
fixed seeds, environment/dependency pins, configs, and released checkpoints
for any of those runs. None of that is reconstructed by this pass.

## Finding 12 — 1 near-duplicate pair evades exact-hash duplicate detection (Finding 7)

Finding 7's duplicate detection is byte-exact (SHA-256): it cannot catch
the same photo, or the same photoshoot moment, saved twice at different
compression or size — common with stock photos. Perceptual hashing
(`imagehash.phash`, 16×16 = 256-bit) instead, run against the deduplicated
v3 image set:

```
uv run --with imagehash --with pillow python scripts/data/find_near_duplicate_images.py \
    --images data/mebeauty_v3/images --threshold 20 \
    --output reports/legacy_audit/near_duplicate_images.json
```

Found exactly **1 pair** at Hamming distance 16 (of 256 bits) —
`male/caucasian/man-945482_1920.jpg` and
`male/caucasian/christopher-campbell-i4OHxtxiMtk-unsplash.jpg`. Confirmed
by direct visual inspection: the same person, same shot, filed under two
completely different stock-platform naming conventions (a numeric Pixabay-
style name and an Unsplash photographer-credit name) — not byte-identical,
so Finding 7 never saw it. The threshold (20) isn't arbitrary: the
next-closest pair in the whole dataset was at distance 58, visually
confirmed to be two different people with similar styling (long blonde
hair, similar framing) — there's a wide, clean gap between a genuine match
and stylistic coincidence in this dataset, and 20 sits in it.

Both images are rated, both in `train`, so this specific pair is not
train/test leakage — but it is redundant: two samples of essentially the
same content with different scores (7.01 vs. 6.31), which inflates train's
effective size by one and is itself a small data point about rater
variance on identical content. Flagged in `metadata.parquet` as
`has_near_duplicate` / `near_duplicate_image_ids`, not removed — the same
"report, don't silently resolve" policy as Finding 8's label collisions.

## Finding 13 — 3 images depict named, recognizable public figures, formerly actively rated in the canonical train split — now excluded

**Different in kind from every other finding here: not a data-quality bug,
a content/consent issue with real legal and reputational stakes.**
Everything else in this audit assumed the dataset's images are anonymous
stock photos of unnamed models — the license/provenance questions in
Findings 2 and the Open section were scoped on that assumption. Two passes
found three unambiguous cases (each visually confirmed, not just
filename-inferred):

- `female/black/michelle-obama-1129160_1920.jpg` — a real, unmistakable
  photograph of Michelle Obama, former U.S. First Lady.
- `female/indian/deepika-padukone-2779557_1920.jpg` — a stylized digital
  painting clearly intended to depict the actress Deepika Padukone.
- `female/indian/aditi-rao-hydari-1748439_1920.jpg` — a stylized digital
  painting, same artistic style as the Padukone one (plausibly the same
  source), depicting the actress Aditi Rao Hydari. Found on a **second**
  screening pass, broader than the first.

All three were actively rated and in the canonical `train` split (scores
5.89, 8.78, 8.89) — not historical artifacts like the celebrity-suggestive
filenames found removed from an earlier split generation (Finding 6's "not
found" rows, e.g. `charlize-theron-669608_1280.jpg`, no longer present in
this snapshot). Rating a named public figure's likeness for
"attractiveness" and distributing that as research data is materially
different from an anonymous stock photo, independent of the image's
stock-license status — publicity rights, reputational concerns, and
consent-for-this-specific-use are separate questions a generic Unsplash/
Pexels/Pixabay license does not resolve.

**Acted on, not just flagged: all three are now excluded from
`data/mebeauty_v3/`** (`EXCLUDED_LEGACY_PATHS` in
`scripts/data/build_v3_dataset.py`) — 2,547 → 2,495 unique images,
`train` 1,749 → 1,746. `data/mebeauty_v2/` (and the legacy snapshot) are
untouched; this is a v3-only exclusion, same as everything else built into
the restructure.

**The screening's own history is the strongest evidence of its limits.**
The first pass matched exactly two hyphenated words before a numeric ID —
it caught Obama and Padukone, and reported "none of the other ~55
candidates recognized" as a real but incomplete result. A second,
deliberately broader pass (three-or-more-word names, still Pixabay-only,
excluding Unsplash/Pexels photographer-credit patterns) immediately found
Hydari — a case the first pass's own regex structurally could not have
caught, not a matter of the reviewer missing something within scope. Two
things follow: **whatever the true count of public figures in this dataset
is, this audit has not found all of it** (a same-artist painting series
found two of three known cases — the third member of that series, if the
filename doesn't happen to contain a recognizable name, would not be
caught by *any* filename-based method), and any future screening pass
needs to be visual/perceptual (e.g. face-matching against a reference set),
not filename-pattern matching, to have real confidence.

**Recommendation, acted on for what could be acted on:** the three found
and confirmed cases are excluded. A perceptual/visual identity screening
pass across the full dataset — not another filename regex — is the next
step, and is not something this audit can complete without either an
external face-matching resource or your explicit direction on how far to
take it.

**Inferred source-URL check (manual, 2026-07-26):** each of the three
images' Pixabay-pattern filename produces an inferred source URL
(`reports/legacy_audit/image_provenance.csv`, `confidence=inferred`, built
as `https://pixabay.com/photos/{filename-stem}/`, never previously
verified). Automated verification from this environment is not possible —
`curl` and `WebFetch` both get HTTP 403 from Pixabay on all three URLs
uniformly, which is bot-protection, not a signal about which pages are
real. You checked manually, in a browser: the Deepika Padukone URL
(`https://pixabay.com/photos/deepika-padukone-2779557_1920/`) opens; the
Michelle Obama and Aditi Rao Hydari URLs do not. That's a real, if partial,
result — one inferred URL is human-confirmed live, two are human-confirmed
not to resolve as constructed. The failure mode was captured, and it's the
same for both: neither 404s outright — Pixabay instead falls back to
treating the whole `{filename-stem}` slug as a *search query*, meaning no
photo page exists at that exact URL/ID any more. The Michelle Obama URL
redirects to a search results page ("886 Free photos of
Michelle-Obama-1129160_1920"); the Aditi Rao Hydari URL redirects to a
search with zero results ("Sorry, we couldn't find any matches"). That
search-fallback behavior is consistent with the specific photo having been
taken down from Pixabay (deleted, or ID reassigned) rather than the
inferred URL pattern itself being malformed — the one URL that *does*
resolve (Deepika Padukone) uses the exact same `{filename-stem}` pattern,
so the pattern isn't the variable here. This is the first case in this
audit where an *inferred* provenance link has been checked against the
live platform at all; a 1-of-3 live rate is consistent with the standing
caveat (Finding 11 / Open section) that inferred URLs are filename-pattern
guesses, not confirmed links, for the other ~50+ Pixabay-pattern images in
this dataset too — and is a concrete reason to expect a nontrivial
fraction of those to also be dead links now, not just unverified ones.

## Finding 14 — rater scoring quality: measured per-rater, deliberately not filtered

**Revised 2026-07-26.** The original version of this finding used the wrong
statistic and reached a conclusion the evidence does not support. Both the
metric and the recommended action changed; the history is kept here because
the reasoning is the point.

### What the first version said (superseded)

Raters with 10+ ratings were checked for **straight-lining** (near-zero
score variance) and **extreme scoring** (mean within 0.5 of the 1–10 scale's
ends), flagging 9 of 831 (1.25% of ratings) as suspicious — 4 straight-liners
and 6 extreme scorers, of which `rater_0606` (940 ratings, mean 1.16) and
`rater_0696` (332 ratings, mean 1.32) dominated by volume.

### Why "extreme mean" is the wrong test

An extreme mean is a *scale preference*; the actual quality question is
whether a rater's scores carry *signal*. A genuinely harsh rater still
discriminates — they rate consensus-good faces above consensus-bad ones,
just compressed low. The informative statistic is therefore **discrimination
spread**: the rater's mean score on images the rest of the pool put in its
top consensus third, minus their mean on the bottom third, with the
consensus computed leave-one-out so no rater helps define the yardstick
they are measured against. Unlike a correlation, this survives range
restriction, so a compressed-but-engaged rater still scores clearly positive.

Measured against this dataset, the old test fails in both directions:

| Rater | n | mean | spread | Old verdict | Actual |
|---|---|---|---|---|---|
| `rater_0262` | 37 | 9.68 | **+0.92** | flagged extreme | engaged, merely generous |
| `rater_0549` | 39 | 9.69 | **+0.69** | flagged extreme | engaged, merely generous |
| `rater_0606` | 940 | 1.16 | **−0.16** | flagged extreme | no signal |
| `rater_0696` | 332 | 1.32 | **−0.09** | flagged extreme | no signal |
| `rater_0405` | 310 | 3.81 | **−2.90** | *not flagged* | strongly non-discriminating |
| `rater_0365` | 180 | 3.09 | **−2.68** | *not flagged* | strongly non-discriminating |
| `rater_0213` | 49 | 5.67 | **−8.71** | *not flagged* | fully inverted ordering |

Two false positives (raters who discriminate fine and simply score high),
and — more seriously — several high-volume raters with strongly negative
discrimination that the mean-based test could never catch, because their
means look perfectly ordinary. For contrast, normal high-volume raters sit
at spread +3.8 to +4.0.

### Why no raters are excluded

The obvious next step — exclude non-discriminating raters — was tested and
rejected on the evidence:

1. **No threshold is principled.** Across the 528 raters with ≥30 ratings,
   discrimination spread is a smooth continuum from −8.71 to +7.56 (median
   +2.33) with no bimodal gap. Every cutoff is arbitrary, and the cost swings
   several-fold across equally defensible ones: spread ≤0 removes 52 raters
   (6.90% of ratings), ≤0.5 removes 69 (11.76%), ≤1.0 removes 102 (18.82%).
2. **The impact is not marginal.** Applied honestly to *all* raters rather
   than only to the previously-flagged few, excluding spread ≤0 moves 576
   images' mean by >0.25 (max 1.10); at ≤1.0 it moves 1,119 images — roughly
   45% of the dataset — by >0.25, max 2.17. That is a wholesale rewrite of
   ground truth, not a cleanup.
3. **The rule is circular.** Filtering by agreement-with-consensus defines
   "good rater" as "agrees with the majority", inflates apparent inter-rater
   reliability, and deletes minority aesthetic viewpoints. For a *multi-ethnic
   beauty* dataset whose stated research goals include studying rater
   variation and subgroup bias, that would remove exactly the signal the
   dataset exists to study. `rater_0213` at −8.71 is a coherent inverted
   preference — information, not noise.
4. **Excluding is irreversible; reporting is not.** Shipping the statistics
   lets any consumer apply their own rule in one line.

Selectively applying the rule to only the 9 originally-flagged raters — the
version first proposed — would have been worse than either option: it is the
cherry-picking the rule was meant to avoid, and it would have missed the
larger non-discriminating raters entirely.

### What ships instead

- `data/mebeauty_v3/ratings/by_rater/rater_quality.parquet` — all 831 raters
  with `n_ratings`, `mean_score`, `std_score`, `n_with_consensus`,
  `discrimination_spread`, `loo_correlation`. Discrimination is null for the
  303 raters under the 30-rating minimum (too few per consensus tercile to
  measure), `std_score` null for 58 single-rating raters.
- `reports/legacy_audit/rater_quality_report.json` — the distribution, the
  20 lowest-discrimination raters, the 4 straight-liners (std < 0.01 over
  11–23 different images, which remains a defensible red flag on its own
  terms), and the full sensitivity/impact tables behind the numbers above.
- **The canonical aggregate is the unweighted mean of all ratings**, with no
  rater excluded. Reproducible, uncontestable, and comparable to the legacy
  release.

## Finding 15 — 5 filenames were truncated to 32 characters, silently costing 5 rated images

Five images are stored under a filename whose stem is cut to exactly 32
characters, while the score files kept the full name:

| Stored file | Name in the score files |
|---|---|
| `male/black/payton-tuttle-n_RdRxH_7h4-unspla.jpg` | `...-unsplash.jpg` |
| `female/indian/shifaaz-shamoon-MqLy-G-dBi8-unsp.jpg` | `...-unsplash.jpg` |
| `female/caucasian/creating-a-brand-WxAOdbDpMiU-uns.jpg` | `...-unsplash.jpg` |
| `female/mideastern/jonathan-borba-5rQG1mib90I-unspl.jpg` | `...-unsplash (1).jpg` |
| `male/indian/puvvukonvict-photography-AtpSEe3.jpg` | `...-AtpSEe3yoIg-unsplash.jpg` |

Every stem is exactly 32 characters, and four of the five end mid-word in a
partial `-unsplash` suffix (`-uns`, `-unsp`, `-unspl`, `-unspla`) — a
truncation at file-creation time, not a naming choice. Because
`resolve_to_existing_images` matched on the full name, all five rows were
counted as `not_found` and dropped: **five images that were actually rated
were excluded from the canonical split for a filename bug.**

Recovered by `_resolve_truncated_basename` in
`src/mebeauty_benchmark/legacy/splits.py`, which only resolves when the
answer is unambiguous — exactly one existing file may match the 32-character
truncation, **and** no file under the full untruncated name may exist. That
second guard matters here: `jonathan-borba-5rQG1mib90I-unspl.jpg` and
`jonathan-borba-5rQG1mib90I-unsplash.jpg` are **two different photographs**
(SHA-256 `ae39f547…` vs `b59e95c8…`). The full-named file takes the
`...-unsplash.jpg` rating directly; the truncated file receives the separate
`...-unsplash (1).jpg` rating, whose own 32-character truncation is exactly
`-unspl.jpg`. Without the guard, one photo's score would have been assigned
to the other.

Effect: `not_found` drops fell 22 → 18, canonical train 1,746 → **1,751**,
and images with no canonical rating fell 10 → 5. Both decline-cases are
covered by regression tests.

## Finding 16 — the per-rater table pooled two different rating tasks into one score column

**This is the most consequential defect found in this review**, and it was
introduced by this repository's own reconciliation, not inherited.

The legacy collection ran **two distinct rating tasks** over the same images:
a *generic* attractiveness question and a *date* attractiveness question.
Their score distributions sit about a full point apart:

| Source | Rows | Mean |
|---|---|---|
| `public_generic/*.xlsx` + `generic_scores_all.xlsx` | 60,046 | **6.00** |
| `public_date/*.xlsx` | 63,131 | **5.01** |

`build_ratings_by_rater.py` loaded both, then discarded the `source` column
and wrote a single undifferentiated `score`. Two consequences:

1. **Two different questions were merged into one label.** Anyone using
   `ratings_by_rater.parquet` for personalization research was training on a
   mixture of "how attractive is this face" and "would you date this person"
   with no way to tell them apart.
2. **Genuine ratings were destroyed as false duplicates.** Deduplication
   keyed on `(image_id, rater)` alone, so a rater who answered *both*
   questions about an image had one answer silently dropped — and which one
   survived depended on the order sources happened to be loaded. **7,555
   real ratings** were lost this way (115,622 → **123,177** after the fix).

The pooling also put the per-rater table on a different scale from the
canonical labels. Measured against the canonical aggregate:

| Per-rater subset | Bias vs. canonical | Mean abs. diff | Correlation |
|---|---|---|---|
| pooled (old behaviour) | **+0.514** | 0.626 | 0.894 |
| **generic only** | **−0.057** | **0.267** | **0.966** |
| date only | +1.095 | 1.218 | 0.718 |

So the canonical score is the **generic** rating, and the old pooled table
was systematically offset from the very labels it was supposed to explain.

**Fixed**: `ratings_by_rater.parquet` now carries a `rating_type` column
(`generic` / `date`), and deduplication keys on
`(image_id, rater_id, rating_type)`. Rater-quality statistics (Finding 14)
compute their leave-one-out consensus **within** a rating task — a pooled
consensus scored every date rating as "below consensus" and every generic
one as "above" it for reasons unrelated to the rater, which alone had
inflated the non-discriminating count from 42 to 52.

**Consumers wanting the per-rater equivalent of the canonical label should
filter `rating_type == "generic"`.** The `date` ratings are a genuine second
annotation layer, not noise — they are simply a different question.

## Finding 17 — `original_images/` is 99.2% *not* original: they are pre-cropped, downsampled face images

Found while sampling the unresolved-provenance images. Measured across all
2,495 v3 images:

| Dimensions | Count |
|---|---|
| 500×500 | 1,302 |
| 400×400 | 1,095 |
| 600×600 | 75 |
| everything else | 23 |

**2,472 of 2,495 (99.1%) are exactly 400×400, 500×500, or 600×600, and only
20 images (0.8%) exceed one megapixel.** Visual inspection confirms these are
tight, centred face crops, not resized full photographs.

The naming makes this actively misleading. `male/caucasian/male-4572748_1920.jpg`
carries Pixabay's `_1920` size marker — the source was a 1920px image — but
the stored file is a **400×400 face crop**. The filename describes what was
downloaded; the file is what survived an undocumented preprocessing step. The
directory name `original_images/` compounds it.

**Consequences that matter:**

1. **"Re-crop from the originals with a better detector" is not available.**
   Any modern detector run against this tree is cropping an existing crop at
   400–600px, not re-deriving a crop from source pixels. This weakens the
   premise behind `docs/REPRODUCE_LEGACY_BASELINE.md` and is the most likely
   explanation for why regenerated MTCNN crops never matched the legacy ones
   bit-for-bit (mean difference 43.9, 0 of 2,401 within 1.0) — the legacy
   crops were probably produced from true source images that are not in this
   release at all.
2. **Effective resolution is heterogeneous.** A 400×400 and a 600×600 face
   crop carry materially different detail, and the split between them is not
   random (see below). Any model trained across the whole set sees mixed
   input quality.
3. **The paper's "in-the-wild, unconstrained" framing describes the
   photographs, not the shipped files**, which are uniformly cropped and
   centred. That is a meaningful difference for anyone reasoning about pose
   and background variability from the distributed data.

**The 600×600 images are a distinct collection batch**, and the correlation
is total:

| Size | Has source link | No source link |
|---|---|---|
| 400×400 | 1,093 | 2 |
| 500×500 | 1,294 | 8 |
| **600×600** | **0** | **75** |
| other | 7 | 16 |

**Every one of the 75 600×600 images lacks provenance, and no image with
provenance is 600×600.** Combined with their filename shapes (bare numbers
like `21.jpg`, short codes like `g_4.jpg`), this is strong evidence of a
separate collection episode with its own preprocessing size and its own
naming convention — one that did not preserve source filenames. It explains
*why* provenance is unrecoverable for that batch: the information was never
kept, rather than lost later.

### Full breakdown of the 101 unresolved images

The size table above accounts for 85 (2 + 8 + 75); the remaining **16** sit
in the "other" bucket and are reported explicitly here, because they are the
most surprising group in the dataset:

| Dimensions | MP | Path |
|---|---|---|
| 5304×6630 | 35.17 | `male/indian/128.jpg` |
| 6240×4160 | 25.96 | `female/mideastern/d10.jpg` |
| 4000×6000 | 24.00 | `male/caucasian/119.jpg` |
| 6200×3600 | 22.32 | `male/caucasian/guy9.jpg` |
| 3456×5184 | 17.92 | `male/hispanic/121.jpg` |
| 3413×5119 | 17.47 | `male/caucasian/117.jpg` |
| 3448×4592 | 15.83 | `female/indian/d14.jpg` |
| 3620×3999 | 14.48 | `male/caucasian/111.jpg` |
| 2339×3508 | 8.21 | `male/caucasian/126.jpg` |
| 2827×2826 | 7.99 | `female/mideastern/50.jpg` |
| 2028×3000 | 6.08 | `female/indian/59.jpg` |
| 2157×2157 | 4.65 | `male/caucasian/g_11.jpg` |
| 2071×2071 | 4.29 | `male/asian/guy1.jpg` |
| 1731×1731 | 3.00 | `male/hispanic/guy8.jpg` |
| 832×832 | 0.69 | `male/indian/guy3.jpg` |
| 688×688 | 0.47 | `male/indian/guy7.jpg` |

**14 of the 20 full-size images in the entire dataset have no provenance.**
The images that *are* genuine uncropped photographs are overwhelmingly the
ones with no recoverable source — the inverse of what one would expect if
provenance loss were random. Together with the 600×600 batch this points to
at least two collection episodes that did not preserve source filenames,
one of which contributed unprocessed originals.

### Recorded in metadata, not repaired

`images/metadata.parquet` now carries `width`, `height`, `megapixels`,
`crop_batch` (`uniform_400` / `uniform_500` / `uniform_600` / `full_image` /
`other`), and `is_preprocessed_crop`:

| `crop_batch` | Count | Has source link | No source link |
|---|---|---|---|
| `uniform_500` | 1,302 | 1,294 | 8 |
| `uniform_400` | 1,095 | 1,093 | 2 |
| `uniform_600` | 75 | 0 | **75** |
| `full_image` | 20 | 6 | **14** |
| `other` | 3 | 1 | 2 |

**Files are shipped byte-unchanged — deliberately not resampled to a common
size.** Upscaling a 400×400 crop to 600×600 invents detail that was never
captured; downscaling the larger ones discards real detail. Consumers resize
at training time regardless, so the correct action is to hand them accurate
dimensions and a batch label, not a uniform-looking set that hides the
heterogeneity.

Nothing here can restore source pixels that were never released. This is
documented so that no one plans a preprocessing comparison on the assumption
that untouched originals are available.

### Resolved by configuration, not by overwriting

Rather than resizing the native files (which would permanently discard
resolution that cannot be recovered -- the source photographs are not in the
release, and the upstream repository is no longer under this project's
control), v3 now ships **two configurations** over the same 2,495 images,
joined by `image_id`:

| Config | Contents |
|---|---|
| `native` (default) | The legacy files byte-unchanged, heterogeneous 400-600px plus 20 full-size |
| `standardized_256` | Uniform 256x256 RGB, aspect-preserving Lanczos, centre-padded, JPEG q95 |

Built by `scripts/data/standardize_images.py`; geometry in
`src/mebeauty_benchmark/legacy/standardize.py` with 10 unit tests. Verified
against the built directory rather than assumed:

- all 2,495 outputs are exactly 256x256 and RGB;
- **byte-identical across two full runs** (0 checksum differences), with
  `preprocessing_version` pinning Pillow's version, resampler, quality and
  subsampling, since both the resampler and the JPEG encoder can shift
  between releases;
- 2,477 square images received **zero** padding; the 18 non-square preserved
  aspect ratio to within 0.24%;
- every recorded `standardized_sha256` matches the file on disk;
- both `load_dataset("data/mebeauty_v3", name="native")` and
  `name="standardized_256"` resolve and return 2,495 rows at 400x400 and
  256x256 respectively -- checked because this exact area previously produced
  silent empty phantom splits.

**Landmarks are transformed, not regenerated**: `new_x = old_x * scale +
pad_left`. Confirmed that the transform introduces **no new** out-of-bounds
points (0 images where an in-frame landmark left the frame). The 106 native
landmark sets that already extended past the frame stay outside,
proportionally -- clamping them would silently alter those images' geometry
and make the two configurations encode different face shapes rather than the
same shape at two scales. `has_out_of_bounds_landmarks` flags them in both.

256 rather than 512: nothing in the release is large enough for 512 to add
real detail, and 256 is what torchvision's standard pipeline resizes to
before a 224 centre crop. JPEG q95 rather than lossless PNG: the *resize* is
the dominant lossy step, so lossless encoding would roughly quadruple the
download (73MB -> ~325MB) to preserve detail the resample already removed.
Native remains authoritative either way.

## Finding 18 — perceptual public-figure screening: works, but only as triage

Finding 13's screening was filename-based and provably incomplete — it cannot
see an image whose filename carries no name, and 101 images have no filename
signal at all. `scripts/data/screen_public_figures.py` screens the *pixels*
instead, using `facenet-pytorch`'s InceptionResnetV1 classification head over
VGGFace2's 8,631 identities (a dataset built by image-searching celebrities,
so its classes are overwhelmingly public figures). Maximum softmax
probability is used as a **ranking signal for human review** — it names
nobody, because the class-index-to-name mapping is not published with the
weights.

**Validated against the three confirmed cases rather than assumed**, and the
first attempt was wrong. Feeding the model raw resized crops performed close
to chance; the VGGFace2 weights expect MTCNN-aligned faces, and adding that
step changed the result substantially:

| Confirmed figure | Rank, unaligned | Rank, MTCNN-aligned |
|---|---|---|
| Aditi Rao Hydari | 347 (top 13.6%) | **3 (top 0.1%)** |
| Michelle Obama | 573 (top 22.5%) | **255 (top 10.0%)** |
| Deepika Padukone | 1,055 (top 41.4%) | **478 (top 18.7%)** |

Catching all three requires reviewing the top **478 of 2,546** — a roughly
**5x reduction** in human review effort versus looking at everything. That is
genuinely useful triage, and it is not a detector.

**Limits, stated because they bound what this can support:**

- **n = 3.** The "top 19% catches everything" figure rests on three known
  positives, two of which are stylized digital paintings rather than
  photographs. The error bars are enormous.
- **VGGFace2's 8,631 identities are a small fraction of all public figures**,
  and skew toward English-language media. A regionally famous person absent
  from that set scores low regardless.
- **High confidence is not identification.** The top two ranked images are the
  same photograph of an unidentified male model under two filenames
  (byte-identical, correctly collapsed in v3 — they appear twice only because
  the screen runs over the legacy snapshot's 2,547 paths).

**The visual pass was then actually done, ranks 1-480** (contact sheets of
the aligned ranking, reviewed in five batches). Result:

- All three known positives were independently spotted at their predicted
  ranks -- Aditi Rao Hydari at 3, Michelle Obama at 255, Deepika Padukone at
  478. That is end-to-end confirmation the pipeline surfaces real positives,
  not just that the numbers looked plausible.
- **No additional public figures were recognised** in those 480 images. The
  remainder read as stock and model photography.
- One false positive worth noting: ranks 1 and 2 are the same photograph of
  an unidentified male model under two filenames. They are byte-identical and
  already collapsed in v3; they appear twice only because the screen runs
  over the legacy snapshot's 2,547 paths, not v3's 2,495 images.

**This is a real but bounded result.** 480 of 2,546 images were reviewed --
the range that provably contains all known positives -- and nothing new
surfaced. What it does *not* establish: the reviewer's recognition skews
heavily toward internationally famous people, so a regionally prominent
person would likely pass unnoticed; and 2,066 lower-ranked images were not
viewed at all.

**Conclusion**: this replaces "review 2,547 images" with "review the top few
hundred", which makes a visual pass tractable -- and that pass has now been
run, finding nothing beyond the three already excluded. It does not license a
claim that the dataset is free of public figures, and no such claim should be
made. Reports: `reports/legacy_audit/public_figure_screen{,_aligned}.json`.

## Finding 19 — roughly 1.7% of images contain a second face

MTCNN with `keep_all=True` over a 300-image random sample found **5 images
(1.7%, extrapolating to ~43 of 2,547) with two or more faces detected above
0.95 confidence** — e.g. `female/mideastern/hamid-tajik-QXbJ3yhMNK4-unsplash.jpg`,
which shows two women together and ranked 4th in Finding 18's screen.

This creates an ambiguity the dataset does not record: **when an image
contains two faces, which one does the rating describe, and which one did the
crop pipeline select?** The legacy pipeline kept a single crop per file with
no record of which detection it chose, so a rating for a two-person photo
cannot be attributed to a specific face. The gender/ethnicity label has the
same problem.

Reported, not resolved. Quantifying it exactly (rather than from a 300-image
sample) and deciding whether such images should be excluded or re-cropped is
a judgement call, and it interacts with Finding 17: these are already crops,
so "re-crop the correct face" is not available from the released pixels.

## Recovery summary

After the recovery work in Findings 3–4, coverage across the full 2,547
`original_images/` is:

| Artifact | Original coverage | After recovery |
|---|---|---|
| MTCNN crop / FaceNet embedding | 2,402 (94.3%) | **2,547 (100%)** |
| MTCNN crop / FaceNet embedding — `male/indian` only | 8/151 (5.3%) | **151/151 (100%)** |
| OpenCV crop | 2,351 (92.3%) | 2,539 (99.7%) |
| 68-point landmarks | 2,445 (96.0%) | **2,547 (100%)** |

`images.parquet` carries this as separate `has_*` (original pipeline) and
`*_recovered` (this session) boolean columns rather than merging them, so
which pipeline produced which artifact is always recoverable from the table
itself.

A consolidated, complete-coverage local copy assembled from all of the
above (legacy + recovered artifacts merged into one tree, orphaned
artifacts excluded) is built by `scripts/data/assemble_v2_dataset.py` into
`data/mebeauty_v2/` (gitignored, local only — see that directory's own
generated `README.md` and `COVERAGE.json`).

## Canonical metadata tables

`scripts/data/build_canonical_dataset.py` joins every finding above into
three artifacts under `reports/legacy_audit/canonical_dataset/`:

- `images.parquet` (2,547 rows) — one row per `original_images/` file: path,
  gender, ethnicity, size, SHA-256, which derived artifacts exist for it
  (MTCNN/OpenCV crop, FaceNet embedding, landmarks), inferred provenance,
  content-duplicate group ID, and label-collision flag.
- `ratings.parquet` (2,488 rows) — the canonical train/val/test ratings from
  Finding 6.
- `checksums.sha256` — plain `sha256sum`-format manifest for
  `original_images/`.

This is metadata only (paths, hashes, boolean flags, scores) — no pixel data,
no rater identifiers — and is the basis for the Hugging Face dataset card
draft in `docs/DATASET_CARD.md`.

## Cross-checks that found no issue

- **Image integrity**: all 2,547 files under `original_images/` open and
  verify cleanly with Pillow — no corrupt or zero-byte files.
- **Rating sanity**: scores in the canonical splits fall in a consistent
  ~1–9.6 range with similar mean (~6.0–6.1) across train/val/test — no
  encoding errors or out-of-range values.

## Open — needs your decision (explicitly out of scope for this pass)

- **Finding 13 — 2 named public figures (Michelle Obama, Deepika Padukone)
  are actively rated in the canonical train split.** The most urgent item
  in this list, and different in kind from the rest — see Finding 13 above
  for full detail. Recommendation, not a neutral report: exclude both and
  do a more thorough identity screening pass before any release.
- **Licensing — release model decided, exact agreement wording still
  open.** A fully open license (CC0, CC BY, MIT) is rejected for the whole
  dataset: it can't express the restrictions this data needs, and CC
  licenses only cover rights the licensor actually owns or controls, not
  third-party source photographs. Decided instead: a **gated release** —
  public dataset page, individual access requests, a custom "MEBeauty
  Research Use Agreement" (non-commercial research only, no redistribution
  or sublicensing, no commercial use, no face-recognition/surveillance/
  profiling use, mandatory takedown compliance, per-user approval,
  citation required). Component licensing: newly written benchmark code
  under Apache-2.0 (where ownership is confirmed); the HF dataset
  repository metadata as `license: other`; photographs/crops/ratings/
  landmarks under the custom agreement; third-party content remains
  subject to its original platform's terms. **Gating controls downstream
  use, it does not create redistribution rights** — images with
  unverified, unknown, or prohibited source permissions (Finding 11) stay
  withheld from the distributed package regardless of gating. The HF
  repository stays **private** until: (a) the actual agreement text is
  reviewed by a qualified lawyer (GDPR, biometric/demographic data,
  international access), and (b) image-level rights and exclusions
  (Finding 11, Finding 13) are resolved, not just the release model
  chosen.

  **Access-control vs. contract, and why both may be needed:** HF gating
  (username/email/optional questions) is access control, not by itself a
  dedicated e-signature workflow — but a click-wrap acceptance is not
  automatically unenforceable either; it depends on whether terms were
  clearly presented, the user actively agreed, and reliable evidence of
  the exact terms and acceptance was retained. Under EU eIDAS, an
  electronic signature can't be denied legal effect merely for being
  electronic. Planned formulation: **HF gating for identity-linked access
  control; where stronger contractual assurance is warranted, access is
  approved only after a separate Research Use Agreement is executed
  through an appropriate e-signature process** (e.g. DocuSign) —
  retaining agreement version, signer identity, approval date, and HF
  username per user, with each agreement revision archived so it's
  provable which terms a given user accepted. The lawyer review above
  should specifically cover: whether ordinary click-wrap is sufficient
  here or an advanced/qualified e-signature is warranted; who the
  contracting party is (you personally, a university, another
  institution); governing law and dispute jurisdiction; institutional vs.
  individual signatories; enforcement of deletion/non-redistribution/
  prohibited-use terms; and how long signed agreements and access records
  should be retained. None of this is decided yet — it's the specific
  question set for that review, not a substitute for it.

  Redistribution rights per source platform (Finding 11 below)
  remain unverified.
- **Finding 11 — image provenance is inferred, not verified.**
  `scripts/data/infer_image_provenance.py` tags all 2,547 images: first by
  filename pattern (1,342 unsplash, 782 pixabay, 306 pexels, 117 unknown),
  then a second pass resolves 40 more of those "unknown" rows by
  cross-referencing Finding 7's content-duplicate groups — e.g.
  `female/asian/30.jpg` inherits `unsplash`/the photo ID from its
  byte-identical duplicate `asian-girl-4819726_1920.jpg`, tagged
  `inferred-via-duplicate` rather than `inferred` so the source of the
  signal (a content match, not the filename) stays visible
  (`reports/legacy_audit/image_provenance.csv`).

  **Copy-marker bug, found and fixed 2026-07-26.** A trailing `(1)` / ` (2)`
  download copy-marker defeated every pattern above, because the stem no
  longer ended in `-unsplash` (etc.). **14 images were tagged "no provenance"
  purely for this reason** — including the obviously-Unsplash
  `gift-habeshaw-KBv5dEN3QtY-unsplash(1).jpg`. The marker is now stripped
  before matching, with regression tests. Unresolved count: 117 legacy paths
  → 115 v3 images (two byte-identical pairs collapsed) → **101 after the
  fix**. Anyone planning to exclude the unresolved set should apply this fix
  first, or discard 14 images whose source is perfectly knowable.

  **101 images remain genuinely unresolvable**, and every automated route has
  been tried and failed — `scripts/data/recover_unknown_provenance.py`:

  | Route | Result |
  |---|---|
  | Filename pattern | No signal. These filenames were **manually renamed at collection time**, not truncated — 69 are a bare number (`21.jpg`), 12 letter+number (`f1.jpg`), 14 a short code (`g_4.jpg`). There is nothing to undo: the source information was replaced, not shortened. |
  | Exact-duplicate inheritance | Already applied upstream; no more available |
  | Embedded EXIF / PNG text metadata | **0 of 101** carry any author, copyright, or description field |
  | Perceptual near-duplicate match | **0 of 101** within the validated threshold (≤20). Closest match across the whole set is distance **60**, median 92 — well past the 58 at which Finding 12 confirmed two *different* people. Portraits of different faces routinely land in the 60–90 range, so accepting these would invent provenance, not recover it. |

  **Reverse image search is the only remaining route, and it cannot be done
  from here** — it requires uploading each file to a search service, which
  needs a tool that can POST binary data, and these images are not published
  at any URL a search engine could be pointed at. It is a manual,
  browser-based task for a human, at roughly 101 lookups.

  ### Photo IDs are verified correct; the generated URLs are not canonical

  First external verification of any inferred provenance in this project
  (2026-07-26). Two Unsplash photo IDs inferred from filenames —
  `gisFZKWpKQ4` (`dorrell-tibbs-...`) and `uHIN59WI_zc` (`marsha-dhita-...`)
  — were confirmed to appear verbatim in the real Unsplash URLs. This
  validates the extraction logic, including the 11-character fix for IDs
  containing internal hyphens.

  It also revealed that **Unsplash's canonical URL is
  `/photos/{description-slug}-{photo-id}`, not the bare `/photos/{photo-id}`
  this repository generates** — e.g.
  `unsplash.com/photos/man-in-black-crew-neck-shirt-gisFZKWpKQ4`. The slug is
  Unsplash-generated alt text and is **not derivable from the filename**, so
  the canonical URL cannot be constructed offline. Whether the bare-ID form
  redirects could not be tested from here: Unsplash returns a 401
  bot-protection challenge to `curl` (notably *not* a 404 — the path was
  preserved in the challenge's redirect parameter).

  **Treat `inferred_photo_id`, not `inferred_source_url`, as the durable
  identifier.** The ID is verified, stable, and resolvable on-platform; the
  slug is descriptive text the platform can change, and the URL column is a
  best-effort convenience.

  A useful side effect: those slugs describe the subject, which allowed an
  independent label cross-check. Both slugs read "man"; both images are
  labelled `male` in `metadata.parquet` — external confirmation of the
  gender labels from a source unconnected to this dataset's labelling.

  Everything that *does* have a link is still `inferred, unverified` — useful
  for future consent/takedown resolution, not a rights determination. Link
  quality is also **not uniform**: Unsplash (1,332) and Pexels (299) encode
  the platform's real, precise identifier, while Pixabay (763) URLs are
  reconstructed from the whole filename stem — and Pixabay is where 2 of 3
  hand-checked links were dead (Finding 13).
- **GDPR / biometric data classification.** Whether face embeddings and
  landmarks constitute special-category biometric data depends on processing
  and intended use, and needs legal review, not a categorical claim here. No
  data controller has been named. **Correction to an earlier framing in this
  project's own discussion**: embeddings are not lower-risk than the source
  images just because they're not pixels — FaceNet-512 embeddings are
  designed specifically to encode stable, matchable facial characteristics,
  and may support re-identification against another face database; they may
  be *more* sensitive than the photo alone, not less. Pseudonymizing rater
  IDs, dropping filenames, or storing derived features instead of raw images
  reduces some risk but does not by itself make embeddings, landmarks, or
  personalized (per-rater) ratings "anonymous data" under GDPR — the
  underlying regulation treats pseudonymization as risk-reduction, not
  anonymization. **Every derived artifact stays gated under the same terms
  as the source images by default** — original images, crops, ratings
  (aggregate and personalized), landmarks, geometric features, FaceNet
  embeddings, demographic labels, and the filename/source-URL provenance
  mapping. None of these should be assumed safe to release openly later
  just because they're "derived" rather than raw pixels; each would need
  its own legal sign-off, not a blanket assumption (e.g. aggregate ratings
  still describe identifiable people once linked back to a photo; the
  ratings table's *selection and arrangement* may separately carry
  copyright/EU sui generis database rights, distinct from any privacy
  question about its contents).
- **Hugging Face release.** Not started. Requires the licensing and rights
  review above first.
- **Preprocessing pipeline choice.** No comparison has been run between
  original images, legacy OpenCV crops, legacy MTCNN crops, or a modern
  detector/alignment pipeline (e.g. RetinaFace). Nothing has been selected as
  canonical.
- **Benchmark performance claims.** No literature review or cross-dataset
  experiment has been run; no plateau or degradation numbers should be
  cited until one has.
- **Historical protocol identification.** Which of the three split
  generations (see Finding 6) actually produced the published paper's
  results is still unverified. Now further complicated by Finding 2: the
  paper's own stated composition (1,250 male / 1,300 female) doesn't match
  any current gender breakdown in this snapshot, so even finding a split
  generation whose *count* matches 2,550 would not prove it matches the
  paper's actual image set. Resolving this needs the paper's full
  methodology section (not just its abstract) and possibly correspondence
  with the original authors about what changed since publication.
- **Repo layout for release, once gating is actually implemented.**
  Planned, not built: keep public benchmark code (this repo) physically
  separate from gated data, e.g. `MEBeauty-Benchmark` (public, Apache-2.0
  code) / `MEBeauty` (gated, images/ratings/landmarks/embeddings/
  provenance under the Research Use Agreement) / a possible future
  `MEBeauty-Open-Annotations` (public, only fields a lawyer confirms are
  safe — not assumed to ever exist). One directory doesn't legally force
  one license, but separate repos reduce the chance of a gated file ending
  up in a public clone by accident.
- **A "re-fetch images from source URLs instead of redistributing files"
  script is not a licensing workaround**, if it's ever considered as a way
  to dodge redistribution-rights questions. Automated re-downloading from
  Unsplash/Pexels/Pixabay would still need those platforms' terms checked
  for that specific use, links can go dead (as already seen — 2 of 3
  checked Pixabay links in Finding 13 are dead), and it doesn't resolve
  the underlying rights question, just relocates it. Not implemented, not
  planned as a workaround.

## Reproducing this document

Every number above comes from one of:

```
uv run python scripts/data/copy_legacy_snapshot.py --source ~/Research/MEBeauty-Legacy --dest data/legacy_snapshot
uv run python scripts/data/audit_legacy_inventory.py --root data/legacy_snapshot --output reports/legacy_audit/inventory
uv run python scripts/data/find_label_collisions.py --root data/legacy_snapshot/original_images --output reports/legacy_audit/label_collisions.json
uv run python scripts/data/find_duplicate_images.py --root data/legacy_snapshot/original_images --output reports/legacy_audit/duplicate_images.json
uv run python scripts/data/infer_image_provenance.py --legacy-copy data/legacy_snapshot --output reports/legacy_audit/image_provenance.csv
uv run python scripts/data/build_canonical_splits.py --legacy-copy data/legacy_snapshot --output reports/legacy_audit/canonical_splits --report-out reports/legacy_audit/split_dedup_report.json
uv run python scripts/data/regenerate_geometric_features.py --legacy-copy data/legacy_snapshot --output data/geometric_features
uv run python scripts/data/pseudonymize_raters.py --legacy-copy data/legacy_snapshot --output data/pseudonymized_scores --mapping-out data/rater_mapping.LOCAL_ONLY.csv --report-out reports/legacy_audit/pseudonymization_report.json
uv run python scripts/data/build_canonical_dataset.py --legacy-copy data/legacy_snapshot --manifest data/legacy_snapshot.manifest.csv --reports reports/legacy_audit --output reports/legacy_audit/canonical_dataset
```

Plus the recovery and assembly scripts (Findings 3–4, 6) that need extra
dependencies deliberately kept out of `pyproject.toml` (heavy, one-off use):

```
uv run --with facenet-pytorch --with opencv-python python scripts/data/recover_missing_crops.py \
    --legacy-copy data/legacy_snapshot --output data/recovered_crops --report-out reports/legacy_audit/crop_recovery_report.json
uv run --with "opencv-python<5" --with facenet-pytorch python scripts/data/improve_opencv_recovery.py \
    --legacy-copy data/legacy_snapshot --crop-report reports/legacy_audit/crop_recovery_report.json --output data/recovered_crops/opencv
uv run --with face-alignment python scripts/data/recover_missing_landmarks.py \
    --legacy-copy data/legacy_snapshot --output data/recovered_landmarks --report-out reports/legacy_audit/landmark_recovery_report.json
uv run python scripts/data/assemble_v2_dataset.py \
    --legacy-copy data/legacy_snapshot --recovered-crops data/recovered_crops --recovered-landmarks data/recovered_landmarks \
    --geometric-features data/geometric_features --reports reports/legacy_audit --output data/mebeauty_v2
```

Plus the v3 restructure (`docs/RESTRUCTURE_PROPOSAL.md`) and Finding 12's
near-duplicate check, which needs two passes since the check scans
`images/` produced by the first build:

```
uv run python scripts/data/build_v3_dataset.py \
    --legacy-copy data/legacy_snapshot --v2 data/mebeauty_v2 \
    --images-parquet reports/legacy_audit/canonical_dataset/images.parquet --output data/mebeauty_v3
uv run --with imagehash --with pillow python scripts/data/find_near_duplicate_images.py \
    --images data/mebeauty_v3/images --threshold 20 --output reports/legacy_audit/near_duplicate_images.json
uv run python scripts/data/build_v3_dataset.py \
    --legacy-copy data/legacy_snapshot --v2 data/mebeauty_v2 \
    --images-parquet reports/legacy_audit/canonical_dataset/images.parquet \
    --near-duplicates reports/legacy_audit/near_duplicate_images.json --output data/mebeauty_v3
uv run python scripts/data/build_ratings_by_rater.py \
    --pseudonymized data/pseudonymized_scores --v3-metadata data/mebeauty_v3/images/metadata.parquet \
    --output data/mebeauty_v3/ratings/by_rater
```

Plus everything from the "do the remaining improvements" pass (Findings 13
resolution, 14; the other two split generations; full-scale reproduction
verification; Croissant metadata):

```
# Finding 13's 3 exclusions are baked into build_v3_dataset.py itself
# (EXCLUDED_LEGACY_PATHS) -- the three commands above already produce the
# post-exclusion counts, nothing extra to run for that.

# Finding 14: per-rater quality (reports + ships the table; excludes nobody)
uv run python scripts/data/check_rater_quality.py \
    --ratings data/mebeauty_v3/ratings/by_rater/ratings_by_rater.parquet \
    --output reports/legacy_audit/rater_quality_report.json \
    --table-out data/mebeauty_v3/ratings/by_rater/rater_quality.parquet

# The other two split generations, same rigor as the canonical one
uv run python scripts/data/build_canonical_splits.py --legacy-copy data/legacy_snapshot --generation original \
    --output reports/legacy_audit/canonical_splits_original --report-out reports/legacy_audit/split_dedup_report_original.json
uv run python scripts/data/build_canonical_splits.py --legacy-copy data/legacy_snapshot --generation crop \
    --output reports/legacy_audit/canonical_splits_crop --report-out reports/legacy_audit/split_dedup_report_crop.json

# Full-scale (2,547 image) reproduction verification, not the original 3-image spot check
uv run --with facenet-pytorch python scripts/data/verify_crop_reproduction.py \
    --legacy-copy data/legacy_snapshot --v2-crops data/mebeauty_v2/crops/mtcnn \
    --crop-recovery-report reports/legacy_audit/crop_recovery_report.json \
    --output reports/legacy_audit/crop_reproduction_verification.json

# Croissant metadata (local generation + validation, no HF upload)
uv run --with mlcroissant python scripts/data/build_croissant_metadata.py \
    --v3 data/mebeauty_v3 --output data/mebeauty_v3/croissant.json
```

plus the ad hoc `find`/`wc -l`/`grep` commands quoted inline above (all
against `data/legacy_snapshot/`, never against `~/Research/MEBeauty-Legacy`
directly).
