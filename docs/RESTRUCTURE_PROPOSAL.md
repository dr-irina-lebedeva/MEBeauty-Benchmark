# Dataset Restructure Proposal

Written in response to a direct question: what issues does the dataset
still have, and — having looked at how a comparable, well-regarded dataset
(SCUT-FBP5500) and modern practice actually structure this kind of
release — what should MEBeauty's structure be, and what's left to do to
get there.

**Update: implemented.** After review and explicit confirmation, this was
built — `data/mebeauty_v3/` (local, gitignored, not uploaded or licensed).
Results below in each relevant section; Part 4's status table has the
current state of every item.

## Part 1 — Issues the dataset still has

Full detail and repro commands: `docs/DATASET_AUDIT.md`. Consolidated list:

**Fixed this session (context, not action items):**
- Missing MTCNN/OpenCV crops, missing landmarks — recovered (100% / 96% /
  100%).
- A 94.7%-in-one-subgroup (`male/indian`) MTCNN failure — a real bias bug
  in the legacy pipeline, now fixed.
- Split leakage (filename + content-duplicate) — fixed.
- 33 canonical rating rows that didn't join to any real image (a parser
  bug + relabeled/removed images) — fixed.
- `Inf` values in derived geometric features — fixed (converted to `NaN`,
  flagged).
- Exposed rater IDs — pseudonymized.

**Still open, not fixable without you or external input:**
1. **Licensing.** No license chosen for code, images, annotations,
   embeddings. Nothing can be publicly released until this is resolved.
2. **Redistribution rights.** Source platform is inferred from filenames
   only (1,342 unsplash / 782 pixabay / 306 pexels / 117 unresolvable),
   never verified against the platforms themselves.
3. **GDPR / biometric classification.** Whether embeddings/landmarks count
   as special-category biometric data depends on intended use — needs
   legal review, not a default assumption.
4. **8 label collisions**, 7 of which are the same photo duplicated across
   two ethnicity/gender folders. Which folder holds the correct label is
   a human judgment call — deliberately not auto-resolved.
5. **Historical split identification.** Now better evidenced (the
   "original" generation has 252 relabeled rows vs. 7 for "2022," pointing
   at "2022" being a later, cleaned-up revision) but not confirmed against
   the paper's actual methodology section — the abstract alone doesn't
   settle it, and the paper's stated composition (1,250M/1,300F) doesn't
   match this snapshot (1,196M/1,351F) at all.
6. **Per-rater scores were unreconciled to canonical image identity — now
   fixed.** The pseudonymized per-rater workbooks
   (`data/pseudonymized_scores/`, 586 rater columns) key images by **bare
   basename only** (`"112.jpg"`, no folder). Checked directly against v3's
   metadata before joining (rather than assuming it'd be safe): only 1
   basename in the entire dataset is genuinely ambiguous (the same
   `shivam-singh...` case from Finding 6), so basename-joining turned out
   to be safe with that one explicit exception. Built
   `ratings/by_rater/ratings_by_rater.parquet`: 115,700 individual ratings,
   2,489 of 2,498 images covered (99.6%), 831 raters — the data
   **personalized prediction** (named as MEBeauty's core differentiator in
   the project README) actually needs. Two more legacy data problems
   surfaced while building this: `date_scores_all(_2022).xlsx` have
   corrupted column headers (not usable, excluded), and several other
   score files turned out to be redundant subsets of each other (excluded
   to avoid double-counting a rating). Full detail in
   `data/mebeauty_v3/ratings/by_rater/reconciliation_report.json`.
7. **Preprocessing pipeline choice** for any future canonical release
   (MTCNN vs. a modern detector) is still unselected — see Part 2, which
   argues this may be a smaller problem than it looked, because a modern
   release may not need to ship a fixed crop at all.

## Part 2 — What modern practice actually looks like

Checked directly against SCUT-FBP5500 (`HCIILAB/SCUT-FBP5500-Database-Release`,
the standard cross-dataset comparison point already in this project's
plan): it ships **raw images + a landmark file (PTS format, 86 points) +
plain-text label/split files.** Nothing else. No precomputed face crops,
no precomputed embeddings, no hand-crafted feature vectors. Its README
explicitly frames resize/crop as a *training-time* step, not a property of
the shipped files: *"Each raw RGB image is resized as 256×256, and then a
227×227 random crop of raw image is obtained to feed into AlexNet."*
"Raw image" is their own word for what's distributed. Its published
baselines (AlexNet, ResNet-18, ResNeXt-50) train end-to-end directly on
the images.

That's also the general shape of modern released datasets on Hugging Face
and elsewhere: ship the source data plus the minimum labels needed to
reproduce results, and ship **preprocessing code**, not **preprocessing
output**. Reasons this matters here specifically, not just as a style
preference:

- `crops/mtcnn/` and `crops/opencv/` are **100% reproducible** from
  `original_images/` plus a documented script (which now exists:
  `recover_missing_crops.py`). Shipping the pixels doubles/triples dataset
  size for zero information gain over shipping the code.
- `embeddings/facenet_512/` bakes in one specific, dated (2015-era)
  embedding model. Shipping it as core data quietly biases downstream
  research toward that model and away from anything newer (ArcFace,
  modern CLIP/DINO-family face embeddings, whatever exists by the time
  this is used). Anyone who wants FaceNet-512 specifically can compute it
  in minutes from the images.
- `geometric_features.npz` (11,628-dim hand-crafted ratio features, ~115MB)
  is a specific, narrow, largely superseded methodology from the original
  paper's non-CNN baseline. It has real value for reproducing *that
  specific historical result*, but it does not belong in a dataset's core
  identity any more than one paper's hand-crafted SIFT features would.
- `landmarks.csv` is the one derived artifact worth keeping as core data:
  it's small, format-standard, doesn't tie the dataset to a specific
  downstream model, and SCUT-FBP5500 itself ships the equivalent.

**A second, more important structural problem, caught on re-reading my own
first draft of Part 3 below: it kept `images/{gender}/{ethnicity}/<file>` —
the same folder-encodes-label scheme the current dataset already uses.**
That's not a neutral choice carried forward from habit; it's the direct
mechanism behind two of this session's largest findings:

- The 8 label collisions (Finding 8) exist *because* a duplicate photo can
  be physically filed in two folders at once — the bug is structurally
  possible only because "label" means "which directory this file sits in."
- The 252-relabeled-row gap between the "original" and "2022" split
  generations (Finding 6) means relabeling an image required someone to
  physically move a file between folders, apparently inconsistently, and
  nothing enforced that a rating's path and the image's current folder
  ever stayed in sync. That's exactly the failure `resolve_to_existing_images`
  had to be built to paper over after the fact.

A structure that keeps folder-encoded labels reproduces the precondition
for both bugs, just with today's data cleaned up. Hugging Face's `datasets`
library already has an idiomatic answer for exactly this case — a flat
image directory plus a `metadata.parquet`/`metadata.csv` sitting alongside
it, with arbitrary columns (this is the documented, supported alternative
to folder-inferred labels, not a workaround). It's also a better fit here
specifically because MEBeauty has *two* orthogonal labels (gender,
ethnicity), which single-level folder-as-label schemes were never designed
for anyway.

**Revised recommendation: flat `images/`, filenames as content-addressed
IDs, labels only in `metadata.parquet`.** Use each image's SHA-256 (already
computed, already the join key for duplicate detection) as its stable
filename/ID instead of its original, legacy-inherited name. This also
kills a second class of bug this session hit repeatedly: several original
filenames are hostile to naive parsing (embedded spaces, parentheses, and
at least one visibly mangled encoding — `s-jDpjN4U5mxE-unsplash - ╕▒▒╛.jpg`)
— the quoted-path parser bug (Finding 6) exists only because a filename
had a space in it. A content-addressed flat namespace can't collide (two
different photos can no longer accidentally share a name, the root cause
of the `shivam-singh...` case), can't be relabeled by moving a file (there
is no gender/ethnicity-bearing path to move it out of), and needs no
filename-quoting logic anywhere downstream. Original filenames are kept as
a `legacy_filename` metadata column for provenance, not dropped.

## Part 3 — Structure (as built)

```text
data/mebeauty_v3/                           # local, gitignored, not uploaded
  images/
    <sha256>.<ext>                          # 2,498 unique images (2,547 legacy paths, content-deduped)
    metadata.parquet                        # image_id (=sha256), file_name, legacy_filename, legacy_path,
                                             #   other_legacy_paths, gender, ethnicity, size_bytes,
                                             #   has_label_collision, label_collision_alternatives,
                                             #   has_near_duplicate, near_duplicate_image_ids (Finding 12),
                                             #   inferred_platform/photo_id/url, provenance_confidence
  landmarks.parquet                         # image_id, landmarks (native list<float>, 136 values) — 2,498 rows
  ratings/
    aggregate/{train,val,test}.parquet      # image_id, score — 1,749 / 222 / 517 (2,488 total)
    by_rater/ratings_by_rater.parquet       # image_id, rater_id, score — 115,700 rows, 2,489 images, 831 raters
    by_rater/reconciliation_report.json     # source selection, what was dropped and why
  reproduce_legacy_baseline/README.md       # pointer to docs/REPRODUCE_LEGACY_BASELINE.md
  README.md, COVERAGE.json                  # generated summary
```

Built by `scripts/data/build_v3_dataset.py` (images/metadata/landmarks/
aggregate ratings) and `scripts/data/build_ratings_by_rater.py` (per-rater
table, run separately since it has its own source-selection logic — see
its docstring and `reconciliation_report.json` for exactly which of the
legacy score files were used, skipped as redundant, or skipped as
corrupted). `crops/`, `embeddings/`, and `geometric_features.npz` are not
in this structure at all — see `docs/REPRODUCE_LEGACY_BASELINE.md` to
regenerate them on demand. Disk size: 116MB vs. 355MB for
`data/mebeauty_v2/` — the "smaller" claim below is measured, not assumed.

**`metadata.parquet` lives inside `images/`, not at the dataset root** —
checked directly against Hugging Face's current documentation (not
memory), which requires a `file_name` column for `ImageFolder` to
auto-load a dataset, and confirmed empirically that placement matters: with
metadata at the dataset root, `ImageFolder`'s split-detection logic gets
confused by the sibling `ratings/aggregate/{train,val,test}.parquet` files
elsewhere in the tree and silently produces empty phantom splits instead of
the real 2,498 images. Moving metadata into `images/` (a sibling of the
image files only) fixed it —
verified with `load_dataset("imagefolder", data_dir="data/mebeauty_v3/images")`
against the actual built directory, not a synthetic example: all 2,498
images load correctly.

Confirmed by actually building it, not just claimed: 2,547 → 2,498 unique
images is exactly the 49-fewer predicted by Finding 7's 49 duplicate
groups; label collisions dropped from 8 to 7 exactly as predicted (the 8th,
`shivam-singh...`, isn't a collision under content-addressing — different
content, different `image_id`).

## Part 4 — Status

| # | Item | Status |
|---|---|---|
| 1 | Confirm scope cut | **Done** — confirmed, implemented |
| 2 | Per-rater ratings table | **Done** — 115,700 rows, 2,489/2,498 images (99.6%), 831 raters. Found two more legacy data problems along the way: `date_scores_all(_2022).xlsx` have corrupted column headers (excluded, documented in the script); `generic_scores_all_2022.xlsx` and `public_generic_all.xlsx`/`public_date_all.xlsx` are redundant subsets of other files (excluded to avoid double-counting) |
| 3 | Label collisions (content-addressing) | **Done** — 7 of 8 collapsed automatically. The remaining human call (which label the 7 collapsed rows should carry) is unchanged and still open |
| 4 | Licensing / redistribution rights | Not solvable by me |
| 5 | Historical split identification | Not solvable by me (needs paper's full text) |
| 6 | Migration mechanics | **Done** |
| 7 | Verify reproduction claim before relying on it | **Done, with a correction to the original claim.** Spot-checked: `facenet_pytorch.MTCNN` reproduces *this session's own recovered crops* exactly (0.0 pixel diff, deterministic) but does **not** bit-for-bit reproduce the *original 2021-era legacy crops* (55.6 mean pixel diff — different underlying library, DeepFace's TensorFlow MTCNN vs. facenet-pytorch). Documented honestly in `docs/REPRODUCE_LEGACY_BASELINE.md` rather than left as an unverified assumption: these scripts give methodologically-equivalent, not pixel-identical, reproduction of the historical pipeline |
| 8 | Fresh review against current HF documentation, verified not assumed | **Done, found and fixed two real gaps + one structural bug.** `metadata.parquet` was missing the `file_name` column HF's `ImageFolder` requires (checked their current docs directly). Landmarks were a comma-joined string instead of a native `list<float>` column, unlike the nested-list convention HF's own docs use for structured fields. Actually running `load_dataset("imagefolder", ...)` against the built directory (not assuming it would work) surfaced a real bug: with `metadata.parquet` at the dataset root, sibling `ratings/aggregate/{train,val,test}.parquet` files confuse `ImageFolder`'s split auto-detection into empty phantom splits. Fixed by moving `metadata.parquet` inside `images/`, isolated from everything else — re-verified against the real directory afterward: all 2,498 images now load correctly with one line of code |
| 9 | Near-duplicate detection beyond byte-exact matching (Finding 12) | **Done.** Perceptual hashing found 1 pair Finding 7's SHA-256 check couldn't (same photo, different compression, filed under two stock-platform naming conventions) — confirmed by direct visual inspection, along with confirming the next-closest pair (58 vs. 16 Hamming distance) is genuinely two different people, not a duplicate, so the threshold isn't arbitrary. Both images are in `train`, so no leakage; flagged in `metadata.parquet`, not removed |
