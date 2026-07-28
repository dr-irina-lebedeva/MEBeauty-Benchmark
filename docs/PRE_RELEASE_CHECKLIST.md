# Pre-release checklist — before MEBeauty v3 goes to Hugging Face

Ordered by **what can invalidate everything else**, not by what is easiest.
Phase 0 depends on other people and is slow, so it starts first even though
it feels least urgent. Phases 2–3 are mechanical once the decisions land.

Status legend: ☐ not started · ◐ partly done · ☑ done and verified

---

## Phase 0 — Depends on other people (start today; weeks, not hours)

These are the only items that can invalidate the entire release. Everything
else is recoverable work. Do not wait for Phase 1 to start these.

### ☐ 0.1 Co-author agreement

MEBeauty has co-authors (Guo, Ying). This release restructures the dataset,
changes its terms, and puts you in the position of approving individual
access requests. Get written confirmation that you may maintain and
administer research access to the dataset.

This does not resolve third-party photo rights — it establishes your
authority over the ratings, structure, and access process. Without it, the
Research Use Agreement's contracting party is unsupported.

### ☐ 0.2 Rater consent review for the per-rater table

**This is new exposure the original release did not have.** The original
published aggregate scores. v3 additionally ships
`ratings/by_rater/ratings_by_rater.parquet` — 141,736 individual ratings
attributable to 860 pseudonymous raters, from which per-rater behavioural
patterns (mean, variance, discrimination) are directly derivable, and are in
fact shipped in `rater_quality.parquet`.

Check what the original AMT task's consent text actually said. Publishing
per-worker behavioural data is a different question from publishing
aggregates, even pseudonymised. If the consent does not cover it, the
options are: ship aggregates only, ship the per-rater table with coarser
pseudonymisation, or seek fresh consent.

### ◐ 0.3 Legal review — *you have decided to proceed without one*

Recorded as your decision (`docs/PROJECT_STATUS.md`). The unresolved items
it would have covered: GDPR/biometric classification of embeddings and
landmarks, naming a data controller, enforceability of the access agreement,
and retention of access records. Noted here so the decision stays visible
rather than forgotten, not to reopen it.

---

## Phase 1 — Decisions only you can make

Each changes what gets built, so all four should land before Phase 2 starts.

### ☐ 1.1 The 101 images with no source link — keep or exclude?

**101 of 2,495 (4.0%)** have no provenance signal at all. (Was 115 until a copy-marker bug was fixed on 2026-07-26, which recovered 14 — apply that fix before excluding anything.) Every automated
recovery route is exhausted and verified failed
(`reports/legacy_audit/unknown_provenance_recovery.json`): no filename
signal, no exact-duplicate inheritance, **0 of 101** carry EXIF/PNG
metadata, **0 of 101** match a known-provenance image within the validated
perceptual threshold (closest is distance 60 against a threshold of 20).

Remaining option is manual reverse image search, ~101 browser lookups. These filenames were manually renamed at collection time (69 are a bare number like `21.jpg`), so nothing can be undone -- the source info was replaced, not shortened.
Calibration: of 3 inferred links opened by hand so far, only 1 still
resolved — expect a meaningful fraction to be unrecoverable regardless of
effort.

Decide: (a) manual pass, then exclude whatever is still unresolved,
(b) exclude all 101 now (costs 2,495 -> 2,394 images, 2,490 -> 2,391 rated rows; subgroup balance shifts by <=0.002, so it is clean), or (c) ship them flagged as unknown-provenance.
**If you exclude, counts, splits, and subgroup balance all shift** — see 2.1.

### ☐ 1.2 What identity-screening standard is sufficient?

3 named public figures were found and excluded. The screening was
filename-based, and is **demonstrably incomplete**: the second pass found a
case the first pass's own regex could not structurally have caught. Treat 3
as a floor.

A perceptual screen now exists (`scripts/data/screen_public_figures.py`,
Finding 18) and is validated: MTCNN-aligned, it ranks all three confirmed
figures inside the top 19%, cutting review effort ~5x. **It is triage, not a
detector** — it names nobody, and its validation rests on n=3.

Decide: review the top ~250-500 ranked images visually (a few hours, and the
most defensible option available), or accept filename screening as
sufficient for a gated research release. Either way the dataset card must
keep saying the screen is incomplete — no method available here supports
claiming otherwise.

### ☐ 1.3 Ship both rating tasks, or generic only?

Found 2026-07-26 (Finding 16). The collection ran **two different
questions**: *generic* attractiveness (68,974 ratings, mean 5.86) and *date*
attractiveness (72,762 ratings, mean 4.80). **All aggregate labels are
computed from `generic` only**; `date` never enters a label.

Decide: ship both with `rating_type` (current state — richest, but users
must be told to filter), or ship generic only (simpler, discards a genuine
annotation layer). If both, the dataset card must lead with the distinction,
because silently averaging them reproduces exactly the bug that was just
fixed.

### ☐ 1.4 Define the takedown process

The terms require takedown compliance, but no mechanism exists. Specify:
who receives requests, the contact route, target response time, and how
removal propagates to users who already downloaded. `image_provenance.csv`
supports resolving a request to a single photograph — the process around it
is undefined.

---

## Phase 2 — Build and verify (mechanical, once Phase 1 lands)

### ☐ 2.1 Apply Phase 1 decisions and rebuild end-to-end

Any exclusion cascades through the whole pipeline. Full sequence:

```
build_canonical_splits.py → assemble_v2_dataset.py → build_v3_dataset.py
  → build_ratings_by_rater.py → check_rater_quality.py
  → build_croissant_metadata.py
```

Then re-verify split balance across score, gender, and ethnicity — excluding
images can quietly skew subgroups, and val/black is already the smallest
cell at 20.

### ☐ 2.2 Write the Research Use Agreement text

Terms are decided (non-commercial research only, no redistribution, no
face-recognition/surveillance use, takedown compliance, per-user approval,
citation). Contracting party and governing law are decided (Irina Lebedeva
as maintainer, for rights she owns or administers; Hungary/Budapest). A v0.1
draft exists in conversation but **is not saved to a file**.

Needs writing to `LICENSE` (Hugging Face expects custom license text there)
plus a liability section, which the draft deliberately left as a placeholder.

### ☑ 2.3 Data-quality verification — done, re-run after any rebuild

Already verified on the current build:

- ☑ Content-addressing invariant — all 2,495 files re-hashed, filenames match content exactly
- ☑ Referential integrity — metadata ↔ images ↔ landmarks ↔ splits ↔ by_rater all consistent
- ☑ No duplicate leakage across splits; no metadata nulls
- ☑ Split balance — score/gender/ethnicity near-identical across train/val/test
- ☑ No raw AMT Worker IDs anywhere; all IDs `rater_NNNN`; mapping file gitignored
- ☑ No absolute paths in shipped artifacts *or* tracked reports (fixed 2026-07-26)
- ☑ `load_dataset("imagefolder", ...)` loads all 2,495
- ☑ Croissant metadata validates (0 errors)
- ☑ `make check` clean, 46/46

### ☐ 2.4 Update documentation to final state

- Version/tag the split as **`benchmark-v1` (maintained v3 split)** — explicitly *not* the original paper split, which remains unidentified.
- Remove or update "nothing has been published anywhere" from `DATASET_CARD.md` and `CHANGES_VS_ORIGINAL.md` — those lines become false on upload.
- Replace the Croissant license placeholder (`LICENSE-NOT-YET-DECIDED-...`) with the real value.
- Final counts and exclusion list reviewed against `COVERAGE.json`.

---

## Phase 3 — Publish

### ☐ 3.1 Commit the work

**Everything from this audit is uncommitted working-tree state on
`chore/legacy-inventory`.** This is the practical first blocker for
anything else. Commit, open a PR, merge.

### ☐ 3.2 Upload as a **private** repo first

Push to Hugging Face private. Verify in the real environment, not locally:
Dataset Viewer renders, `load_dataset` works from the Hub, HF's own
auto-generated Croissant matches expectations, and the card renders.

### ☐ 3.3 Enable gating, then make public

Configure gated access with the agreement text. Test the request→approve
flow end-to-end with a second account before opening it.

### ☐ 3.4 Tag `v3.0.0`

Tag the GitHub repo to match the Hub release so the code that built the
dataset is pinned to the dataset itself.

---

## The three things most likely to go wrong

1. **Phase 0 is started too late.** Co-author sign-off and consent review
   depend on other people's response times and can invalidate everything
   downstream. They are listed first for that reason.
2. **Exclusions are decided after the build is "final."** Any exclusion
   restarts Phase 2. Decide 1.1 and 1.2 before rebuilding.
3. **The dataset ships without the `rating_type` warning prominent.**
   Averaging generic and date ratings together reproduces a defect that was
   already found and fixed once — it is an easy mistake for a downstream
   user to repeat.
