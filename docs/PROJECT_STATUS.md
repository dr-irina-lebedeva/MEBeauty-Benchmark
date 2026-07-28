# Project Status

## Current phase

Project infrastructure setup.

## Completed

- GitHub repository created
- Hugging Face dataset repository created
- README and dataset card drafted
- uv environment configured
- Agent instructions, tests, Ruff, pre-commit, and CI added
- Legacy repository cloned and frozen at commit `7b849562ee92d99d34d56afa2ebe85e5075f9b20` (`docs/LEGACY_SOURCE.md`)
- Dataset audit (files, counts, splits, raters) reproduced against a checksummed
  snapshot and written up in `docs/DATASET_AUDIT.md` — see that document for
  the full findings list and its "Open — needs your decision" section
- Deeper duplicate-image audit added: content-hash (not just filename)
  deduplication found 18 previously-undetected train/val/test leaks and 49
  total duplicate-image groups (Findings 6–7); canonical split corrected to
  1,766/224/521
- Canonical metadata tables built (`reports/legacy_audit/canonical_dataset/`:
  `images.parquet`, `ratings.parquet`, `checksums.sha256`) — the schema a
  Hugging Face release would be built from
- Draft `docs/DATASET_CARD.md` and `docs/DATASHEET.md` written (both marked
  unreleased/draft; license and rights review still open)
- Fixed a stray artifact in `README.md`'s citation block and added
  `CITATION.cff`
- Recovered the two silent-failure gaps from Findings 3–4: 145/145 missing
  MTCNN crops + FaceNet embeddings (100%), 189/197 missing OpenCV crops
  (96%, after a verified second tuning pass — 8 remain genuinely
  undetectable), and 102/102 missing landmark rows (100%) + their geometric
  features. Corrected the landmark-gap count itself from 88 to 102 (14
  duplicate rows in the legacy `landmarks.csv` don't add coverage).
  Recovered artifacts are gitignored under `data/recovered_*/`, kept
  separate from the legacy pipeline's own outputs; `images.parquet` now has
  both `has_*` and `*_recovered` columns so provenance stays explicit.
- Found and fixed a real bug in `dedupe_across_splits`: it keyed on bare
  basename, which could conflate two genuinely different photos that share
  a filename across folders (confirmed for one real image,
  `shivam-singh-2_X6NMP-E_U-unsplash.jpg`). Now keys on the full normalized
  path. Didn't corrupt the persisted `benchmark-v1` canonical split, but did
  affect an earlier in-conversation reconciliation of the other two split
  generations — corrected in `docs/DATASET_AUDIT.md`, Finding 6.
- Cross-referenced Finding 7 (content duplicates) against Finding 8 (label
  collisions): 7 of 8 label collisions are the same photo filed under two
  labels (a duplication error); 1 (`shivam-singh...`) is two different
  photos coincidentally sharing a filename — the same case as the dedup bug
  above.
- Assembled `data/mebeauty_v2/` (gitignored, local only) — a single,
  consolidated directory merging legacy + recovered artifacts (crops,
  embeddings, landmarks, geometric features, canonical ratings), with
  orphaned legacy artifacts excluded. Built by
  `scripts/data/assemble_v2_dataset.py`.
- Found external corroboration for Finding 2: the published paper's
  abstract states "2550... 1250 men and 1300 women," but that gender split
  doesn't match this snapshot at all (1,196 male / 1,351 female) — the
  underlying image set has changed since publication, a new, still-open
  question (see Finding 2 and the "Historical protocol identification"
  open item in `docs/DATASET_AUDIT.md`).
- Found and fixed a real data-quality bug in the merged v2 dataset: 2 of
  the 102 recovered geometric-feature rows contained `inf` (division by
  zero from coincident landmark points on extreme profile photos) — now
  converted to `NaN` and flagged in `COVERAGE.json` instead of silently
  passed through. Verified all 2,547 FaceNet-512 embeddings directly:
  correct dimensionality, no NaN/Inf, no zero-norm vectors.
- **Major finding**: the legacy MTCNN pipeline's 5.7% aggregate failure
  rate was hiding a 94.7% failure rate isolated to a single subgroup,
  `male/indian` (143/151 missing, vs. ~0% everywhere else) — confirmed via
  the independent OpenCV backend (which processed that folder normally) to
  be a pipeline/environment bug, not a property of the images. Visually
  verified a sample of the recovered crops as genuine. This means the
  original `FaceNet_512_features` had only 5.3% coverage for that subgroup.
  Documented prominently in `docs/DATASET_AUDIT.md` (Finding 3) and
  `docs/DATASET_CARD.md`'s bias section.
- **Second major finding**: cross-checking every canonical rating row
  against the merged v2 crops found 33 of 2,511 didn't join to any image.
  Root causes: a path-quoting parser bug (4 rows, now fixed in
  `parse_split_lines`), 7 images relabeled to a different ethnicity/gender
  folder since rating (remapped via new `resolve_to_existing_images`), and
  22 images removed from the dataset entirely since rating (dropped).
  Applying the same check to the "original" split generation found **252
  relabeled rows out of 2,514 (10%)** vs. 7 (0.3%) for "2022" — strong
  evidence "2022" is a later, cleaned-up revision. Canonical split
  corrected again: **train 1,749 / val 222 / test 517 (2,488 total)**,
  every row now verified end-to-end to join to a real file.
- **Restructured the dataset** per `docs/RESTRUCTURE_PROPOSAL.md`, after
  confirmation: `data/mebeauty_v3/` — flat, content-addressed images
  (`<sha256>.<ext>`), labels only in `metadata.parquet` (no more
  folder-encoded gender/ethnicity, the mechanism behind the two findings
  above). Building it confirmed the proposal's predictions exactly: 2,547
  legacy paths → 2,498 unique images (49 fewer, matching Finding 7's 49
  duplicate groups), label collisions 8 → 7 (content-addressing collapses
  same-photo duplicates automatically). Also built the per-rater ratings
  table (115,700 ratings, 831 raters, 99.6% image coverage) needed for the
  personalization research goal, and verified — rather than assumed — the
  "regenerate crops on demand" premise: it reproduces this session's own
  recovered crops exactly, but not the original 2021 legacy crops
  bit-for-bit (different underlying library), now documented honestly in
  `docs/REPRODUCE_LEGACY_BASELINE.md`. Dropped `crops/`, `embeddings/`,
  and `geometric_features.npz` from the shipped structure (116MB vs. 355MB
  for `data/mebeauty_v2/`, which still exists as-is). Nothing in
  `data/mebeauty_v2/` was deleted.
- **Re-reviewed v3 against Hugging Face's current documentation** (fetched
  directly, not recalled) and fixed two real gaps plus one structural bug:
  `metadata.parquet` was missing the `file_name` column `ImageFolder`
  requires; landmarks were a comma-joined string instead of a native
  `list<float>` column. Actually running
  `load_dataset("imagefolder", data_dir=...)` against the built directory
  (rather than assuming it would work) surfaced a real bug — sibling
  `ratings/aggregate/{train,val,test}.parquet` files elsewhere in the tree
  confused `ImageFolder`'s split auto-detection into empty phantom splits.
  Fixed by moving `metadata.parquet` inside `images/`; re-verified against
  the real directory afterward — all 2,498 images now load correctly with
  one line of code.
- **Found a new class of duplicate beyond Finding 7's byte-exact check**:
  perceptual hashing (`scripts/data/find_near_duplicate_images.py`) found 1
  near-duplicate pair (same photo/moment, different compression, two
  different stock-platform filenames) that SHA-256 matching couldn't catch.
  Validated the detection threshold against the dataset's own distribution
  rather than picking one blind — confirmed the next-closest pair (58 vs.
  16 Hamming distance) is genuinely two different people. Flagged in
  `metadata.parquet` (`has_near_duplicate`), not removed. Documented as
  Finding 12.
- **Finding 13, most urgent open item**: 2 named public figures (Michelle
  Obama, Deepika Padukone) found actively rated in the canonical `train`
  split — a content/consent issue, not a data-quality bug, found by
  scanning filenames for name patterns and visually confirming each
  candidate. The screen is filename-based and incomplete by construction
  (can't rule out figures under generic names or ones not personally
  recognized) — flagged with a strong recommendation to exclude both and
  do a fuller screening pass, not silently resolved.
- **Executed the full "can still improve" list from `docs/FINAL_STATUS_2026-07-26.md`**,
  on request:
  - Finding 13 acted on, not just flagged: excluded from `data/mebeauty_v3/`
    (`EXCLUDED_LEGACY_PATHS` in `build_v3_dataset.py`). A **second, broader**
    screening pass then found a **third** public figure (Aditi Rao Hydari) —
    a case the *first* pass's own regex could not structurally have caught.
    Also excluded. Direct evidence the screening method has a real blind
    spot, documented as such rather than treated as now-complete.
  - Both other legacy split generations rebuilt with the same full rigor as
    the canonical one (existence resolution + both dedup passes) —
    `reports/legacy_audit/canonical_splits_{original,crop}/`, real files,
    not informal counts.
  - Per-rater quality check (Finding 14): 9 of 831 raters flagged
    (straight-lining or extreme scoring), 1.25% of all ratings.
  - Reproduction-verification claim re-checked at full scale (2,547 images,
    not 3): confirmed exactly, not just plausible — 145/145 recovered crops
    reproduce at 0.0 pixel diff, 0/2,401 legacy crops come close.
  - Croissant metadata generated and schema-validated locally
    (`data/mebeauty_v3/croissant.json`, `mlcroissant`) — license field is an
    explicit non-license placeholder string, not a real decision.

## Current work

- Prepare a validated Hugging Face dataset release
- Licensing review (still open — see `docs/DATASET_AUDIT.md`)

## Next steps

1. Resolve licensing and redistribution-rights questions in `docs/DATASET_AUDIT.md`
2. Decide how far to take Finding 13's identity screening (visual/perceptual,
   not filename-based — the demonstrated way to catch what's been missed)
3. Identify which legacy split generation produced the published paper's results
4. Compare preprocessing pipelines (original / legacy OpenCV / legacy MTCNN / a
   modern detector) before selecting a canonical variant
5. Once licensing clears, upload the candidate release from
   `data/mebeauty_v3/` to a gated Hugging Face repo and validate
   `load_dataset()` / the Dataset Viewer / the real (HF-generated) Croissant
   metadata against the local draft

## Session handoff

At the end of each session, update:

- completed work
- current branch and commit
- tests run
- unresolved questions
- exact next action

### 2026-07-25

Branch `chore/legacy-inventory`. Tests: `make check` clean, 35/35 passing.
Full narrative report, conclusions, and recommendations:
`docs/SESSION_REPORT_2026-07-25.md`. Unresolved: licensing, redistribution
rights, GDPR classification, historical split identification (see
`docs/DATASET_AUDIT.md`, "Open" section). Next action: licensing/rights
decision, or (if that's not ready) try to obtain the paper's methodology
section to settle the historical-split question, or start the
metadata-only SCUT-FBP5500 comparison.

### 2026-07-26

Branch `chore/legacy-inventory`. Tests: `make check` clean, 41/41 passing.
Recovered the two data gaps from Findings 3–4 (crops/embeddings 100% MTCNN,
96% OpenCV after a verified second pass; landmarks 100%; discovered along
the way that the MTCNN failures were 94.7% concentrated in one subgroup,
`male/indian` — a real bias finding, not a random failure rate), fixed a
real basename-vs-full-path bug in the dedup logic, cross-referenced label
collisions against content duplicates, assembled a consolidated local
`data/mebeauty_v2/`, found external (paper-abstract) corroboration for
Finding 2 that also surfaces a new open question (gender composition drift
since publication), fixed an `inf`-vs-`NaN` bug in the merged geometric
features, and — from actually cross-checking every rated row against the
merged crops — found and fixed 33 orphaned/mis-pathed canonical ratings
(quoting bug, relabeled images, removed images), landing the canonical
split at 1,749/222/517 (2,488), now verified end-to-end. All
recovery/assembly scripts need ephemeral extra deps (`uv run --with ...`,
documented in each script's docstring and in the audit doc), not added to
`pyproject.toml`. Unresolved items unchanged from 2026-07-25 — licensing is
still the actual blocker; none of this session's work touched it. Next
action: same as before, or the SCUT-FBP5500 metadata-only comparison.

**Later the same day**: found Finding 13 (2 named public figures actively
rated in the canonical split — see above), now the standing top priority,
ahead of licensing. Full final status, organized as "can still improve"
vs. "cannot solve without you": `docs/FINAL_STATUS_2026-07-26.md`.

**Later still, same day**: Finding 13 extended to 3 images (see
`docs/DATASET_AUDIT.md`); manually verified 2 of the 3 inferred Pixabay
source links are now dead (Pixabay search-fallback pages, not 404s) and 1
resolves — first live-platform check of any inferred provenance link in
this audit, supporting evidence for the standing "inferred, not verified"
caveat on the other ~50+ Pixabay-pattern images. **Licensing decision
made**: gated release, not an open license (CC0/CC BY/MIT rejected as a
poor fit given identifiable faces, unverified redistribution rights, and
confirmed public-figure content). Recorded in
`docs/DATASET_AUDIT.md` ("Open" section) and `docs/DATASET_CARD.md`
(frontmatter + Licensing section). Still open: the actual Research Use
Agreement wording needs a qualified lawyer before anything is public, and
redistribution rights per source photo are still unverified regardless of
gating — unresolved images stay withheld from the distributed package even
once gating exists. HF repo stays private until that legal review and
Finding 13's fuller identity-screening question are both resolved.

**Refined further**: HF's own gating (username/email/questions) is access
control, not automatically either "sufficient contract" or "worthless
click-wrap" — enforceability turns on presentation, active agreement, and
evidence retained, and EU eIDAS specifically protects e-signatures from
being dismissed just for being electronic. Planned mechanism: HF gating
for access control, backed by a separately executed Research Use Agreement
(e.g. via DocuSign) when stronger contractual assurance is warranted,
archived per agreement version/signer/approval date/HF username. Added as
specific lawyer-review questions (click-wrap sufficiency, e-signature
tier, contracting party, governing law, signatory type, enforcement,
retention period) in `docs/DATASET_AUDIT.md` and `docs/DATASET_CARD.md` —
not resolved, just scoped.

**Corrected once more**: an earlier framing in this project's own
discussion implied embeddings/derived features were lower-risk than the
source photos ("de-identified"). Corrected in `docs/DATASET_AUDIT.md`
(GDPR bullet) and `docs/DATASET_CARD.md` (Sensitive content bullet):
FaceNet embeddings are built to encode stable, matchable facial
characteristics and may be *more* re-identification-prone than the photo
alone; pseudonymization/dropping filenames reduces risk but isn't
anonymization under GDPR. Made explicit: every derived artifact (crops,
embeddings, landmarks, geometric features, aggregate and per-rater
ratings, provenance mapping) stays gated under the same terms as the
source images by default — none assumed safe to open later without its
own review. Also recorded as planned-not-built: separating public code
from gated data into different repos at actual release time, and an
explicit note that a "re-fetch from source URL instead of redistributing"
script would not be a licensing workaround (and 2 of 3 checked source
links are already dead anyway).

**User decision**: proceed without formal lawyer review before finalizing
license text, weighing that an earlier, less-audited release of this
dataset had modest reach (~50 likes / ~50 citations reported) with no
known issues, against the fact this project has now explicitly documented
finding public figures and unverified rights — asked to finalize
regardless. Contracting party and governing law for the actual agreement
text still not provided; agreement text itself not yet drafted. Drafted
RUA v0.1 in-conversation per user-specified contracting party (Irina
Lebedeva, PhD, as maintainer, rights she owns/administers only — not sole
owner of all photos) and governing law (Hungary/Budapest); user then
pointed out this had scaled past what's typical for an academic dataset
release (contrasted against SCUT-FBP5500-style lightweight terms) — not
saved to a file, deferred, to revisit at the end of session.

**Finding 8 resolved, 2026-07-26**: maintainer visually reviewed all 7
real label collisions and chose the correct label for each (table in
`docs/DATASET_AUDIT.md`). Implemented as `LABEL_COLLISION_RESOLUTIONS` in
`scripts/data/build_v3_dataset.py`; `data/mebeauty_v3/` rebuilt end-to-end
(images/metadata.parquet, landmarks.parquet, ratings/aggregate/,
ratings/by_rater/) — `COVERAGE.json` now shows 7/7 collisions resolved, 0
remaining. Found and fixed a real bug along the way while pulling source
links for review: `infer_provenance()`'s Unsplash-id extraction truncated
ids containing an internal hyphen (195/1,325 = 14.7% of Unsplash-pattern
filenames affected); fixed in
`src/mebeauty_benchmark/legacy/provenance.py`, covered by a new
regression test, `image_provenance.csv` regenerated (317 rows changed).
`make check`: 43/43 passing throughout.

**Finding 14 rewritten, 2026-07-26** — both the metric and the conclusion
were wrong and are now corrected. The original "extreme mean" heuristic
measures scale preference, not signal: verified that it false-positived two
raters who discriminate fine and merely score generously (spread +0.92,
+0.69), while missing several high-volume raters with strongly *negative*
discrimination whose means look ordinary (`rater_0405` 310 ratings at
−2.90, `rater_0213` at −8.71). Replaced with a **discrimination spread**
statistic (mean score on the rest of the pool's top consensus tercile minus
the bottom tercile, consensus computed leave-one-out so a rater never
defines their own yardstick) — robust to range restriction, so a harsh but
engaged rater still scores clearly positive.

Two intermediate recommendations were made and then withdrawn on evidence
before landing on the final one, which is worth recording because the
reasoning matters more than the outcome: (1) "drop the 4 straight-liners,
keep the 2 high-volume extremes" — backwards, since the straight-liners
move the aggregate by 0.001 while the two big ones move 477 images >0.10;
(2) "exclude non-discriminating raters" — rejected after testing the rule
against *all* 528 measurable raters instead of only the pre-flagged ones,
which is what it would honestly require. That showed the spread is a smooth
continuum (−8.71 to +7.56, median +2.33) with no gap, so every threshold is
arbitrary and the cost swings 7%→19% of ratings across defensible cutoffs;
worse, applied fully it moves 576 images' mean by >0.25 at spread ≤0 and
1,119 images (~45% of the dataset) at ≤1.0. Filtering by
agreement-with-consensus is also circular and would delete the minority
aesthetic variation this dataset exists to study.

**Final: measure and ship, exclude nobody.** Canonical score stays the
unweighted mean of all ratings. `scripts/data/check_rater_quality.py`
rewritten to emit the statistic, the distribution, the 20 lowest raters,
and full sensitivity/impact tables, plus a shipped
`data/mebeauty_v3/ratings/by_rater/rater_quality.parquet` (831 raters) so
consumers can filter themselves. Docs updated (`DATASET_AUDIT.md` Finding
14 rewritten with the superseded version kept for the record,
`DATASET_CARD.md` Ratings + known-issues). Also corrected a stale
by-rater image count in the card (2,484 → 2,486).

**`docs/CHANGES_VS_ORIGINAL.md` written, 2026-07-26** — the
publication-facing changelog against the original
`fbplab/MEBeauty-database` repo (pinned commit
`7b849562ee92d99d34d56afa2ebe85e5075f9b20`), intended to ship with the
GitHub repo and any eventual HF dataset card. Covers: the headline
before/after table, silent-failure recovery (leading with the
`male/indian` embedding gap as the one change that can alter published
conclusions), corrections, the 3 removals, structural changes, new
artifacts, rater-quality policy, an explicit "what did NOT change"
section, known unresolved discrepancies, and the still-open licensing /
rights / GDPR items. All counts re-verified against the built artifacts
before writing rather than taken from memory (2,495 images; 1,746/222/517
= 2,485 canonical rows; 115,622 by-rater rows over 2,486 images and 831
raters; legacy 1,351 F / 1,196 M). README restructured with a
"Coming from the original repository?" pointer, a documentation index, and
an unlicensed-status warning; all relative doc links verified to resolve.

### 2026-07-26 (review pass 3)

Branch `chore/legacy-inventory`. Tests: `make check` clean, 46/46 passing.
Fresh review of the rebuilt `data/mebeauty_v3/`. Verified clean: the
content-addressing invariant (all 2,495 filenames re-hashed and matched
their content exactly), image/metadata/landmark/split/by-rater referential
integrity, no metadata nulls, split balance (score, gender, ethnicity all
near-identical across train/val/test), and HF `load_dataset` still loading
all 2,495 with the new schema.

Two new findings, both real and both fixed:

- **Finding 15 — 32-character filename truncation.** 5 images stored under a
  stem cut to exactly 32 chars while the score files kept the full name, so
  all 5 were dropped as "not found": rated images lost to a filename bug.
  Recovered via `_resolve_truncated_basename`, which declines unless exactly
  one file matches the truncation *and* no full-named file exists — the
  `jonathan-borba` case is two different photographs, and a naive match
  would have attributed one photo's rating to the other. `not_found` 22 →
  18; canonical train 1,746 → 1,751; images with no canonical rating 10 → 5.
  Both decline-paths regression-tested.
- **Finding 16 — two rating tasks pooled into one column.** The legacy
  collection ran a *generic* and a *date* attractiveness task over the same
  images (means 6.00 vs 5.01). `build_ratings_by_rater.py` discarded the
  source label and deduped on `(image_id, rater)`, which merged two
  different questions into one `score` *and* destroyed 7,555 genuine ratings
  as false duplicates (115,622 → 123,177). Verified the canonical aggregate
  corresponds to `generic` (bias -0.06, corr 0.966) not the pooled mixture
  (bias +0.51, corr 0.894). Fixed with a `rating_type` column, task-aware
  dedup, and task-aware leave-one-out consensus in `check_rater_quality.py`
  — the pooled consensus alone had inflated the non-discriminating rater
  count from 42 to 52.

Rebuilt end-to-end: canonical splits → v2 → v3 → by_rater → rater_quality →
croissant (0 validation errors). Docs updated: Findings 15/16 added to
`DATASET_AUDIT.md`, `DATASET_CARD.md` (rating-task warning + counts),
`CHANGES_VS_ORIGINAL.md` (sections 2.5/2.6 + summary table).

Still open and unchanged: licensing/RUA text (deferred to end of session by
user), redistribution rights, GDPR classification, identity-screening depth,
historical split identification, score-disparity interpretation.

### 2026-07-28

Branch `feat/label-provenance`. Tests: `make check` clean, 67/67 passing.

Independent review of the built `data/mebeauty_v3/` (not of the docs) found
five issues the audit did not cover. **Finding 20 — the canonical labels are
not recomputable from the released ratings — was solved this session; the
other four are recorded but untouched.**

- **Finding 20 added and solved.** Averaging the per-rater table reproduces
  only 11.6% of canonical labels. Root cause found by reading the 2021
  notebooks in `data/legacy_snapshot/MEBeauty_creation_cleaning/`: the labels
  are the output of a **five-step rater-cleaning pipeline** (drop <50-rating
  raters; mask per-image >2σ scores; average; drop raters with
  `abs(corr) < 0.10`; drop a rater's ratings for one gender if they scored
  >90% of it at the floor), not a plain mean. Verified
  `generic_scores_all_2022.xlsx` is already the post-cleaning matrix —
  re-applying the 2σ step degrades the match (mad 0.012 → 0.067). The
  pipeline's inputs (`pers.xlsx`, `/home/ubuntu/ECUST_FBP/scores/*.xlsx`)
  are gone, so labels are explicable but permanently unrecomputable.
  **Resolution: keep the labels byte-identical, ship the evidence.**
  `scripts/data/enrich_label_provenance.py` +
  `src/mebeauty_benchmark/legacy/label_provenance.py` (8 tests) add
  `n_ratings`, `score_std`, `recomputed_score`, `score_delta`,
  `label_discrepancy` to `ratings/aggregate/*.parquet`. 17 images flagged
  (16 train, 1 test; worst 3.10). Canonical `score`/`image_id` verified
  byte-identical against pre-enrichment copies for all three splits.
- **"Historical protocol identification" open item CLOSED — as unanswerable.**
  The 2021 code calls `train_test_split()` with no `random_state` on a
  shuffled frame. No seed exists; the published split cannot be regenerated
  by anyone. Finding 2's separate composition mismatch stays open.
- **Root cause found for Finding 6's duplicate split rows**: the same cell
  writes with `to_csv(..., mode='a')` (append), so re-running it duplicates
  rows. Documented but previously unexplained.
- Corrected a now-false claim in `DATASET_CARD.md` that the canonical score
  is "the unweighted mean of every rating — no rater is excluded." True of
  `ratings/by_rater/`; false of the canonical labels, which inherit 2021's
  exclusions. Both now stated together, in tension, deliberately.
- `build_croissant_metadata.py` extended to describe the five new columns
  (it hardcoded `image_id`/`score` and would have under-described the
  release). Regenerated: 0 validation errors. `load_dataset()` re-verified
  for both configs: 2,495 rows at 400×400 and 256×256, no phantom splits.

**Four issues found and recorded but NOT addressed** (see the agent memory
note and the list below):

1. `scores/private_generic/` (10 files), `scores/private_date/` (25 files)
   and `private_generic_all.xlsx` are **never read by any script** — `grep -rl
   private scripts/ src/` returns nothing. Finding 9's "43 files" does not
   mention they exist. Their raters are demographically coded (`cf41` =
   caucasian female 41), i.e. rater age/gender/ethnicity — directly relevant
   to the personalization goal.
2. Pseudonymization misses non-MTurk IDs: `data/pseudonymized_scores/
   generic_scores_all.xlsx` still contains literal `cm39, cf41, af48, hm23,
   cf34_2, cf34, cm17, cf25, af18, cm37`, and the script's "no unmapped
   worker ID remains" assertion passes anyway. Confined to a gitignored local
   intermediate (not in shipped `ratings_by_rater.parquet`), so low severity —
   but those 10 raters are silently absent from the shipped per-rater table.
3. Finding 19's 25 multi-face + 4 no-face rated images have **no flag in
   `images/metadata.parquet`** — the list lives only in `reports/`, which is
   not part of an HF release. `has_out_of_bounds_landmarks` is likewise
   absent despite Finding 17 claiming it ships.
4. Generic ratings per image range 9–78 (median 21); 444 images have <10.
   Label reliability varies ~8x. Now partly visible via Finding 20's
   `n_ratings` on the aggregate tables, but not in `metadata.parquet`.

Next action: decide on the four above (1 and 3 are the ones that matter for
a credible release), then licensing — still the actual blocker.
