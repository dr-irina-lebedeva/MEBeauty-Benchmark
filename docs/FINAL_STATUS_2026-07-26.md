# Final Status — `data/mebeauty_v3/`, 2026-07-26

Written after a fresh review pass specifically for this question: what can
still be improved, and what genuinely can't be solved without you. Full
detail and repro commands for everything below: `docs/DATASET_AUDIT.md`.

**Update, later the same day: all 6 items in "Issues / improvements I can
still do" below were executed on request.** That list is left as-written
for the historical record of what was proposed; see `docs/PROJECT_STATUS.md`
for what actually happened when it was carried out — notably, executing
item 5 (extend the identity screen) found a *third* public figure the
original Finding 13 write-up didn't have, which is itself evidence for
item-9-style humility: acting on a "can do" list can surface things the
list itself didn't anticipate. The "Issues I cannot solve" list below is
still accurate and still stands.

This pass added three new checks not done before:
1. Extended the public-figure screen (Finding 13) to Unsplash/Pexels
   name-pattern filenames (1,474 of them) — concluded, with reasoning not
   just absence of results, that they encode the *photographer's* credit
   by platform convention, not the subject, so they're not the same risk
   channel as Pixabay's occasional subject-descriptive naming (already
   exhaustively reviewed, all 56 candidates).
2. Verified all 2,498 `data/mebeauty_v3/images/` files open correctly
   (0 failures) and `metadata.parquet` has zero nulls / empty required
   fields — basic hygiene never explicitly re-checked for v3 specifically.
3. Checked rating-score distribution by gender/ethnicity — found real
   disparities (female 6.58 vs. male 5.56 mean; ethnicity 5.41–6.46) and
   documented them carefully in `docs/DATASET_CARD.md` as a descriptive
   observation, explicitly *not* claimed as a bug — unlike every technical
   finding this session, there's no objective ground truth to compare a
   rating distribution against, and subgroup analysis is a stated research
   goal of this project, not something an audit resolves.

## Issues / improvements I can still do

Everything in this list is mechanical or investigative — no judgment call
about someone else's rights, consent, or the project's direction required.

1. **Act on Finding 13** — actually exclude the 2 identified images
   (Michelle Obama, Deepika Padukone) from `data/mebeauty_v3/` and rebuild.
   Flagged, not yet done, because removing rated data is the kind of
   action I do on request, not unilaterally.
2. **Extend the near-duplicate/identity screen further** — e.g. run
   perceptual hashing against a public figure reference set if one is
   available, rather than relying on filename patterns and personal
   recognition. Would need an external face-matching resource I don't
   currently have access to; framing and threshold-setting work already
   done (Finding 12's methodology) would carry over directly.
3. **Reconcile the other two split generations** (`train`/`val`/`test.txt`
   "original", `train_crop`/`test_crop.csv`) to the same full rigor as the
   canonical 2022-based split — existence resolution, exact-path dedup,
   content-hash dedup, near-duplicate check. Only ever done as informal
   reconciliation counts, not built into real, usable, verified artifacts
   the way `benchmark-v1` was.
4. **Per-rater data quality pass** — e.g. flag raters with suspiciously
   uniform or extreme scoring patterns (straight-lining). Never attempted;
   the per-rater table (Finding 2 of the restructure) only reconciled
   *identity*, not rating *quality*.
5. **`reproduce_legacy_baseline/` scripts are documented but only
   spot-verified** (3 images) against `data/mebeauty_v2/`'s crops. A full
   re-run and comparison across all 2,547 would either strengthen or
   correct the "methodologically equivalent, not pixel-identical" claim in
   `docs/REPRODUCE_LEGACY_BASELINE.md` with complete rather than sampled
   evidence.
6. **Croissant / HF metadata generation** — not yet attempted at all. HF
   auto-generates Croissant metadata for Parquet-convertible datasets;
   worth generating and inspecting once the structure is otherwise final,
   per the original plan's stated preference for that over hand-writing
   `croissant.json`.

None of these are urgent the way Finding 13 is. I'd want to know which (if
any) you want before spending more time — several are diminishing-returns
verification of things already reasonably well-established.

## Issues I cannot solve

1. **Finding 13 itself, the underlying question** — whether these two
   images (or others not caught) may ethically/legally be included at all,
   and what "a fuller identity screening pass" should actually consist of.
   I can execute a removal on request; I can't make the underlying call.
2. **Licensing.** No license chosen for code, images, annotations,
   embeddings, splits.
3. **Redistribution rights per source platform** — inferred, never
   verified against Unsplash/Pexels/Pixabay directly.
4. **GDPR / biometric data classification.** Depends on intended use and
   needs legal review; no data controller has been named.
5. **Which label is correct** for the 7 collapsed label-collision images
   (Finding 8) — content-addressing removed the *duplication*, not the
   ambiguity about which folder's label was right.
6. **Historical split identification.** Tried again this session
   (Semantic Scholar, ResearchGate, direct search) — still can't get past
   the abstract. The 252-vs-7 relabel-count gap remains a strong clue, not
   proof, of which split generation is "the" historical one.
7. **117 images with no resolvable provenance signal at all** — genuinely
   unrecoverable from any signal available in the data itself.
8. **8 OpenCV crops with no detectable face**, even under expanded,
   MTCNN-cross-validated tuning — a real limitation of that detector
   family on those specific images, not a bug.
9. **Interpreting the rating-score subgroup disparities** — documented as
   a fact, but *why* they exist (rater bias vs. sample composition vs.
   genuine difference vs. some mix) is a research question requiring
   proper controlled analysis, not something this audit can determine.
10. **Whether `data/mebeauty_v3`'s structure is the one you actually want
    to ship.** I implemented the restructure after you confirmed it, but
    "is this right for an eventual public/HF release" is a standing
    product decision, not a closed technical question.
