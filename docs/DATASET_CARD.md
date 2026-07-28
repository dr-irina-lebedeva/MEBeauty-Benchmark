---
pretty_name: MEBeauty (candidate release, not yet published)
license: other
license_name: mebeauty-research-use-agreement-draft-pending-legal-review
license_link: https://github.com/dr-irina-lebedeva/MEBeauty-Benchmark/blob/main/docs/DATASET_AUDIT.md
task_categories:
  - other
tags:
  - facial-beauty-prediction
  - facial-attractiveness
  - multi-ethnic
  - regression
  - gated-pending-review
size_categories:
  - 1K<n<10K
---

# MEBeauty Dataset Card (draft — not released)

**Status: draft candidate, not published to Hugging Face.** This card
describes the dataset as it exists after the audit in
[`docs/DATASET_AUDIT.md`](DATASET_AUDIT.md), not a live release. Licensing,
redistribution rights, and GDPR review are still open — see that document's
"Open — needs your decision" section before treating anything here as final.
No images, embeddings, or annotations have been uploaded anywhere.

> **⚠ `docs/DATASET_AUDIT.md` Finding 13 — read this even though it's
> "resolved".** Three images of named public figures (Michelle Obama,
> Deepika Padukone, Aditi Rao Hydari) were found and are now excluded from
> this dataset. But the *second* of two screening passes found the third
> case — a case the *first* pass's own filename pattern could not
> structurally have caught. That is direct evidence the screening method
> (filename patterns) has a real, demonstrated blind spot, not a
> reassurance that it's now thorough. Treat "3 found and removed" as a
> floor, not a ceiling.
>
> A perceptual screen has since been built and validated (Finding 18): it
> ranks all three confirmed figures inside the top 19% of images, cutting a
> visual review ~5x. It is **triage, not a detector** — it identifies nobody,
> its validation rests on three examples, and VGGFace2's 8,631 identities are
> a small, English-media-skewed fraction of all public figures.
>
> That visual pass has been run over ranks 1-672 (26% of images, past the
> rank-478 threshold containing all three known cases). All three were
> re-identified at their predicted ranks and **no additional public figures
> were recognised**. Still not a guarantee: 1,874 lower-ranked images were
> not reviewed, and
> recognition skews to internationally famous people. No method available
> here supports claiming the dataset is free of public figures.

## Dataset summary

MEBeauty is a multi-ethnic facial beauty dataset with human attractiveness
ratings on a 1–10 scale.

**What this release actually contains.** The images were originally collected
from unconstrained, in-the-wild photographs, but the surviving legacy release
is **primarily centred face crops, not the complete source photographs** —
2,472 of 2,495 files (99.1%) are 400×400, 500×500, or 600×600 tight crops,
and only 20 exceed one megapixel (Finding 17). This distinction matters for
what the benchmark measures:

- **Strength**: diverse, multi-ethnic attractiveness ratings, including
  per-rater annotations for personalization research.
- **Input**: heterogeneous, preprocessed face crops.
- **Limitation**: the original full-scene photographs are mostly unavailable,
  so this release **does not support measuring robustness to background,
  framing, or full-scene variation**, and a modern detector cannot be used to
  re-crop from source pixels.
Introduced in Lebedeva, Guo & Ying, *"MEBeauty: a multi-ethnic facial beauty
dataset in-the-wild"* (Neural Computing and Applications, 2022,
[doi:10.1007/s00521-021-06535-0](https://doi.org/10.1007/s00521-021-06535-0)).
This repository re-audits and re-packages the original release; see
`README.md` for project scope.

## Dataset structure

### Instances

`data/mebeauty_v3/images/` — **2,495 unique images**, flat and
content-addressed (`<sha256>.<ext>` filename), labels in
`images/metadata.parquet` only (no folder-encoded gender/ethnicity — see
`docs/RESTRUCTURE_PROPOSAL.md` for why). Started from 2,547 legacy files;
49 collapsed as byte-identical duplicates (Finding 7) and 3 were removed
as named public figures (Finding 13). 7 images (7 rows, now one row each
post-dedup) had a conflicting label between two source folders; resolved
by maintainer review — see Finding 8.

> **⚠ These are pre-cropped face images, not original photographs.**
> 2,472 of 2,495 (99.1%) are exactly 400×400, 500×500, or 600×600 tight face
> crops; only 20 exceed one megapixel. Filenames can be misleading —
> `male-4572748_1920.jpg` carries the source's 1920px marker but is stored as
> a 400×400 crop. **You cannot meaningfully re-crop these with a different
> detector** (you would be cropping a crop), and effective resolution is
> heterogeneous across the set. The paper's "in-the-wild / unconstrained"
> description refers to the source photographs, not to these files. See
> Finding 17.

**Two configurations** ship over the same 2,495 images, joined by
`image_id`: `native` (default -- the legacy files byte-unchanged,
heterogeneous resolution) and `standardized_256` (uniform 256x256 RGB,
aspect-preserving Lanczos, centre-padded, JPEG q95, landmarks transformed to
match). Native is authoritative; standardized is a reproducible convenience
layer, verified byte-identical across runs. See Finding 17.

```python
load_dataset("<repo>", name="native")  # default, 2,495 rows
load_dataset("<repo>", name="standardized_256")  # uniform 256x256
```

`load_dataset("imagefolder", data_dir="data/mebeauty_v3/images")` also loads
the native config directly — verified against the real directory, not assumed.
Machine-readable [Croissant](https://mlcommons.org/croissant) metadata is
at `data/mebeauty_v3/croissant.json` (generated and schema-validated
locally, license field is a placeholder pending the real decision).

### Canonical metadata tables

`data/mebeauty_v3/` (built by `scripts/data/build_v3_dataset.py`):

| File | Rows | Description |
|---|---|---|
| `images/metadata.parquet` | 2,495 | image_id (SHA-256), file_name, legacy_filename/path, gender, ethnicity, **width/height/megapixels/crop_batch/is_preprocessed_crop**, label-collision flag, near-duplicate flag, inferred provenance |
| `landmarks.parquet` | 2,495 | image_id, 68-point landmarks (native `list<float>`, 136 values) |
| `ratings/aggregate/{train,val,test}.parquet` | 1,751 / 222 / 517 | Canonical split ratings (generic attractiveness), plus per-label rater support: `n_ratings`, `score_std`, `recomputed_score`, `score_delta`, `label_discrepancy` |
| `ratings/by_rater/ratings_by_rater.parquet` | 123,177 | Individual pseudonymized rater scores with `rating_type` (`generic` 60,046 / `date` 63,131), 2,486 images, 831 raters |
| `ratings/by_rater/rater_quality.parquet` | 831 | Per-rater quality statistics (volume, mean, std, discrimination spread) — for consumer-side filtering; no filtering is applied to the canonical score |

No pixel data or rater identifiers in any of these beyond the pseudonymous
`rater_XXXX` id. (`reports/legacy_audit/canonical_dataset/` has the
equivalent pre-restructure, path-keyed tables from the audit itself.)

### Data splits

Canonical split (`benchmark-v1`, derived from the legacy "2022" split
generation — every rating resolved against the current `original_images/`
tree, deduped by both exact path and image content, Finding 13's 3 images
excluded — see Finding 6):

| Split | Rows |
|---|---|
| train | 1,751 |
| val | 222 |
| test | 517 |

This is **not the original paper's split, and the paper's split no longer
exists.** The 2021 code generated it with `train_test_split()` without a
`random_state`, on a randomly shuffled frame, so no seed was ever recorded
and the published partition cannot be regenerated by anyone (Finding 20).
`benchmark-v1` is therefore the forward protocol; results computed on it are
not directly comparable to the paper's reported numbers, and no such
equivalence should be claimed.

The other two legacy split generations have the same full rigor
applied (`reports/legacy_audit/canonical_splits_{original,crop}/`) — the
"original" generation has 252 relabeled rows out of 2,514 (10%, vs. 0.3%
for "2022"), suggestive evidence "2022" is a later, cleaned-up revision.

### Ratings

Score is a continuous attractiveness rating, roughly in the 1–9.6 range
(mean ≈ 6.0–6.1, consistent across splits). Collected from ~300 Amazon
Mechanical Turk raters; raw Worker IDs have been pseudonymized (831 unique
raters, one stable `rater_XXXX` id each) and the identity mapping is kept
local-only, never committed. Individual (not just averaged) rater scores
are now available in `ratings/by_rater/` for personalization research.

> **⚠ Two different rating tasks — check `rating_type` before use.** The
> collection ran a *generic* attractiveness question (60,046 ratings, mean
> 6.00) and a *date* attractiveness question (63,131 ratings, mean 5.01)
> over the same images. These are different questions and sit about a full
> point apart. **The canonical aggregate corresponds to `generic`** (bias
> −0.06, correlation 0.966 against the per-rater generic mean; the pooled
> mixture is offset +0.51). If you want the per-rater equivalent of the
> canonical label, filter `rating_type == "generic"`. The `date` ratings
> are a genuine second annotation layer, not noise. See Finding 16.

> **⚠ The canonical score cannot be recomputed from the per-rater table, by
> design.** Averaging `rating_type == "generic"` reproduces only 11.6% of the
> labels exactly. This is not a defect in the ratings: the 2021 collection
> pipeline applied rater-cleaning steps before averaging. Two are confirmed
> to have taken effect — per-image outlier masking (scores >2σ from that
> image's mean) and dropping raters whose scores barely correlate with the
> pooled average (`abs(corr) < 0.10`) — plus a minimum-ratings floor at 30.
> Two further steps appear in the notebooks but are **verified not to have
> run**, including one that silently no-ops; see Finding 20 for which, and
> for how each verdict was tested. The closest surviving rater
> matrix reproduces the labels to a mean absolute difference of **0.012**,
> confirming provenance — but the pipeline's intermediate inputs are lost, so
> the labels are distributed **as-is rather than recomputed**. Every
> `ratings/aggregate/*.parquet` row carries `n_ratings`, `score_std`,
> `recomputed_score`, `score_delta` and `label_discrepancy` so the gap is
> visible per image. **17 images (16 train, 1 test) disagree by more than
> 0.25**, worst case 3.10 — consider excluding them for label-sensitive work.
> See Finding 20.

**No rater is excluded from `ratings/by_rater/` — but the canonical labels
inherit 2021's rater exclusions and cannot be un-inherited.** These two facts
sit in tension and are both true; read them together. This project's own
policy (Finding 14) is to measure rater quality and exclude nobody, and the
per-rater table follows it. The canonical labels predate that policy and
already embed consensus-based filtering of exactly the kind Finding 14 argues
against. Per-rater quality statistics ship alongside,
in `ratings/by_rater/rater_quality.parquet`: `n_ratings`, `mean_score`,
`std_score`, and a **discrimination spread** (how much higher a rater
scores images the rest of the pool ranked in its top consensus third
versus its bottom third, computed leave-one-out). Filtering is left to the
consumer, deliberately: the spread is a smooth continuum from −8.71 to
+7.56 with no natural cutoff, and excluding the non-discriminating tail
would move 576–1,119 images' mean score by >0.25 depending purely on where
the (arbitrary) line is drawn. Filtering by agreement-with-consensus is
also circular for a dataset whose research goals include studying rater
variation. See Finding 14 for the full argument and the sensitivity
tables.

## Known data issues (see `docs/DATASET_AUDIT.md` for full detail and repro commands)

- **The canonical labels cannot be recomputed from the released ratings, and
  17 of them materially disagree with every surviving rater matrix** (16
  train, 1 test; worst delta 3.10). The labels are the output of a 2021
  rater-cleaning pipeline whose intermediate inputs no longer exist. Each row
  carries `score_delta` and `label_discrepancy` so this is auditable per
  image. See Finding 20 and the Ratings section above.
- **The original paper's train/val/test split is permanently unrecoverable** —
  generated without a random seed (Finding 20). Use `benchmark-v1`; do not
  claim comparability with the paper's reported numbers.
- **The legacy MTCNN pipeline failed on 94.7% of `male/indian` images
  (143/151), versus ~0% everywhere else** — confirmed to be an isolated
  pipeline/environment failure, not a property of those images (the
  independent OpenCV backend processed the same folder normally). Anyone
  using the original `FaceNet_512_features` had only 8 of 151 male/Indian
  embeddings (5.3% coverage) vs. near-100% for every other subgroup. This
  session recovered all 143 (visually spot-checked as genuine); see
  Finding 3.
- The remaining crop/landmark gaps (145 missing MTCNN, 197 missing OpenCV,
  102 missing landmarks, across all subgroups) have also been recovered:
  100% MTCNN, 96% OpenCV (8 genuinely undetectable, verified against a
  second detector to rule out false positives), 100% landmarks. See
  `data/mebeauty_v2/COVERAGE.json`.
- 2 recovered images have some geometric-ratio features that are
  mathematically undefined (`NaN`, not a missing-data gap) — both extreme
  profile poses where the 68-point landmark scheme collapses two distinct
  points onto one pixel.
- 33 of the original 2,511 canonical rating rows referenced an image that
  no longer existed at its rated path — 4 from a path-quoting parser bug
  (now fixed), 7 from images later reclassified to a different
  ethnicity/gender folder (remapped, rating kept), and 22 from images
  removed from the dataset entirely since rating (dropped — nothing to
  join them to). Every remaining row is now verified to join to a real
  image file; final canonical size is 2,488, not 2,511.
- 49 groups of images (98 files) are byte-identical duplicates saved under
  different filenames; 18 of those pairs previously leaked across
  train/val/test and have been removed from the canonical split.
- 8 images were filed under two conflicting ethnicity/gender labels — 7
  were the same photo duplicated across labels (a filing error, now
  resolved by maintainer review; `metadata.parquet`'s `gender`/`ethnicity`
  reflect the chosen label, `has_label_collision` still records that the
  conflict existed), 1 is two different photos that coincidentally share a
  filename (not a labeling issue at all, but a latent conflation risk for
  any basename-keyed code).
- 1 orphaned derived artifact (an MTCNN crop + FaceNet embedding) has no
  corresponding file left in `original_images/` — excluded from
  `data/mebeauty_v2/`.
- 1 near-duplicate pair (perceptual hash, not byte-identical — the same
  photo/moment re-saved under two different stock-platform names) found in
  `data/mebeauty_v3/` beyond the byte-exact duplicates above. Both images
  are in `train`, so not train/test leakage, but flagged in `metadata.parquet`
  (`has_near_duplicate`) as redundant content, not removed.
- **3 images of named public figures found and excluded** (Michelle Obama,
  Deepika Padukone, Aditi Rao Hydari — Finding 13). Not a data-quality bug;
  a content/consent issue. See the warning at the top of this card — the
  screening method has a demonstrated blind spot, this is not a complete
  identity check.
- **52 of 831 raters (6.9% of ratings) show no discrimination** — their
  scores are unrelated to (or inverted from) what the rest of the pool sees
  on the same images, including 4 who gave one identical score to every
  image they rated (Finding 14). Measured and shipped in
  `ratings/by_rater/rater_quality.parquet`; **not** excluded or
  down-weighted, for the reasons in Finding 14. An earlier version of this
  check used an "extreme mean" test that both false-positived engaged
  raters and missed the worst offenders; it has been replaced.
- **25 rated images (0.98%) contain two or more faces**, and 4 contain no
  detectable face at all (Finding 19). Where two people share a frame,
  nothing records which face the rating, gender and ethnicity labels
  describe. Affected images are listed in
  `reports/legacy_audit/faces_per_image.json`.
- Image provenance (source platform) is a filename-pattern inference;
  **101 of 2,495 images (4.0%) have no source link at all**.
  Use `inferred_photo_id`, not `inferred_source_url`, as the durable
  identifier — two Unsplash IDs have been externally confirmed correct, but
  Unsplash's canonical URL embeds a description slug that is not derivable
  from the filename, so the generated URL is best-effort. Link quality is
  uneven: Unsplash/Pexels encode the platform's real identifier; Pixabay
  URLs are reconstructed from the filename stem and are the least reliable. Every automated recovery route has been tried and
  failed for those — no filename signal, no embedded EXIF/PNG metadata (0 of
  115), and no perceptual match to a known-provenance image within the
  validated threshold (closest is distance 60 against a threshold of 20).
  Reverse image search is the only remaining option and must be done by hand.
  See `reports/legacy_audit/unknown_provenance_recovery.json`.
- The published paper's own abstract states 2,550 images (1,250 men /
  1,300 women); this snapshot has 2,547 images (1,196 men / 1,351 women) —
  the underlying image set has changed since publication in a way not
  reconstructable from this repository.

## Considerations for using the data

- **Sensitive content.** Images depict identifiable human faces. Face
  embeddings and landmarks may constitute biometric data depending on
  processing and use — this has not had a GDPR/legal classification pass.
  No data controller has been designated. **Embeddings are not lower-risk
  than the photos** — FaceNet-512 embeddings are built to encode stable,
  matchable facial characteristics and could support re-identification
  against another face database; removing filenames or storing derived
  features instead of raw pixels does not make them anonymous data.
  Everything derived from a face — crops, embeddings, landmarks, aggregate
  and per-rater ratings — is gated under the same terms as the source
  images (see Licensing below), not treated as automatically lower-risk.
- **Subgroup representation bias in derived features, now fixed but worth
  knowing about.** The `male/indian` FaceNet-embedding gap above means any
  model or analysis built on the *original* legacy `FaceNet_512_features`
  (rather than this session's recovered/merged set) would have been
  trained on a dataset that was effectively missing 95% of one ethnicity/
  gender subgroup's embeddings — worth flagging explicitly to anyone who
  may have already used the un-recovered legacy artifacts.
- **Subjectivity and bias.** Attractiveness ratings reflect the specific
  rater pool's judgments (~300 MTurk workers, demographics not verified) at
  time of collection. Do not treat scores as an objective ground truth;
  results should be read alongside subgroup and rater-demographic analysis
  once conducted. Raw group means in the canonical split (unadjusted for
  rater demographics, photo selection, or sample size — a starting
  observation, not a finished analysis): female 6.58 vs. male 5.56;
  by ethnicity, black 5.41 (lowest) to hispanic 6.46 (highest), with
  caucasian 6.30 and indian/asian/mideastern clustered 5.89–6.05. This is
  not evidence of a data-quality defect the way e.g. Finding 3's detection
  failure was — it's a descriptive fact about the score distribution, and
  properly characterizing *why* (rater bias, sample composition, genuine
  differences in the specific photos collected, or some mix) is exactly
  the subgroup/fairness research this project already lists as a goal, not
  something resolved by this audit.
- **Redistribution rights.** Source images were drawn from Unsplash, Pexels,
  and Pixabay by filename inference; per-platform redistribution rights have
  not been verified for this release.
- **Consent / takedown.** `images.parquet`'s `inferred_platform` /
  `inferred_photo_id` / `inferred_source_url` columns exist specifically so
  a future consent or takedown request can be resolved against a single
  photo rather than "somewhere in 2,547 files."

## Licensing

**Release model decided; exact legal wording still pending.** This will
**not** be a fully open dataset (no CC0, CC BY, or MIT on the images/
ratings/landmarks) — the plan is a **gated release**: a public dataset
page, individual access requests, and a custom "MEBeauty Research Use
Agreement" (non-commercial research use only, no redistribution or
sublicensing, no commercial use, no face-recognition/surveillance/
profiling use, mandatory takedown compliance, per-user approval, citation
required). Newly written benchmark code (where ownership is confirmed)
will use Apache-2.0; the dataset repository itself is marked
`license: other` above.

**Gating does not create redistribution rights.** Images with unverified
or unresolved source permissions (see "Redistribution rights" below and
`docs/DATASET_AUDIT.md` Finding 11) will be withheld from the distributed
package regardless of gating — an access agreement controls what an
approved user may *do* with the data, not whether the photo was ever
permitted to be included at all.

This repository stays **private**, not just gated, until (a) the actual
Research Use Agreement text has been reviewed by a qualified lawyer,
covering GDPR, demographic labels, facial landmarks, and international
access, and (b) image-level rights/exclusions (Findings 11 and 13) are
resolved. See `docs/DATASET_AUDIT.md` for full detail.

**Planned access mechanism**: Hugging Face gating for identity-linked
access control; where stronger contractual assurance is warranted, access
approved only after a separate Research Use Agreement is executed through
an appropriate e-signature process (agreement version, signer identity,
approval date, and HF username retained per user). Exact requirements —
whether click-wrap suffices or an advanced/qualified e-signature is
needed, contracting party, governing law — determined by the legal review
above, not decided here.

## Citation

```bibtex
@article{lebedeva2022mebeauty,
  title   = {MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author  = {Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal = {Neural Computing and Applications},
  volume  = {34},
  number  = {17},
  pages   = {14169--14183},
  year    = {2022},
  doi     = {10.1007/s00521-021-06535-0}
}
```

## Maintainer

Irina Lebedeva, PhD — see `README.md`.
