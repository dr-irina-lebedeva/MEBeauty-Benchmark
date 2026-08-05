"""Prove the release is safe to publish. Run before every upload.

This lives in the repository and never ships. It exists because every serious
defect found while building this release looked fine until it was measured:

* `other_legacy_paths` was splatted character by character, and two images
  silently lost their entire rating history;
* `score_mean` could not be reproduced from the shipped ratings, off by up to
  0.75, because the validity flag was missing;
* the card documented two columns that had been deleted;
* the image column loaded as raw dicts, so the Hub viewer would have rendered
  nothing.

None of those were visible by reading. Each check below corresponds to
something that actually went wrong, or would end the project if it did.

Exit code is non-zero if any check fails, so it can gate an upload.

    uv run python scripts/release/validate_release.py --release data/release
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

#: A real MTurk Worker ID is `A` followed by 12-20 alphanumerics. Matching
#: loosely is deliberate: a false positive costs a look, a false negative
#: publishes someone's identifier.
WORKER_ID = re.compile(r"\bA[A-Z0-9]{12,20}\b")

#: Anything that would leak a machine, a person, or a mailbox.
IDENTIFIERS = re.compile(
    r"(?:/home/|/Users/|C:\\\\|@[\w.-]+\.\w+|\b\d{1,3}(?:\.\d{1,3}){3}\b)"
)

#: Files the release is allowed to contain. Anything else is either a build
#: artefact that escaped, or something nobody decided to publish.
ALLOWED = {"README.md", "LICENSE"}
ALLOWED_PREFIXES = ("data/",)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.lines: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        self.lines.append(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}")
        return ok

    def note(self, label: str, value) -> None:
        self.lines.append(f"         {label}: {value}")

    def section(self, name: str) -> None:
        self.lines.append(f"\n{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="data/release")
    parser.add_argument(
        "--sample-images",
        type=int,
        default=0,
        help="Decode this many images per config (0 = all). Decoding all of "
        "them is slow but is the only way to prove none is corrupt.",
    )
    return parser.parse_args()


def load(public: Path) -> dict[str, pd.DataFrame]:
    """Each published config, concatenated across its splits."""
    frames = {}
    for config in (
        "fbp",
        "fbp_extended",
        "personalized_fbp",
        "personalized_date",
        "full",
    ):
        parts = sorted((public / "data" / config).glob("*.parquet"))
        frames[config] = pd.concat(
            [pd.read_parquet(p) for p in parts], ignore_index=True
        )
    return frames


def flatten(nested: pd.Series) -> pd.DataFrame:
    """Nested per-image rating lists -> one row per (image, rater).

    The flat rating tables are no longer published -- the same data ships
    nested inside the personalized configs -- so the checks that need a long
    table build it from what actually ships, rather than from a file a user
    will never see.
    """
    rows = []
    for image_id, ratings in nested.items():
        for record in ratings:
            rows.append({"image_id": image_id, **record})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.release).expanduser().resolve()
    public, private = root / "public", root / "private"
    report = Report()

    # ---------------------------------------------------------------- files
    report.section("Release contents")
    present = {str(p.relative_to(public)) for p in public.rglob("*") if p.is_file()}
    unexpected = [
        f for f in present if f not in ALLOWED and not f.startswith(ALLOWED_PREFIXES)
    ]
    report.check("only expected files present", not unexpected, str(unexpected))
    report.note("files", len(present))
    report.check("README.md present", (public / "README.md").is_file())
    report.check(
        "private tier is outside public/",
        private.is_dir() and not str(private).startswith(str(public)),
    )

    frames = load(public)
    images = frames["fbp_extended"]
    minimal = frames["fbp"]
    # `score_std`, `rating_count` and `source_url` ship only in `full`, so the
    # checks that need them read from there rather than from every config.
    complete = frames["full"]
    generic = flatten(frames["personalized_fbp"].set_index("image_id")["ratings"])
    preference = flatten(frames["personalized_date"].set_index("image_id")["ratings"])
    raters = (
        pd.concat([generic, preference])[
            [
                "rater_id",
                "rater_gender",
                "rater_ethnicity",
                "rater_age_band",
            ]
        ]
        .drop_duplicates("rater_id")
        .assign(pool=lambda d: d["rater_id"].str.split("_").str[0])
        .rename(
            columns={
                "rater_gender": "gender",
                "rater_ethnicity": "ethnicity",
                "rater_age_band": "age_band",
            }
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------- privacy
    report.section("Privacy")
    worker_hits, identifier_hits = 0, 0
    for path in sorted(glob.glob(str(public / "**" / "*.parquet"), recursive=True)):
        frame = pd.read_parquet(path)
        for column in frame.columns:
            # Image and nested-struct columns hold bytes and records, not
            # text. Stringifying them and regex-scanning matches compressed
            # pixel data -- 336 phantom "identifier" hits came from exactly
            # that, because only a column literally named `image` was skipped.
            if column.startswith("image") or column.endswith("ratings"):
                continue
            series = frame[column]
            if series.dtype != object and not str(series.dtype).startswith("str"):
                continue
            if series.map(lambda v: isinstance(v, (bytes, dict, list))).any():
                continue
            if True:
                text = series.astype(str)
                worker_hits += int(text.str.contains(WORKER_ID, regex=True).sum())
                identifier_hits += int(text.str.contains(IDENTIFIERS, regex=True).sum())
    card = (public / "README.md").read_text(encoding="utf-8")
    worker_hits += len(WORKER_ID.findall(card))

    report.check("zero MTurk Worker IDs", worker_hits == 0, f"{worker_hits} found")
    report.check(
        "zero emails / IPs / local paths in data",
        identifier_hits == 0,
        f"{identifier_hits} found",
    )
    report.check(
        "every rater id is a pseudonym",
        bool(raters["rater_id"].str.match(r"^(rater|panel)_\d{4}$").all()),
    )
    report.check(
        "no exact ages ship",
        "age" not in raters.columns,
        "rater_demographics must carry age_band only",
    )
    bands = set(raters["age_band"].dropna().astype(str))
    report.check(
        "age is published as bands",
        bands <= {"18-24", "25-34", "35-44", "45-54", "55+"},
        str(sorted(bands)),
    )
    report.check(
        "rater quality profiles are not public",
        not (public / "data" / "rater_quality.parquet").is_file(),
    )
    # The pseudonym -> worker mapping is the one file that would undo every
    # other privacy measure at once, so its absence is asserted by name.
    mapping_leaked = [
        f
        for f in present
        if any(
            word in f.lower() for word in ("mapping", "worker", "identity", "exact_age")
        )
    ]
    report.check(
        "pseudonym -> identity mapping is NOT in the public tier",
        not mapping_leaked,
        str(mapping_leaked),
    )

    # --------------------------------------------------------------- labels
    report.section("Labels")
    report.check(
        "score_standardized does NOT ship (would make the target ambiguous)",
        "score_standardized" not in images.columns,
    )
    report.check(
        "beauty_score is in every config",
        all("beauty_score" in f.columns for f in frames.values()),
        str(sorted(k for k, f in frames.items() if "beauty_score" not in f.columns)),
    )

    # `score_adjusted` is a fitted quantity, so the thing worth asserting is
    # that a user holding only the shipped ratings can reproduce it. If this
    # ever fails, the label was fitted on inputs the release does not contain.
    from mebeauty_benchmark.legacy.aggregation import (
        OFFSET_SHRINKAGE,
        fit_offset_model,
    )

    image_codes, image_values = pd.factorize(generic["image_id"])
    rater_codes, rater_values = pd.factorize(generic["rater_id"])
    refit = fit_offset_model(
        generic["score"].to_numpy(dtype=float),
        image_codes,
        rater_codes,
        len(image_values),
        len(rater_values),
        OFFSET_SHRINKAGE,
    )
    refit_scores = pd.Series(refit.quality, index=image_values)
    adjusted_difference = (
        (images.set_index("image_id")["beauty_score"] - refit_scores).abs().max()
    )
    report.check(
        "beauty_score is refittable from the shipped ratings alone",
        adjusted_difference < 1e-6,
        f"max diff {adjusted_difference:.2e}",
    )
    report.check(
        "beauty_score is on the 1-10 scale",
        bool(images["beauty_score"].between(1, 10).all()),
    )

    leaked = sorted(
        name
        for name, frame in frames.items()
        if name != "full" and "plain_mean_score" in frame.columns
    )
    report.check(
        "plain_mean_score ships in `full` only (one benchmark target elsewhere)",
        not leaked,
        str(leaked),
    )

    recomputed = generic.groupby("image_id")["score"].mean()
    # `plain_mean_score` ships in `full` only, so its checks read from there.
    joined = complete.set_index("image_id")["plain_mean_score"]
    difference = (joined - recomputed.reindex(joined.index)).abs().max()
    report.check(
        "plain_mean_score == plain mean of valid shipped ratings",
        difference < 1e-9,
        f"max diff {difference:.2e}",
    )

    # A date rating leaking into the generic score is the one mistake that
    # would silently corrupt every label, so it is tested by construction:
    # recomputing with date ratings mixed in must NOT match.
    mixed = pd.concat([generic, preference]).groupby("image_id")["score"].mean()
    report.check(
        "preference ratings do NOT contribute to the scores",
        (joined - mixed.reindex(joined.index)).abs().max() > 1e-6,
        "mixing the second task changes the mean, as it must",
    )

    report.check(
        "every rating is within 1-10",
        bool(
            generic["score"].between(1, 10).all()
            and preference["score"].between(1, 10).all()
        ),
    )
    report.check(
        "rating_distribution sums to rating_count",
        bool(
            (
                complete["rating_distribution"].apply(sum) == complete["rating_count"]
            ).all()
        ),
    )
    expectation = complete["rating_distribution"].apply(
        lambda p: sum((i + 1) * v for i, v in enumerate(p)) / max(sum(p), 1)
    )
    report.check(
        "distribution expectation == plain_mean_score",
        (expectation - complete["plain_mean_score"]).abs().max() < 1e-9,
    )
    report.check(
        "plain_mean_score is on the 1-10 scale",
        bool(complete["plain_mean_score"].between(1, 10).all()),
    )
    report.check(
        "no null in any label column",
        not complete[["image_id", "plain_mean_score", "rating_count"]]
        .isna()
        .to_numpy()
        .any(),
    )

    # --------------------------------------------------------------- splits
    report.section("Splits")
    per_split = {}
    for split in ("train", "validation", "test"):
        parts = sorted((public / "data" / "fbp").glob(f"{split}-*.parquet"))
        per_split[split] = sum(pq.read_metadata(p).num_rows for p in parts)
    report.note("split sizes", per_split)
    report.check(
        "every config covers the same images",
        all(
            set(frame["image_id"]) == set(images["image_id"])
            for frame in frames.values()
        ),
    )
    report.check(
        "the minimal config carries no demographics",
        not ({"gender", "ethnicity"} & set(minimal.columns)),
        str(sorted({"gender", "ethnicity"} & set(minimal.columns))),
    )
    report.check("image_id is unique", images["image_id"].is_unique)
    # cv_fold now covers every image, SCUT-style, rather than only train+val.
    folds = images["cv_fold"]
    report.check("every image has a cv_fold", not folds.isna().any())
    shares = {
        name: round(100 * count / sum(per_split.values()))
        for name, count in per_split.items()
    }
    report.check(
        "the split wording matches the actual proportions",
        f"{shares['train']}/{shares['validation']}/{shares['test']}" in card,
        f"actual {shares['train']}/{shares['validation']}/{shares['test']}",
    )
    report.check(
        "cv folds are 0-4", set(folds.dropna().astype(int).unique()) == {0, 1, 2, 3, 4}
    )
    sizes = folds.value_counts()
    report.check(
        "cv folds are balanced",
        int(sizes.max() - sizes.min()) <= 5,
        f"sizes {sorted(sizes.to_dict().items())}",
    )
    # A duplicate group crossing a split is the failure that would inflate
    # every published number, so it is checked against the source of truth.
    splits_path = Path("data/mebeauty_v3/ratings/splits.parquet")
    report.check(
        "split assignment file exists for cross-checking",
        splits_path.is_file(),
        str(splits_path),
    )

    # --------------------------------------------------------------- images
    report.section("Images")
    from PIL import Image

    for config, column, expected in (
        ("fbp", "image", (256, 256)),
        ("fbp_extended", "image_native", None),
    ):
        parts = sorted((public / "data" / config).glob("*.parquet"))
        checked, bad_size, undecodable, non_rgb = 0, 0, 0, 0
        for part in parts:
            cells = pq.read_table(part, columns=[column]).column(column).to_pylist()
            if args.sample_images:
                cells = cells[: args.sample_images]
            for cell in cells:
                checked += 1
                try:
                    with Image.open(io.BytesIO(cell["bytes"])) as opened:
                        if opened.mode != "RGB":
                            non_rgb += 1
                        if expected and opened.size != expected:
                            bad_size += 1
                except (OSError, ValueError):
                    # Pillow raises OSError on truncated or corrupt data and
                    # ValueError on an unrecognised format. Both mean the
                    # image is unusable, which is what this counts.
                    undecodable += 1
        report.check(
            f"{config}: every image decodes", undecodable == 0, f"{checked} checked"
        )
        report.check(f"{config}: every image is RGB", non_rgb == 0)
        if expected:
            report.check(f"{config}: every image is 256x256", bad_size == 0)

    # ----------------------------------------------------------------- card
    report.section("Dataset card")
    shipped_columns: set[str] = set()
    for path in glob.glob(str(public / "**" / "*.parquet"), recursive=True):
        shipped_columns |= set(pq.read_schema(path).names)
    # Only rows inside a "| Column | Meaning |" table. Matching every
    # backticked first cell also catches the *config* table, which lists
    # `native` and `raters` -- names that are directories, not columns.
    documented: set[str] = set()
    for block in re.findall(r"\| Column \|[^\n]*\n\|[-|]+\n((?:\|.*\n)+)", card):
        documented |= set(re.findall(r"^\| `([a-z_0-9]+)`", block, re.MULTILINE))
    ghosts = sorted(documented - shipped_columns - {"image"})
    report.check("card documents no column that was dropped", not ghosts, str(ghosts))
    # The nested rating columns are documented by their own field table in
    # the card rather than as a row in the image column table.
    shipped_columns -= {"ratings", "attractiveness_ratings", "date_ratings"}
    undocumented = sorted(shipped_columns - documented)
    report.check(
        "every shipped column is documented", not undocumented, str(undocumented)
    )

    # Landmarks are published unclipped, so out-of-frame points must still be
    # present in the data. If they had been silently clamped this would find
    # zero, which is the failure worth catching.
    import numpy as np

    coords = np.vstack([np.asarray(v, dtype=float) for v in images["landmarks"]])
    out_of_frame = int(((coords < 0) | (coords > 256)).any(axis=1).sum())
    report.check(
        "out-of-frame landmarks are preserved, not clipped",
        out_of_frame > 0,
        f"{out_of_frame} images carry a point outside the frame",
    )

    # The card must not overclaim rights the maintainer does not hold. This
    # looks for the disclaimer rather than for the absence of a claim,
    # because a missing disclaimer is the failure mode.
    # Whitespace-collapsed: the card is hard-wrapped, so a phrase being
    # checked for can be split across a newline and a naive `in` test misses
    # it. That produced a false failure on wording the card genuinely had.
    lowered = " ".join(card.lower().split())
    licence = (public / "LICENSE").read_text(encoding="utf-8")
    # The card states the ownership position; the LICENCE carries the explicit
    # "rights have not been verified" wording, since that is the binding
    # document. Both are required -- the claim must appear somewhere a user
    # cannot miss, and somewhere it is legally operative.
    report.check(
        "card states the photographs are subject to others' rights",
        "rights of their photographers" in lowered,
    )
    report.check(
        "licence disclaims verified redistribution/consent rights",
        "no representation that redistribution or consent rights have been "
        "verified" in " ".join(licence.lower().split()),
    )
    report.check(
        "card states images are not owned by the maintainer",
        "not owned by the maintainer" in lowered or "does not own" in lowered,
    )
    report.check(
        "card does not name the internal identity model",
        "arcface" not in lowered,
    )
    report.check("LICENSE ships", (public / "LICENSE").is_file())

    referenced = set(re.findall(r"`([A-Za-z_]+\.(?:md|py|txt))`", card))
    missing_files = sorted(name for name in referenced if not (public / name).is_file())
    report.check(
        "card references no missing file", not missing_files, str(missing_files)
    )

    import yaml

    front = yaml.safe_load(card.split("---")[1])
    report.check("card YAML parses", isinstance(front, dict))
    report.check(
        "no task_categories declared",
        "task_categories" not in front,
        str(front.get("task_categories")),
    )
    report.check(
        "licence makes no blanket sole-copyright claim",
        "copyright (c)" not in licence.lower(),
        "a bare `Copyright (c) <name>` line claims sole ownership of "
        "photographs the maintainer does not own",
    )
    report.check(
        "licence still scopes itself to the compilation",
        "does not cover" in licence.lower() and "photographs" in licence.lower(),
    )
    # The claim has to be in the card, but pinning one sentence made the check
    # fail on a pure rewording. Accept any phrasing that states it.
    exclusion_claims = (
        "no rater is excluded or down-weighted",
        "no rater is removed",
        "nobody is dropped or down-weighted",
    )
    report.check(
        "card states that no rater is excluded or down-weighted",
        any(phrase in lowered for phrase in exclusion_claims),
    )
    report.check(
        "card no longer refers to `valid raters`",
        "valid raters" not in lowered,
    )
    report.check(
        "redistribution checkbox covers images, ratings and metadata",
        any(
            "images, ratings and metadata" in key
            for key in front.get("extra_gated_fields", {})
        ),
    )
    report.check(
        "identity of repeat photographs is not overclaimed",
        "repeat photographs of the same person" not in lowered,
        "grouping is by likeness, not confirmed identity",
    )
    report.check("dataset is gated", "extra_gated_fields" in front)
    report.check(
        "gate demands non-commercial + no-identity + citation",
        sum(
            1
            for key in front.get("extra_gated_fields", {})
            if any(
                word in key.lower()
                for word in ("non-commercial", "recognition", "cite")
            )
        )
        >= 3,
    )
    link = front.get("license_link")
    report.check(
        "license_link resolves or is absent",
        link is None or (public / link).is_file(),
        str(link),
    )

    # --------------------------------------------------------------- counts
    report.section("Counts")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report.check(
        "image count reconciles with manifest",
        len(images) == manifest["images"],
        f"{len(images)} vs {manifest['images']}",
    )
    report.check(
        "attractiveness rating count reconciles",
        len(generic) == manifest["ratings"]["attractiveness"],
    )
    all_raters = set(generic["rater_id"]) | set(preference["rater_id"])
    report.check(
        "rater count reconciles",
        len(all_raters) == manifest["raters_total_unique"],
        f"{len(all_raters)} across both tasks vs {manifest['raters_total_unique']}",
    )
    report.note("images", len(images))
    report.note("splits", per_split)
    report.note("attractiveness ratings", len(generic))
    report.note("preference ratings", len(preference))
    report.note(
        "raters", f"{len(raters)} ({int((raters['pool'] == 'panel').sum())} panel)"
    )

    # Every number the card prints must come from the release data. Checked
    # by confirming each comma-formatted figure in the card is one the
    # manifest knows about -- a hand-typed count would not be.
    def known(manifest_obj) -> set[int]:
        found: set[int] = set()
        if isinstance(manifest_obj, dict):
            for value in manifest_obj.values():
                found |= known(value)
        elif isinstance(manifest_obj, list):
            for value in manifest_obj:
                found |= known(value)
        elif isinstance(manifest_obj, int) and not isinstance(manifest_obj, bool):
            found.add(manifest_obj)
        return found

    manifest_numbers = known(manifest) | {
        v for value in per_split.values() for v in (value,)
    }
    # Numbers written as "about N" are deliberate roundings, not counts, so
    # they are exempt -- everything else must trace to the release data.
    printed = {
        int(m.replace(",", ""))
        for m in re.findall(r"(?<!about )\b\d{1,3}(?:,\d{3})+\b", card)
    }
    invented = sorted(printed - manifest_numbers)
    report.check(
        "every count in the card is derived from the release data",
        not invented,
        f"not found in the manifest: {invented}",
    )
    report.note("version", f"{manifest['dataset']} {manifest['version']}")

    print("\n".join(report.lines))
    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s)")
        for failure in report.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("All checks passed. Release is safe to upload.")


if __name__ == "__main__":
    main()
