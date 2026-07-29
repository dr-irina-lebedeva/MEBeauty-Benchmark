"""Replace inferred source links with the maintainer's documented link list.

Until now every source URL in this dataset was **reconstructed from the
filename** -- the legacy repository records no provenance at all (verified
against the live repo, not just the snapshot). Those reconstructions were
right about *which platform* but frequently wrong about the URL: Pixabay's
real slug is a description, not the filename, so
`pixabay.com/photos/male-4572748_1920/` was a guess that does not resolve.
Two of the three links ever hand-checked were dead for exactly this reason.

The maintainer's spreadsheet is an actual record, and it is better on both
counts: it carries the real slugs, and it covers images whose filenames are
bare numbers (`1.jpg`) where no pattern could recover anything.

**It is not treated as infallible.** Cross-checking every row against the
filename-derived link found the spreadsheet also contains truncated Unsplash
ids and a few malformed URLs, so each image is classified rather than
overwritten blindly:

- the two sources name the same photo -> take the documented URL, which has
  the better slug;
- only the spreadsheet has anything -> take it;
- they name *different* photos -> keep both and flag `link_conflict`, since a
  provenance record is exactly the thing that must not be silently guessed.

Matching tries the primary filename, then the alternate filenames of
byte-identical duplicates, then the 32-character truncation of Finding 15 --
each of which recovers images a plain filename join misses.

    uv run python scripts/data/apply_documented_links.py \\
        --v3 data/mebeauty_v3 \\
        --links ~/Downloads/image-link.xlsx \\
        --report-out reports/legacy_audit/documented_links.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

#: Localised Pixabay paths (`pixabay.com/nl/photos/...`) resolve to the same
#: photo as the canonical `.com/photos/...`. They are normalised so links are
#: uniform, and the untouched original is kept in `source_url_as_recorded`.
LOCALE_PATH = re.compile(r"(pixabay\.com)/[a-z]{2}/(photos/)")

#: A few recorded URLs are missing the slash after `/photos`.
MISSING_SLASH = re.compile(r"(unsplash\.com/photos)(?=[A-Za-z0-9_-])")

#: Links the maintainer supplied directly for images the spreadsheet does not
#: cover (2026-07-28). Keyed by legacy path, and treated as `documented` --
#: they come from the same person and the same collection records as the
#: spreadsheet itself, just typed in later.
MANUAL_LINKS = {
    "female/caucasian/f3.png": "https://www.pexels.com/photo/woman-taking-selfie-59552/",
    "male/black/pexels-teddy-joseph-2955375.jpg": (
        "https://www.pexels.com/photo/man-wearing-black-notched-lapel-blazer-2955375/"
    ),
    # Looked up by the maintainer while resolving the conflicts below. Note the
    # canonical Unsplash form: `/photos/{description-slug}-{id}`. The slug is
    # platform-generated alt text and cannot be derived offline, so every other
    # Unsplash link here is the bare-id form, which redirects to it.
    "female/indian/arvin-keynes-IPETsB4dcCs-unsplash.jpg": (
        "https://unsplash.com/photos/woman-in-white-and-black-striped-pullover-"
        "top-covered-with-smoke-IPETsB4dcCs"
    ),
    # Resolves a spreadsheet/filename conflict in the filename's favour again:
    # this page is id 1284347 (the filename), not the recorded 1246224. Kept on
    # its Hungarian locale path exactly as found -- see below.
    "female/mideastern/woman-1284347_1920.jpg": (
        "https://pixabay.com/hu/photos/nő-arc-mellbőség-fej-smink-1284347/"
    ),
}

#: Where the spreadsheet and the filename disagree about an **Unsplash** photo,
#: the filename wins.
#:
#: This is not a preference, it is Unsplash's own convention: their download
#: filenames are `photographer-name-{photo_id}-unsplash.jpg`, so the id is
#: carried by the file itself. Checked across all 10 Unsplash conflicts here --
#: the filename-derived id appears verbatim in the filename every time, while
#: the spreadsheet sometimes holds an unrelated id, presumably a row that
#: slipped while it was compiled. Confirmed on one case by the maintainer
#: looking it up: for `arvin-keynes-IPETsB4dcCs`, the live page is
#: `...-IPETsB4dcCs` (the filename), not the recorded `dRatGbY1k8Y`.
#:
#: Pixabay and Pexels conflicts are *not* resolved this way. Their filenames
#: carry a numeric id too, but the recorded links there differ in ways
#: (5090230 vs 5009607) that a naming convention cannot adjudicate.
PREFER_FILENAME_FOR_UNSPLASH = True

#: Pixabay/Pexels conflicts the maintainer could not resolve by looking them up
#: (2026-07-28), decided in the filename's favour. That is the weaker of the two
#: justifications used here -- unlike Unsplash there is no naming convention
#: guaranteeing it -- but the filename id had already proved correct in every
#: conflict that *was* resolved, so it is the better bet of the two.
#:
#: The URL emitted is the **bare-id** form, not the filename-derived slug. The
#: reconstructed slug (`/photos/hijab-5090230_1920/`) is exactly the malformed
#: pattern that made earlier hand-checked links dead; both platforms accept a
#: bare id and redirect to the real page.
#:
#: These stay `inferred`, never `documented`: the id is well-founded, but
#: nobody has confirmed the page opens.
FILENAME_ID_WINS = {
    "female/asian/hijab-5090230_1920.jpg": "https://pixabay.com/photos/5090230/",
    "female/caucasian/woman-3718859_1920.jpg": "https://pixabay.com/photos/3718859/",
    "male/mideastern/pexels-emre-keshavarz-3518392.jpg": (
        "https://www.pexels.com/photo/3518392/"
    ),
    # Not in the spreadsheet and not findable; same bare-id treatment.
    "male/hispanic/couple-5917009_1920.png": "https://pixabay.com/photos/5917009/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, help="data/mebeauty_v3 directory")
    parser.add_argument(
        "--links", required=True, help="Spreadsheet of documented links"
    )
    parser.add_argument("--report-out", required=True, help="JSON report path")
    return parser.parse_args()


def normalise(url: str) -> str:
    """Repair the recorded URL's form without changing which photo it names."""
    fixed = MISSING_SLASH.sub(r"\1/", url.strip())
    return LOCALE_PATH.sub(r"\1/\2", fixed)


#: Unsplash photo ids are a fixed 11 characters and may themselves contain
#: `-` or `_`, so the id is the last 11 characters of the final path segment
#: -- never the last hyphen-delimited piece. Both URL forms then compare
#: equal: the canonical `/photos/{slug}-{id}` and the bare `/photos/{id}`.
#: (`provenance.py` was fixed for this same trap; 195 links were wrong.)
UNSPLASH_ID_LENGTH = 11


def photo_identity(url: str) -> tuple[str, set[str]]:
    """The photo a URL names: (unsplash id, numeric ids)."""
    trimmed = url.rstrip("/")
    unsplash = ""
    if "unsplash.com" in trimmed:
        segment = trimmed.split("/")[-1]
        unsplash = segment[-UNSPLASH_ID_LENGTH:]
    return unsplash, set(re.findall(r"\d{4,}", trimmed))


def same_photo(a: str, b: str) -> bool | None:
    """True/False if the two URLs are comparable, None if they are not."""
    ua, na = photo_identity(a)
    ub, nb = photo_identity(b)
    if ua and ub:
        # Unsplash ids are a fixed 11 characters; a prefix match means one
        # side is truncated, not that they are different photographs.
        # Equal ids, or one side truncated (a recorded id shorter than 11).
        return ua == ub or ua.endswith(ub) or ub.endswith(ua)
    if na and nb:
        return bool(na & nb)
    return None


def candidate_names(row) -> list[str]:
    names = [row.legacy_filename.lower()]
    if row.other_legacy_paths:
        names += [
            path.rsplit("/", 1)[-1].lower()
            for path in row.other_legacy_paths.split(", ")
        ]
    return names


def main() -> None:
    args = parse_args()
    v3_dir = Path(args.v3).expanduser().resolve()
    sheet = pd.read_excel(Path(args.links).expanduser(), sheet_name=0)
    sheet["image"] = sheet["image"].astype(str).str.strip().str.lower()
    sheet["link"] = sheet["link"].astype(str).str.strip()
    recorded = dict(zip(sheet["image"], sheet["link"]))
    print(f"{len(sheet)} recorded links, {len(recorded)} unique filenames")

    metadata = pd.read_parquet(v3_dir / "images" / "metadata.parquet")

    def lookup(row) -> tuple[str | None, str]:
        if row.legacy_path in FILENAME_ID_WINS:
            return FILENAME_ID_WINS[row.legacy_path], "filename id (bare)"
        if row.legacy_path in MANUAL_LINKS:
            return MANUAL_LINKS[row.legacy_path], "maintainer"
        names = candidate_names(row)
        for index, name in enumerate(names):
            if name in recorded:
                return recorded[
                    name
                ], "filename" if index == 0 else "duplicate filename"
        # Finding 15: five filenames were truncated to exactly 32 characters
        # at collection time while the record kept the full name.
        stem = names[0].rsplit(".", 1)[0]
        if len(stem) >= 30:
            hits = {
                value
                for key, value in recorded.items()
                if key.rsplit(".", 1)[0].startswith(stem[:32])
            }
            if len(hits) == 1:
                return hits.pop(), "truncated filename"
        return None, "unmatched"

    urls, methods, conflicts, as_recorded = [], [], [], []
    counts = {
        "identical": 0,
        "better_slug": 0,
        "new": 0,
        "filename_wins": 0,
        "conflict": 0,
        "none": 0,
    }

    for row in metadata.itertuples():
        found, method = lookup(row)
        inferred = row.inferred_source_url or ""
        if found is None:
            # Nothing recorded: keep whatever was inferred, or nothing at all.
            urls.append(inferred)
            as_recorded.append("")
            methods.append("inferred" if inferred else "none")
            conflicts.append(False)
            counts["none"] += 1
            continue

        # A link the maintainer looked up and confirmed is left exactly as
        # given. Normalisation exists to repair the *spreadsheet's* recording
        # errors; applying it to a verified URL would only risk breaking one.
        cleaned = found if method == "maintainer" else normalise(found)
        as_recorded.append(found if found != cleaned else "")
        if not inferred:
            counts["new"] += 1
            conflicts.append(False)
        else:
            verdict = same_photo(cleaned, inferred)
            if verdict is False:
                if PREFER_FILENAME_FOR_UNSPLASH and "unsplash.com" in inferred:
                    # The filename carries the id by Unsplash's own convention.
                    cleaned = inferred
                    method = "filename (unsplash id)"
                    counts["filename_wins"] += 1
                    conflicts.append(False)
                else:
                    counts["conflict"] += 1
                    conflicts.append(True)
            else:
                counts["identical" if cleaned == inferred else "better_slug"] += 1
                conflicts.append(False)
        urls.append(cleaned)
        methods.append(method)

    metadata["source_url"] = urls
    metadata["source_url_as_recorded"] = as_recorded
    metadata["source_link_method"] = methods
    metadata["link_conflict"] = conflicts
    metadata["provenance_confidence"] = [
        # "filename id (bare)" is deliberately NOT documented: the id is
        # well-founded but no one has confirmed the page opens, and claiming
        # otherwise would put a false assurance in the shipped metadata.
        "inferred"
        if m == "filename id (bare)"
        else (
            "documented"
            if m.startswith("filename")
            or m in {"duplicate filename", "truncated filename", "maintainer"}
            else ("inferred" if u else "unknown")
        )
        for m, u in zip(methods, urls)
    ]
    metadata.to_parquet(v3_dir / "images" / "metadata.parquet", index=False)

    documented = int((metadata["provenance_confidence"] == "documented").sum())
    report = {
        "source": Path(args.links).name,
        "note": (
            "The legacy repository records no image provenance at all; every "
            "prior URL was reconstructed from the filename. These are the "
            "maintainer's documented links."
        ),
        "images": len(metadata),
        "documented": documented,
        "still_inferred": int((metadata["provenance_confidence"] == "inferred").sum()),
        "no_link": int((metadata["provenance_confidence"] == "unknown").sum()),
        "agreement_with_inferred": counts,
        "normalised": {
            "locale_paths_and_malformed": int(
                (metadata["source_url_as_recorded"] != "").sum()
            )
        },
        "match_method": metadata["source_link_method"].value_counts().to_dict(),
        "conflicts": [
            {
                "legacy_filename": r.legacy_filename,
                "documented": r.source_url,
                "inferred": r.inferred_source_url,
            }
            for r in metadata[metadata["link_conflict"]].itertuples()
        ],
    }
    report_path = Path(args.report_out).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"  documented : {documented} of {len(metadata)}")
    print(f"  no link    : {report['no_link']}")
    print(f"  conflicts  : {len(report['conflicts'])} (flagged, both kept)")
    print(f"  agreement  : {counts}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
