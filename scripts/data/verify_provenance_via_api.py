"""Turn *inferred* provenance into *verified* provenance using the official
platform APIs, and write canonical URLs that actually resolve.

Why this exists: every source link in this repository is reconstructed from a
filename pattern and has never been confirmed. Two limitations were proven
directly (docs/DATASET_AUDIT.md, Finding 11):

1. **The generated Unsplash URL is not canonical.** Unsplash's real URL is
   `/photos/{description-slug}-{photo-id}`, e.g.
   `unsplash.com/photos/man-in-black-crew-neck-shirt-gisFZKWpKQ4`. The slug is
   platform-generated alt text, present nowhere in the filename
   (`dorrell-tibbs-gisFZKWpKQ4-unsplash.jpg`), so it cannot be derived
   offline -- only fetched. Scraping is blocked (HTTP 401 bot challenge) and
   the public oEmbed endpoint now answers "Authorization required".
2. **Inferred links go stale.** Of three Pixabay links opened by hand, two no
   longer resolved -- the photos appear to have been removed from the
   platform since collection.

The official APIs solve both, and give more than a tidy URL: the real
photographer, the licence in force, and -- critically -- whether the photo
still exists at all. That converts the largest open rights question in this
project from "inferred, unverified" to evidence, for the ~68% of images on
platforms with a free API.

**Credentials are read from the environment, never stored in the repository
and never written to any output file.**

    export UNSPLASH_ACCESS_KEY=...     # unsplash.com/developers (free)
    export PEXELS_API_KEY=...          # pexels.com/api (free)

    uv run --with requests python scripts/data/verify_provenance_via_api.py \\
        --metadata data/mebeauty_v3/images/metadata.parquet \\
        --output reports/legacy_audit/provenance_verification.json \\
        --platform unsplash --limit 25

Pixabay has no per-photo lookup by ID on its free API, so its 763 images
cannot be verified this way and stay `inferred`. That matters: Pixabay is
both the least reliable link format here *and* the channel that surfaced the
public-figure images (Finding 13).

Rate limits are real (Unsplash demo tier is 50 requests/hour), so `--limit`
and `--resume` exist to run this in batches across sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

_ENDPOINTS = {
    "unsplash": "https://api.unsplash.com/photos/{photo_id}",
    "pexels": "https://api.pexels.com/v1/photos/{photo_id}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="v3 metadata.parquet")
    parser.add_argument(
        "--platform",
        choices=sorted(_ENDPOINTS),
        required=True,
        help="Which platform's API to query (Pixabay has no by-ID lookup)",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many lookups (free tiers are rate-limited)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip photo ids already present in the output file",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds between requests, to stay inside the rate limit",
    )
    return parser.parse_args()


def auth_header(platform: str) -> dict[str, str]:
    if platform == "unsplash":
        key = os.environ.get("UNSPLASH_ACCESS_KEY")
        if not key:
            raise SystemExit(
                "UNSPLASH_ACCESS_KEY is not set. Get a free key at "
                "https://unsplash.com/developers and export it."
            )
        return {"Authorization": f"Client-ID {key}"}
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise SystemExit(
            "PEXELS_API_KEY is not set. Get a free key at https://pexels.com/api"
        )
    return {"Authorization": key}


def extract(platform: str, payload: dict) -> dict[str, str | None]:
    """Pull only the fields this project needs; ignore the rest of the response."""
    if platform == "unsplash":
        user = payload.get("user") or {}
        return {
            "canonical_url": (payload.get("links") or {}).get("html"),
            "photographer": user.get("name"),
            "photographer_username": user.get("username"),
            "license": "Unsplash License",
            "description": payload.get("description") or payload.get("alt_description"),
        }
    return {
        "canonical_url": payload.get("url"),
        "photographer": payload.get("photographer"),
        "photographer_username": None,
        "license": "Pexels License",
        "description": payload.get("alt"),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = auth_header(args.platform)

    metadata = pd.read_parquet(args.metadata)
    targets = metadata[
        (metadata["inferred_platform"] == args.platform)
        & metadata["inferred_photo_id"].notna()
    ][["image_id", "inferred_photo_id", "inferred_source_url"]]

    results: dict[str, dict] = {}
    if args.resume and output_path.is_file():
        results = json.loads(output_path.read_text(encoding="utf-8")).get("results", {})
        print(f"Resuming: {len(results)} already verified")

    pending = [r for r in targets.itertuples() if r.inferred_photo_id not in results]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(targets)} {args.platform} images; querying {len(pending)} now")

    session = requests.Session()
    for index, row in enumerate(pending, start=1):
        url = _ENDPOINTS[args.platform].format(photo_id=row.inferred_photo_id)
        try:
            response = session.get(url, headers=headers, timeout=30)
        except requests.RequestException as error:
            results[row.inferred_photo_id] = {
                "status": "request_failed",
                "error": str(error),
            }
            continue

        if response.status_code == 200:
            record = extract(args.platform, response.json())
            record["status"] = "verified"
            # The whole point: does the URL we generated actually match reality?
            record["generated_url_was_correct"] = (
                record["canonical_url"] == row.inferred_source_url
            )
        elif response.status_code == 404:
            # A real answer, not a failure: the photo is gone from the platform.
            record = {"status": "not_found_on_platform"}
        elif response.status_code in (403, 429):
            print(f"  rate-limited at {index}/{len(pending)} -- stopping, use --resume")
            break
        else:
            record = {"status": f"http_{response.status_code}"}

        record["image_id"] = row.image_id
        results[row.inferred_photo_id] = record
        if index % 10 == 0:
            print(f"  {index}/{len(pending)}")
        time.sleep(args.sleep)

    statuses: dict[str, int] = {}
    for record in results.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    verified = [r for r in results.values() if r["status"] == "verified"]
    url_correct = sum(1 for r in verified if r.get("generated_url_was_correct"))

    report = {
        "platform": args.platform,
        "total_images_on_platform": len(targets),
        "lookups_completed": len(results),
        "status_counts": statuses,
        "generated_url_matched_canonical": url_correct,
        "generated_url_mismatched": len(verified) - url_correct,
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nStatus counts: {statuses}")
    if verified:
        print(f"Generated URL matched the canonical one: {url_correct}/{len(verified)}")
    print(f"Wrote {output_path}")
    print(
        "\nNOTE: no API key is written to this report. Re-run with --resume to "
        "continue past a rate limit."
    )


if __name__ == "__main__":
    main()
