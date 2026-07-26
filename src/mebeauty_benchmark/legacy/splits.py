"""Load and de-duplicate the legacy train/val/test rating splits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mebeauty_benchmark.legacy.paths import normalize_image_path


@dataclass(frozen=True)
class Rating:
    path: str
    score: float


def parse_split_lines(lines: Iterable[str]) -> list[Rating]:
    """Parse ``"<path> <score>"`` lines (score is always the last token).

    A handful of lines quote the path (`"foo (1).jpg" 5.5`) because the
    filename itself contains a space. Unquoted, the leading `"` becomes
    part of the path string and every downstream prefix check silently
    fails to match it -- the row survives parsing but can never be joined
    back to an actual file. Strip a matching pair of surrounding quotes
    before returning.
    """
    ratings: list[Rating] = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        path, _, score = line.rpartition(" ")
        if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
            path = path[1:-1]
        ratings.append(Rating(path=path, score=float(score)))
    return ratings


def resolve_to_existing_images(
    splits: dict[str, list[Rating]],
    valid_paths: set[str],
    basename_to_paths: dict[str, list[str]],
) -> tuple[dict[str, list[Rating]], list[dict[str, str]]]:
    """Drop or remap rating rows whose rated path no longer holds a file.

    Splits are parsed from raw text and never checked against the actual
    `original_images/` tree -- a rating can reference a file that has since
    been reclassified into a different ethnicity/gender folder (still
    exists, wrong path) or removed outright (doesn't exist anywhere).
    Neither `dedupe_across_splits` nor `dedupe_by_content_hash` catch this:
    both operate on whatever path string is already in the row.

    `valid_paths` is every current normalized path under `original_images/`.
    `basename_to_paths` maps each lowercased basename to every current path
    ending in that basename (usually one; more than one means the basename
    itself is ambiguous, e.g. Finding 8's label collisions).

    - Path still valid: row kept, path normalized.
    - Path invalid but basename resolves to exactly one current path: row
      kept, remapped to that path (a relabeling, not a data loss).
    - Basename resolves to more than one current path: dropped -- picking
      one would silently guess which relabeling is correct.
    - Basename matches nothing: dropped -- no image to join this rating to.

    Returns the resolved splits and a list of every non-trivial outcome
    (remapped, ambiguous, or dropped) for the audit trail. Rows that were
    already valid are not included in that list.
    """
    resolved: dict[str, list[Rating]] = {name: [] for name in splits}
    issues: list[dict[str, str]] = []

    for split_name, ratings in splits.items():
        for rating in ratings:
            normalized = normalize_image_path(rating.path)
            if normalized in valid_paths:
                resolved[split_name].append(Rating(path=normalized, score=rating.score))
                continue

            basename = normalized.rsplit("/", 1)[-1].lower()
            candidates = basename_to_paths.get(basename, [])
            if len(candidates) == 1:
                resolved[split_name].append(
                    Rating(path=candidates[0], score=rating.score)
                )
                issues.append(
                    {
                        "split": split_name,
                        "rated_path": normalized,
                        "status": "relabeled",
                        "resolved_path": candidates[0],
                    }
                )
            elif len(candidates) > 1:
                issues.append(
                    {
                        "split": split_name,
                        "rated_path": normalized,
                        "status": "ambiguous",
                        "candidates": ", ".join(candidates),
                    }
                )
            else:
                truncated = _resolve_truncated_basename(
                    normalized, basename_to_paths, valid_paths
                )
                if truncated is not None:
                    resolved[split_name].append(
                        Rating(path=truncated, score=rating.score)
                    )
                    issues.append(
                        {
                            "split": split_name,
                            "rated_path": normalized,
                            "status": "truncated_filename",
                            "resolved_path": truncated,
                        }
                    )
                else:
                    issues.append(
                        {
                            "split": split_name,
                            "rated_path": normalized,
                            "status": "not_found",
                        }
                    )

    return resolved, issues


# Some legacy filenames were truncated to exactly this many characters
# (stem only, extension preserved) somewhere in the original dataset's
# creation, e.g. "payton-tuttle-n_RdRxH_7h4-unsplash.jpg" is stored as
# "payton-tuttle-n_RdRxH_7h4-unspla.jpg". The score files kept the full
# name, so those rows silently failed to join and were counted as
# "not_found" -- a rated image dropped for a filename bug, not because the
# image was gone. See docs/DATASET_AUDIT.md Finding 15.
_TRUNCATED_STEM_LENGTH = 32


def _resolve_truncated_basename(
    normalized: str,
    basename_to_paths: dict[str, list[str]],
    valid_paths: set[str],
) -> str | None:
    """Match a rated path against a file stored under a truncated filename.

    Only resolves when the answer is unambiguous, because a wrong match here
    would attribute one photograph's rating to a different photograph:

    - the rated stem must actually be longer than the truncation length
      (otherwise nothing was truncated and this does not apply);
    - exactly one existing file may match the truncated stem;
    - a file under the *full* untruncated name must not also exist -- if it
      does, the rating belongs to that file, and the truncated file is a
      different image that merely shares a prefix. This dataset has one such
      case (`jonathan-borba-5rQG1mib90I-unspl.jpg` vs `...-unsplash.jpg`,
      different SHA-256), which stays unresolved by design.
    """
    basename = normalized.rsplit("/", 1)[-1]
    stem, _, extension = basename.rpartition(".")
    if not extension or len(stem) <= _TRUNCATED_STEM_LENGTH:
        return None

    if basename_to_paths.get(basename.lower()):
        return None  # the full-named file exists; the rating is that file's

    candidate_basename = f"{stem[:_TRUNCATED_STEM_LENGTH]}.{extension}".lower()
    candidates = basename_to_paths.get(candidate_basename, [])
    if len(candidates) != 1 or candidates[0] not in valid_paths:
        return None
    return candidates[0]


def dedupe_across_splits(
    splits: dict[str, list[Rating]], priority: list[str]
) -> tuple[dict[str, list[Rating]], list[tuple[str, str]]]:
    """Keep exactly one row per *normalized full path*, anywhere in `splits`.

    Keys on the full normalized path (folder + filename), not the bare
    basename. This dataset has one confirmed case of two genuinely
    different photos sharing an identical filename in different folders
    (`male/indian/shivam-singh-2_X6NMP-E_U-unsplash.jpg` vs
    `male/mideastern/...`) -- a basename-only key would treat them as the
    same image and silently drop one. Cross-folder duplicates that are the
    *same content* under a *different* filename (the far more common case
    here -- 7 of the 8 basename collisions found) are a separate concern,
    handled by `dedupe_by_content_hash`.

    `priority` lists split names from highest to lowest priority. Rows are
    walked in priority order, then file order within each split; the
    first occurrence of a path wins and every later occurrence is
    dropped, whether it repeats within the same split (a duplicated row)
    or leaks into another split. Returns the deduped splits and a
    ``(normalized_path, dropped_from)`` list for the audit trail.
    """
    seen_paths: set[str] = set()
    deduped: dict[str, list[Rating]] = {name: [] for name in splits}
    removed: list[tuple[str, str]] = []

    for split_name in priority:
        for rating in splits[split_name]:
            normalized = normalize_image_path(rating.path)
            if normalized in seen_paths:
                removed.append((normalized, split_name))
                continue
            seen_paths.add(normalized)
            deduped[split_name].append(rating)

    return deduped, removed


def dedupe_by_content_hash(
    splits: dict[str, list[Rating]],
    priority: list[str],
    path_to_hash: dict[str, str],
) -> tuple[dict[str, list[Rating]], list[tuple[str, str, str]]]:
    """Keep exactly one row per distinct image content (by SHA-256), across all splits.

    Complements `dedupe_across_splits`, which only catches duplicate
    *filenames*. Two rows referencing byte-identical images saved under
    different filenames are invisible to that check and can silently put
    the same photo in both train and test. Rows whose path is missing
    from `path_to_hash` are left untouched (never merged with anything).
    Returns the deduped splits and a ``(path, sha256, dropped_from_split)``
    audit trail.
    """
    seen_hashes: set[str] = set()
    deduped: dict[str, list[Rating]] = {name: [] for name in splits}
    removed: list[tuple[str, str, str]] = []

    for split_name in priority:
        for rating in splits[split_name]:
            content_hash = path_to_hash.get(rating.path)
            if content_hash is None:
                deduped[split_name].append(rating)
                continue
            if content_hash in seen_hashes:
                removed.append((rating.path, content_hash, split_name))
                continue
            seen_hashes.add(content_hash)
            deduped[split_name].append(rating)

    return deduped, removed
