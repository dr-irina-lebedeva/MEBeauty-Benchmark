"""The in-house rater panel: identity, demographics, and ID canonicalization.

Alongside the anonymous Mechanical Turk pool, the legacy collection used a
small in-house panel whose identifiers encode the rater's own demographics --
`caucasian_female_29` is a 29-year-old Caucasian woman. That is genuinely
useful for personalization research (rater demographics are exactly what a
personalized model needs) and it is also why the raw form must not ship: a
free-text demographic string is a far more identifying label than an opaque
pseudonym, especially for a panel this small.

So the panel is handled in two parts: a stable opaque `panel_XXXX` id, and a
separate structured demographics table. Consumers get the research value
without the identifier carrying it.

**The same panel member appears under four different spellings** across the
legacy files, which is why they were never reconciled before:

| Shape | Example | Where |
|---|---|---|
| full   | `caucasian_female_29` | `private_*/` filenames |
| label- | `labelcaucasian_female_29.xlsx` | `date_scores_all.xlsx` columns |
| short  | `cf29` | `generic_scores_all.xlsx` columns |
| gender-prefixed short | `femalecf29` | `generic_scores_all_2022.xlsx` |

`canonical_panel_id` maps all four onto the full form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Ethnicity tokens used by the panel identifiers, and the initial each is
#: abbreviated to in the short forms.
ETHNICITIES = ("asian", "black", "caucasian", "hispanic", "mideast")

GENDERS = ("female", "male")

_FULL_RE = re.compile(
    rf"^(?P<ethnicity>{'|'.join(ETHNICITIES)})_(?P<gender>{'|'.join(GENDERS)})"
    r"_(?P<age>\d{1,3})(?P<session>_\d+)?$"
)

#: Short form: ethnicity initial + gender initial + age, optional session
#: suffix, e.g. `cf34_2`. Ambiguous by construction -- `m` could be `mideast`
#: -- so it is resolved against the known full-form roster, never guessed.
_SHORT_RE = re.compile(r"^(?P<initials>[a-z]{2})(?P<age>\d{1,3})(?P<session>_\d+)?$")

_PREFIXES = ("label",) + GENDERS


@dataclass(frozen=True)
class PanelRater:
    """One in-house panel member."""

    canonical_id: str
    ethnicity: str
    gender: str
    age: int
    session: int

    @property
    def short_id(self) -> str:
        base = f"{self.ethnicity[0]}{self.gender[0]}{self.age}"
        return base if self.session == 1 else f"{base}_{self.session}"


def parse_panel_id(name: str) -> PanelRater | None:
    """Parse a full-form panel identifier, or return None if it isn't one."""
    match = _FULL_RE.match(_strip_decoration(name))
    if match is None:
        return None
    session = match.group("session")
    return PanelRater(
        canonical_id=_strip_decoration(name),
        ethnicity=match.group("ethnicity"),
        gender=match.group("gender"),
        age=int(match.group("age")),
        session=int(session.lstrip("_")) if session else 1,
    )


def _strip_decoration(name: str) -> str:
    """Remove the wrappers the legacy files add around a panel id."""
    cleaned = str(name).strip().lower()
    cleaned = cleaned.removesuffix(".xlsx")
    # `caucasian_female_34_2..xlsx` leaves a trailing dot once the suffix goes.
    cleaned = cleaned.rstrip(".")
    for prefix in _PREFIXES:
        if not cleaned.startswith(prefix):
            continue
        remainder = cleaned[len(prefix) :]
        # Strip the wrapper only if what's left is itself a panel id in either
        # shape -- `femalecf29` is gender-prefixed *short* form, not full.
        if _FULL_RE.match(remainder) or _SHORT_RE.match(remainder):
            return remainder
    return cleaned


def canonical_panel_id(name: str, roster: dict[str, PanelRater]) -> str | None:
    """Resolve any of the four spellings to a canonical panel id.

    `roster` maps canonical id -> PanelRater, built from the authoritative
    `private_*/` filenames. Short forms are resolved by lookup against it
    rather than by expanding initials, because `m` is ambiguous between
    `mideast` and nothing else in the ethnicity list while `b`/`c`/`h`/`a`
    are unique -- guessing would silently invent a rater.
    """
    cleaned = _strip_decoration(name)
    if cleaned in roster:
        return cleaned

    match = _SHORT_RE.match(cleaned)
    if match is None:
        return None

    initials, age = match.group("initials"), int(match.group("age"))
    session = match.group("session")
    session_number = int(session.lstrip("_")) if session else 1
    candidates = [
        rater
        for rater in roster.values()
        if rater.age == age
        and rater.session == session_number
        and rater.ethnicity[0] == initials[0]
        and rater.gender[0] == initials[1]
    ]
    # Exactly one match, or nothing. Two panel members sharing an initial,
    # gender and age would make the short form genuinely ambiguous, and
    # picking one would attribute real ratings to the wrong person.
    return candidates[0].canonical_id if len(candidates) == 1 else None


def build_roster(names: list[str]) -> dict[str, PanelRater]:
    """Build the canonical roster from authoritative full-form identifiers."""
    roster: dict[str, PanelRater] = {}
    for name in names:
        rater = parse_panel_id(name)
        if rater is not None:
            roster[rater.canonical_id] = rater
    return roster


def assign_pseudonyms(roster: dict[str, PanelRater]) -> dict[str, str]:
    """canonical id -> stable opaque `panel_XXXX`, sorted so it never drifts."""
    return {
        canonical: f"panel_{index:04d}"
        for index, canonical in enumerate(sorted(roster), start=1)
    }
