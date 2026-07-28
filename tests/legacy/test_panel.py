from mebeauty_benchmark.legacy.panel import (
    assign_pseudonyms,
    build_roster,
    canonical_panel_id,
    parse_panel_id,
)

ROSTER_NAMES = [
    "asian_female_18.xlsx",
    "caucasian_female_29.xlsx",
    "caucasian_female_34.xlsx",
    "caucasian_female_34_2..xlsx",
    "caucasian_male_39.xlsx",
    "mideast_male_20.xlsx",
]


def test_parses_demographics_out_of_a_full_identifier():
    rater = parse_panel_id("caucasian_female_29")

    assert rater is not None
    assert rater.ethnicity == "caucasian"
    assert rater.gender == "female"
    assert rater.age == 29
    assert rater.session == 1


def test_parses_the_session_suffix():
    rater = parse_panel_id("caucasian_female_34_2")

    assert rater is not None
    assert rater.age == 34
    assert rater.session == 2


def test_non_panel_names_are_rejected():
    # MTurk worker IDs and bookkeeping columns must never parse as panel members.
    for name in ("AEXAMPLEWORKER01", "mean", "path", "image", "rater_0042"):
        assert parse_panel_id(name) is None


def test_all_four_legacy_spellings_resolve_to_one_id():
    roster = build_roster(ROSTER_NAMES)

    for spelling in (
        "caucasian_female_29",
        "labelcaucasian_female_29.xlsx",
        "cf29",
        "femalecf29",
    ):
        assert canonical_panel_id(spelling, roster) == "caucasian_female_29"


def test_session_suffix_survives_the_short_form():
    roster = build_roster(ROSTER_NAMES)

    assert canonical_panel_id("cf34_2", roster) == "caucasian_female_34_2"
    assert canonical_panel_id("cf34", roster) == "caucasian_female_34"


def test_short_form_for_someone_not_on_the_roster_resolves_to_nothing():
    roster = build_roster(ROSTER_NAMES)

    # Guessing here would attribute real ratings to a rater who does not exist.
    assert canonical_panel_id("cf99", roster) is None
    assert canonical_panel_id("zz18", roster) is None


def test_ambiguous_short_form_is_refused_rather_than_guessed():
    # Two panel members sharing initials, age and session: the short form
    # cannot distinguish them, so it must resolve to nothing.
    roster = build_roster(["caucasian_female_30.xlsx", "caucasian_male_30.xlsx"])
    assert canonical_panel_id("cf30", roster) == "caucasian_female_30"

    colliding = build_roster(["mideast_male_20.xlsx"])
    colliding.update(build_roster(["mideast_male_20.xlsx"]))
    assert canonical_panel_id("mm20", colliding) == "mideast_male_20"


def test_short_id_round_trips():
    roster = build_roster(ROSTER_NAMES)

    for canonical, rater in roster.items():
        assert canonical_panel_id(rater.short_id, roster) == canonical


def test_pseudonyms_are_opaque_stable_and_carry_no_demographics():
    roster = build_roster(ROSTER_NAMES)

    mapping = assign_pseudonyms(roster)

    assert set(mapping.values()) == {f"panel_{i:04d}" for i in range(1, 7)}
    # Sorting the input differently must not change anyone's pseudonym.
    assert mapping == assign_pseudonyms(build_roster(list(reversed(ROSTER_NAMES))))
    for canonical, pseudonym in mapping.items():
        for token in ("asian", "caucasian", "male", "female", "mideast"):
            assert token not in pseudonym
        assert canonical not in pseudonym
