"""Training schedules. These are provenance, not preferences."""

from __future__ import annotations

import pytest

from fbp_benchmark.registry import available
from fbp_benchmark.setups import (
    MAX_EPOCHS,
    MEBEAUTY_TRAIN_IMAGES,
    SETUPS,
    capped_epochs,
    setup_for,
)


def test_every_setup_records_where_its_numbers_came_from():
    # `source` is the difference between quoting a paper and inventing a
    # schedule; a benchmark that loses it measures its author's tuning.
    for name, setup in SETUPS.items():
        assert setup.source in ("paper", "adapted", "default"), name


def test_an_adapted_setup_explains_what_was_changed():
    # `deviation` covers gradient-training changes; the classical methods have
    # no training schedule to deviate from, so theirs is recorded in `notes`.
    # Either is fine -- silently substituting a value is not.
    for name, setup in SETUPS.items():
        if setup.source == "adapted":
            assert setup.deviation or setup.notes, (
                f"{name} adapted a published value without recording why"
            )


def test_setup_names_match_their_keys():
    for name, setup in SETUPS.items():
        assert setup.method == name


def test_unknown_method_lookup_is_explicit():
    with pytest.raises(KeyError):
        setup_for("no-such-method")


def test_epochs_are_capped_but_modest_schedules_are_untouched():
    assert capped_epochs(10) == 10
    assert capped_epochs(500) == MAX_EPOCHS


def test_no_setup_exceeds_the_cap():
    for name, setup in SETUPS.items():
        assert setup.epochs <= MAX_EPOCHS, name


def test_documented_training_size_matches_the_released_split():
    # This constant appears throughout the setup rationales. It silently went
    # stale once when the splits were rebuilt, making every argument that
    # cites it wrong.
    from fbp_benchmark.data import DatasetSpec

    assert MEBEAUTY_TRAIN_IMAGES == 1962, (
        "update every rationale that cites the training-set size"
    )
    assert DatasetSpec().config == "fbp_extended"


def test_every_trainable_method_has_positive_hyperparameters():
    for entry in available():
        if not entry.trainable:
            continue
        setup = SETUPS[entry.name]
        assert setup.learning_rate > 0, entry.name
        assert setup.batch_size > 0, entry.name
        assert setup.epochs > 0, entry.name
