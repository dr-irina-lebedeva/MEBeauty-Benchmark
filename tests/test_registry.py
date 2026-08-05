"""The registry is the repository's index. If it drifts, nothing else is trusted."""

from __future__ import annotations

import pytest

from fbp_benchmark import registry
from fbp_benchmark.setups import SETUPS


def test_every_method_is_registered_under_a_known_era():
    entries = registry.available()
    assert entries, "no methods registered -- methods/__init__ failed to import"
    assert {e.era for e in entries} <= set(registry.ERA_ORDER)


def test_names_are_unique_and_sorted_by_era():
    entries = registry.available()
    names = [e.name for e in entries]
    assert len(names) == len(set(names))
    eras = [registry.ERA_ORDER.index(e.era) for e in entries]
    assert eras == sorted(eras), "listing must group by era"


def test_every_trainable_method_has_a_published_setup():
    # A trainable method with no entry in setups.py would silently fall back to
    # a default schedule, and the table would then measure this benchmark's
    # tuning rather than the paper's.
    missing = [
        e.name for e in registry.available() if e.trainable and e.name not in SETUPS
    ]
    assert not missing, f"no setup recorded for: {missing}"


def test_unknown_method_names_are_rejected_with_a_hint():
    with pytest.raises(KeyError, match="resnet"):
        registry.get("resnet")


def test_duplicate_registration_is_refused():
    # Two entries under one name would silently overwrite a result.
    with pytest.raises(ValueError, match="already registered"):

        @registry.register("mean-baseline", era="baseline", reference="dup")
        class _Duplicate:
            pass


def test_every_era_has_at_least_one_method():
    by_era = {era: registry.available(era) for era in registry.ERA_ORDER}
    assert all(by_era.values()), (
        f"empty eras: {[k for k, v in by_era.items() if not v]}"
    )
