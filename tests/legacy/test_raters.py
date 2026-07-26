"""Fixtures use deliberately synthetic AMT-shaped Worker IDs
(``AEXAMPLEWORKER0N``). Earlier versions of this file used real Worker
IDs copied from the legacy data -- five of them, each resolving to a
real rater in the pseudonym mapping. That contradicted this project's
own guarantee that raw Worker IDs are never committed, and was caught
only by scanning staged content against the mapping before the first
commit. Never paste a real identifier into a fixture.
"""

import pandas as pd

from mebeauty_benchmark.legacy.raters import (
    build_pseudonym_mapping,
    collect_worker_ids,
    collect_worker_ids_from_value_column,
    is_amt_worker_id,
    pseudonymize_columns,
    pseudonymize_value_column,
)


def test_recognizes_amt_worker_ids():
    assert is_amt_worker_id("AEXAMPLEWORKER01")
    assert is_amt_worker_id("AEXAMPLEWORKER02")


def test_rejects_non_worker_id_columns():
    for name in ["mean", "image", "path", "malecm39", "mean_pr_f", "Unnamed: 0"]:
        assert not is_amt_worker_id(name)


def test_collect_worker_ids_filters_other_columns():
    columns = ["mean", "image", "AEXAMPLEWORKER01", "malecm39", "AEXAMPLEWORKER02"]
    assert collect_worker_ids(columns) == {"AEXAMPLEWORKER01", "AEXAMPLEWORKER02"}


def test_pseudonym_mapping_is_deterministic_regardless_of_input_order():
    ids = ["AEXAMPLEWORKER03", "AEXAMPLEWORKER01", "AEXAMPLEWORKER02"]
    mapping_a = build_pseudonym_mapping(ids)
    mapping_b = build_pseudonym_mapping(list(reversed(ids)))
    assert mapping_a == mapping_b
    assert len(set(mapping_a.values())) == len(ids)


def test_pseudonymize_columns_renames_only_mapped_worker_ids():
    frame = pd.DataFrame({"mean": [1.0], "image": ["x.jpg"], "AEXAMPLEWORKER01": [5.0]})
    mapping = build_pseudonym_mapping(["AEXAMPLEWORKER01"])

    renamed = pseudonymize_columns(frame, mapping)

    assert list(renamed.columns) == ["mean", "image", mapping["AEXAMPLEWORKER01"]]
    assert "AEXAMPLEWORKER01" not in renamed.columns


def test_collect_worker_ids_from_value_column_ignores_non_ids():
    series = pd.Series(["AEXAMPLEWORKER04", "AEXAMPLEWORKER05", None, "not-an-id"])
    assert collect_worker_ids_from_value_column(series) == {
        "AEXAMPLEWORKER04",
        "AEXAMPLEWORKER05",
    }


def test_pseudonymize_value_column_replaces_known_ids_only():
    series = pd.Series(["AEXAMPLEWORKER04", "unknown-rater"])
    mapping = build_pseudonym_mapping(["AEXAMPLEWORKER04"])

    result = pseudonymize_value_column(series, mapping)

    assert result.tolist() == [mapping["AEXAMPLEWORKER04"], "unknown-rater"]
