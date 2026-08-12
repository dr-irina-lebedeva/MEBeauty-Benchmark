"""Seeding and the provenance stamped onto every result."""

from __future__ import annotations

import json

import numpy as np

from fbp_benchmark.reproducibility import environment, set_seed, write_result


def test_seeding_makes_numpy_and_random_repeat():
    import random

    set_seed(7)
    first = (np.random.rand(3).tolist(), random.random())
    set_seed(7)
    assert (np.random.rand(3).tolist(), random.random()) == first


def test_different_seeds_give_different_draws():
    set_seed(1)
    a = np.random.rand(5).tolist()
    set_seed(2)
    assert np.random.rand(5).tolist() != a


def test_environment_records_what_is_needed_to_reproduce_or_discount():
    info = environment()
    for key in (
        "timestamp",
        "git_commit",
        "git_tree_dirty",
        "reproducible_from_commit",
        "python",
        "platform",
        "numpy",
    ):
        assert key in info


def test_reproducible_flag_is_the_inverse_of_a_dirty_tree():
    info = environment()
    assert info["reproducible_from_commit"] is (not info["git_tree_dirty"])


def test_written_results_carry_their_environment(tmp_path):
    path = tmp_path / "nested" / "r.json"
    write_result(path, {"method": "m", "metrics": {"PC": 1.0}})
    payload = json.loads(path.read_text())
    assert payload["method"] == "m"
    assert "git_commit" in payload["environment"]
