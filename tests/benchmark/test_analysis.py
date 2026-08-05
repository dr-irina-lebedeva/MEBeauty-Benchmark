"""Tests for the post-run analysis: ensembling and significance correction.

These decide what the results table is allowed to claim, so they need to be
right for the same reason the metrics do. Both were written after the methods
and neither had a test until this file existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mebeauty_benchmark.methods.foundation import Ensemble


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """The implementation under test, imported from the analysis script.

    `scripts/benchmark/` is not an installed package, so the path is extended
    here rather than at module scope where the import sorter cannot keep the
    two statements in the required order.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmark"))
    from compare import holm_bonferroni as implementation

    return implementation(p_values)


# ------------------------------------------------------------- Holm-Bonferroni


def test_holm_is_less_conservative_than_bonferroni():
    """Holm must never reject less than Bonferroni, which is why it is used."""
    raw = [0.001, 0.02, 0.03, 0.5]

    adjusted = holm_bonferroni(raw)
    bonferroni = [min(1.0, p * len(raw)) for p in raw]

    assert all(a <= b + 1e-12 for a, b in zip(adjusted, bonferroni, strict=True))
    assert adjusted[0] < 0.05  # the strongest result survives correction


def test_holm_adjusted_values_are_monotone():
    """A more extreme raw p must never get a larger adjusted p."""
    raw = [0.04, 0.01, 0.03, 0.02]

    adjusted = holm_bonferroni(raw)

    by_raw = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
    assert by_raw == sorted(by_raw)


def test_holm_caps_at_one():
    assert all(p <= 1.0 for p in holm_bonferroni([0.5, 0.6, 0.9, 0.95]))


def test_holm_corrects_the_multiple_comparisons_problem():
    """105 pairs of pure noise must not yield ~5 'significant' results."""
    rng = np.random.default_rng(0)
    # Under the null, p-values are uniform.
    raw = list(rng.uniform(0, 1, 105))

    uncorrected = sum(p < 0.05 for p in raw)
    corrected = sum(p < 0.05 for p in holm_bonferroni(raw))

    assert uncorrected >= 3, "sanity: some noise should look significant raw"
    assert corrected == 0


# -------------------------------------------------------------------- Ensemble


def _members(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(6.0, 1.2, n)
    return truth, {
        "a": truth + rng.normal(0, 0.8, n),
        "b": truth + rng.normal(0, 0.8, n),
        # Deliberately on a different scale: a raw average would let the
        # wide-spread members swamp this one.
        "c": truth * 0.3 + 5.0 + rng.normal(0, 0.3, n),
    }


def test_ensemble_beats_its_members():
    truth, members = _members()
    ensemble = Ensemble(list(members), truth)

    combined = ensemble.predict_from(members)

    def pc(values):
        return np.corrcoef(values, truth)[0, 1]

    assert pc(combined) > max(pc(v) for v in members.values())


def test_ensemble_is_immune_to_a_members_scale():
    """Rank-averaging must ignore how wide a member's predictions are.

    A member that predicts the same *ordering* on a compressed scale should
    contribute identically. Averaging raw scores would not have this property,
    which is the reason for ranking.
    """
    truth, members = _members()
    ensemble = Ensemble(list(members), truth)

    baseline = ensemble.predict_from(members)
    rescaled = dict(members)
    rescaled["c"] = members["c"] * 100.0 - 400.0  # same order, absurd scale

    assert np.allclose(baseline, ensemble.predict_from(rescaled))


def test_ensemble_output_lies_on_the_training_label_scale():
    """Quantile mapping keeps MAE and RMSE meaningful.

    A pure rank average would correlate fine and score terribly on error
    metrics; the mapping is what stops that.
    """
    _truth, members = _members()
    train_labels = np.random.default_rng(1).normal(6.0, 1.2, 1000)
    ensemble = Ensemble(list(members), train_labels)

    combined = ensemble.predict_from(members)

    assert combined.min() >= train_labels.min() - 1e-9
    assert combined.max() <= train_labels.max() + 1e-9
    assert abs(combined.mean() - train_labels.mean()) < 0.2


def test_ensemble_rejects_a_missing_member():
    truth, members = _members()
    ensemble = Ensemble(["a", "b", "nonexistent"], truth)

    with pytest.raises(KeyError, match="nonexistent"):
        ensemble.predict_from(members)


def test_ensemble_needs_at_least_one_member():
    with pytest.raises(ValueError, match="at least one member"):
        Ensemble([], np.zeros(10))


def test_a_single_member_ensemble_preserves_its_ranking():
    truth, members = _members()
    ensemble = Ensemble(["a"], truth)

    combined = ensemble.predict_from(members)

    # Same order as the member, remapped onto the label scale.
    assert np.array_equal(np.argsort(combined), np.argsort(members["a"]))
