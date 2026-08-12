"""Report rendering, including the README gate that keeps the table honest."""

from __future__ import annotations

import json

import pytest

from fbp_benchmark import report


def write(directory, method, era="deep", pc=0.5, seconds=10.0, **metrics):
    payload = {
        "method": method,
        "era": era,
        "metrics": {"PC": pc, "SROCC": pc, "MAE": 1 - pc, "RMSE": 1.2 - pc, **metrics},
        "seconds": seconds,
        "dataset": "org/data",
        "config": "fbp",
        "label": "beauty_score",
        "protocol": "holdout",
        "seed": 0,
        "sizes": {"train": 100, "val": 20, "test": 20},
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{method}.json").write_text(json.dumps(payload))


def test_runs_load_and_ignore_unrelated_json(tmp_path):
    write(tmp_path, "a")
    (tmp_path / "manifest.json").write_text('{"not": "a result"}')
    runs = report.load_runs(tmp_path)
    assert [r.method for r in runs] == ["a"]


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert report.load_runs(tmp_path / "nope") == []


def test_leaderboard_groups_by_era_then_ranks_within_it(tmp_path):
    write(tmp_path, "weak-deep", era="deep", pc=0.3)
    write(tmp_path, "strong-deep", era="deep", pc=0.9)
    write(tmp_path, "a-baseline", era="baseline", pc=0.0)
    table = report.leaderboard(report.load_runs(tmp_path))
    order = [
        line.split("`")[1] for line in table.splitlines() if line.startswith("| `")
    ]
    # Baseline era first, and the stronger deep method above the weaker one.
    assert order == ["a-baseline", "strong-deep", "weak-deep"]


def test_every_row_is_a_complete_markdown_row(tmp_path):
    write(tmp_path, "a")
    for line in report.leaderboard(report.load_runs(tmp_path)).splitlines():
        assert line.startswith("|") and line.endswith("|"), line


def test_a_method_missing_a_metric_renders_a_dash_not_a_crash(tmp_path):
    write(tmp_path, "a")
    runs = report.load_runs(tmp_path)
    object.__setattr__(runs[0], "metrics", {"PC": 0.5})
    assert "--" in report.leaderboard(runs)


def test_fold_summary_reports_mean_and_spread(tmp_path):
    for fold, pc in enumerate((0.70, 0.80, 0.90)):
        write(tmp_path / "cv" / f"fold{fold}", "m", pc=pc)
    summary = report.fold_summary(report.load_folds(tmp_path / "cv"))
    assert "0.8000" in summary  # mean of 0.7, 0.8, 0.9
    assert "±" in summary


def test_fold_summary_ranks_by_mean_correlation(tmp_path):
    for fold in range(2):
        write(tmp_path / "cv" / f"fold{fold}", "better", pc=0.9)
        write(tmp_path / "cv" / f"fold{fold}", "worse", pc=0.4)
    summary = report.fold_summary(report.load_folds(tmp_path / "cv"))
    rows = [line for line in summary.splitlines() if line.startswith("| `")]
    assert "better" in rows[0] and "worse" in rows[1]


def test_readme_update_is_idempotent(tmp_path):
    write(tmp_path, "a")
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n\n{report.START}\nold\n{report.END}\n\noutro\n")

    assert report.update_readme(readme, tmp_path) is True
    assert report.update_readme(readme, tmp_path) is False, "second write changed it"
    text = readme.read_text()
    assert "intro" in text and "outro" in text and "old" not in text


def test_readme_without_markers_refuses_to_guess(tmp_path):
    write(tmp_path, "a")
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError, match="no results markers"):
        report.update_readme(readme, tmp_path)


def test_empty_results_render_a_notice_rather_than_an_empty_table(tmp_path):
    assert "No results" in report.leaderboard([])
    assert "No cross-validation" in report.fold_summary({})
