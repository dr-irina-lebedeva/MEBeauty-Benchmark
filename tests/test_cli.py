"""The command line, exercised without touching the network."""

from __future__ import annotations

import json

import pytest

from fbp_benchmark import cli


def test_list_prints_every_era(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    for era in ("BASELINE", "CLASSICAL", "DEEP", "FOUNDATION"):
        assert era in out


def test_list_can_be_filtered_to_one_era(capsys):
    assert cli.main(["list", "--era", "classical"]) == 0
    out = capsys.readouterr().out
    assert "eisenthal2006" in out
    assert "dinov2-linear" not in out


def test_list_shows_what_a_method_requires(capsys):
    cli.main(["list", "--era", "classical"])
    assert "needs landmarks" in capsys.readouterr().out


def test_an_unknown_era_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.main(["list", "--era", "nonsense"])


def test_report_on_an_empty_directory_says_so(tmp_path, capsys):
    assert cli.main(["report", "--out", str(tmp_path)]) == 0
    assert "No results" in capsys.readouterr().out


def test_report_renders_results(tmp_path, capsys):
    (tmp_path / "m.json").write_text(
        json.dumps({"method": "m", "era": "deep", "metrics": {"PC": 0.5}, "seconds": 1})
    )
    assert cli.main(["report", "--out", str(tmp_path)]) == 0
    assert "`m`" in capsys.readouterr().out


def test_check_readme_fails_when_the_table_is_stale(tmp_path, capsys):
    (tmp_path / "m.json").write_text(
        json.dumps({"method": "m", "era": "deep", "metrics": {"PC": 0.5}, "seconds": 1})
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- RESULTS:START -->\nstale\n<!-- RESULTS:END -->\n"
        "<!-- METHODS:START -->\n<!-- METHODS:END -->\n"
    )
    code = cli.main(
        ["report", "--out", str(tmp_path), "--readme", str(readme), "--check-readme"]
    )
    assert code == 1


def test_check_readme_passes_after_updating(tmp_path):
    (tmp_path / "m.json").write_text(
        json.dumps({"method": "m", "era": "deep", "metrics": {"PC": 0.5}, "seconds": 1})
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n"
        "<!-- METHODS:START -->\n<!-- METHODS:END -->\n"
    )
    args = ["report", "--out", str(tmp_path), "--readme", str(readme)]
    assert cli.main([*args, "--update-readme"]) == 0
    assert cli.main([*args, "--check-readme"]) == 0


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
