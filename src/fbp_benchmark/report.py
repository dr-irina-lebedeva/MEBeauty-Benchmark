"""Turn a results directory into tables, and keep the README's copy honest.

The README's results are **generated**, never hand-edited, and CI fails if they
drift from `results/`. A leaderboard typed by hand is a leaderboard that
quietly stops matching the code that produced it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registry import ERA_ORDER

#: Columns shown, and whether higher is better.
COLUMNS: tuple[tuple[str, bool], ...] = (
    ("PC", True),
    ("SROCC", True),
    ("MAE", False),
    ("RMSE", False),
)

#: The README block this module owns. Anything between the markers is replaced.
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


@dataclass(frozen=True)
class Run:
    """One result file, flattened to what a table needs."""

    method: str
    era: str
    metrics: dict[str, float]
    seconds: float
    protocol: dict

    @classmethod
    def from_path(cls, path: Path) -> Run | None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload or "method" not in payload:
            return None
        return cls(
            method=payload["method"],
            era=payload.get("era", "unknown"),
            metrics=payload["metrics"],
            seconds=float(payload.get("seconds", 0.0)),
            protocol={
                k: payload.get(k)
                for k in ("dataset", "config", "label", "protocol", "seed", "sizes")
            },
        )


def load_runs(directory: str | Path) -> list[Run]:
    """Every result directly in `directory`, newest schema only."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return []
    runs = [Run.from_path(p) for p in sorted(directory.glob("*.json"))]
    return [r for r in runs if r is not None]


def load_folds(directory: str | Path) -> dict[str, list[Run]]:
    """Cross-validation results: method -> one Run per fold."""
    directory = Path(directory).expanduser()
    folds: dict[str, list[Run]] = {}
    for fold_dir in sorted(directory.glob("fold*")):
        for run in load_runs(fold_dir):
            folds.setdefault(run.method, []).append(run)
    return folds


def _row(method: str, era: str, values: dict[str, float], trailing: str | None) -> str:
    cells = [f"{values[name]:.4f}" if name in values else "--" for name, _ in COLUMNS]
    if trailing is not None:
        cells.append(trailing)
    return f"| `{method}` | {era} | " + " | ".join(cells) + " |"


def leaderboard(runs: list[Run], time_column: bool = True) -> str:
    """Markdown table, grouped by era and ranked within it.

    Ranked *within* era rather than globally: putting a 2006 geometric
    regressor on the same line as a fine-tuned transformer invites a
    conclusion neither supports.
    """
    if not runs:
        return "_No results yet. Run `fbp-benchmark run`._"

    header = "| Method | Era | " + " | ".join(c for c, _ in COLUMNS)
    header += " | Time |" if time_column else " |"
    lines = [header, "|---|---|" + "---|" * (len(COLUMNS) + int(time_column))]
    for era in ERA_ORDER:
        for run in sorted(
            (r for r in runs if r.era == era),
            key=lambda r: -r.metrics.get("PC", float("-inf")),
        ):
            trailing = f"{run.seconds:.0f}s" if time_column else None
            lines.append(_row(run.method, run.era, run.metrics, trailing))
    return "\n".join(lines)


def fold_summary(folds: dict[str, list[Run]]) -> str:
    """Mean +/- sd across folds, which is what a CV number means."""
    if not folds:
        return "_No cross-validation results yet._"
    lines = [
        "| Method | Era | " + " | ".join(f"{c} (mean ± sd)" for c, _ in COLUMNS) + " |",
        "|---|---|" + "---|" * len(COLUMNS),
    ]
    ranked = sorted(
        folds.items(),
        key=lambda kv: -float(np.mean([r.metrics.get("PC", 0) for r in kv[1]])),
    )
    for method, runs in ranked:
        cells = []
        for name, _ in COLUMNS:
            values = [r.metrics[name] for r in runs if name in r.metrics]
            cells.append(
                f"{np.mean(values):.4f} ± {np.std(values):.4f}" if values else "--"
            )
        lines.append(f"| `{method}` | {runs[0].era} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def provenance(runs: list[Run]) -> str:
    """One line describing what the table was produced against."""
    if not runs:
        return ""
    head = runs[0].protocol
    sizes = head.get("sizes") or {}
    size_text = " / ".join(f"{k} {v:,}" for k, v in sizes.items())
    return (
        f"`{head.get('dataset')}` config `{head.get('config')}`, "
        f"label `{head.get('label')}`, seed {head.get('seed')} — {size_text}."
    )


def readme_section(results: str | Path = "results") -> str:
    """The generated block that lives between the README markers."""
    results = Path(results)
    runs = load_runs(results)
    folds = load_folds(results / "cv")

    parts = [
        "### Held-out split",
        "",
        provenance(runs),
        "",
        leaderboard(runs),
        "",
    ]
    if folds:
        n_folds = max(len(v) for v in folds.values())
        parts += [
            f"### {n_folds}-fold cross-validation",
            "",
            (
                "Every image is tested exactly once across the folds, so this "
                "is the comparison to trust — the held-out split has only "
                f"{(runs[0].protocol.get('sizes') or {}).get('test', '?')} "
                "test images, too few to separate methods within about 0.04 "
                "correlation."
            ),
            "",
            fold_summary(folds),
            "",
        ]
    return "\n".join(parts).strip() + "\n"


def update_readme(
    path: str | Path = "README.md", results: str | Path = "results"
) -> bool:
    """Rewrite the README's results block. True if the file changed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(
            f"{path} has no results markers. Add:\n{START}\n{END}\n"
            "so this module knows what it may replace."
        )
    block = f"{START}\n\n{readme_section(results)}\n{END}"
    updated = re.sub(
        re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.DOTALL
    )
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def build_table(directory: str | Path) -> str:
    """Human-readable leaderboard for `fbp-benchmark report`."""
    runs = load_runs(directory)
    header = f"# Results\n\n{provenance(runs)}\n\n" if runs else ""
    text = header + leaderboard(runs) + "\n"
    folds = load_folds(Path(directory) / "cv")
    if folds:
        text += "\n\n# Cross-validation\n\n" + fold_summary(folds) + "\n"
    return text
