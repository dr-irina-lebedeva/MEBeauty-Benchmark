"""Render the results directory as a leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

from .registry import ERA_ORDER

#: Columns shown, and which direction is better.
COLUMNS = (("PC", "up"), ("SROCC", "up"), ("MAE", "down"), ("RMSE", "down"))


def build_table(directory: str | Path) -> str:
    """A markdown leaderboard, grouped by era and ranked within it."""
    directory = Path(directory).expanduser().resolve()
    results = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" in payload:
            results.append(payload)
    if not results:
        return f"No results in {directory}. Run `fbp-benchmark run` first."

    head = results[0]
    lines = [
        f"# Results — {head.get('dataset', '?')} [{head.get('config', '?')}]",
        "",
        (
            f"Label `{head.get('label', '?')}`, "
            f"protocol `{head.get('protocol', '?')}`, "
            f"seed {head.get('seed', '?')}. Sizes {head.get('sizes', {})}."
        ),
        "",
        "| Method | Era | " + " | ".join(c for c, _ in COLUMNS) + " | Time |",
        "|---|---|" + "---|" * (len(COLUMNS) + 1),
    ]
    # Ranked by correlation within era: comparing a classical method against a
    # fine-tuned transformer on one line invites a conclusion neither supports.
    for era in ERA_ORDER:
        rows = [r for r in results if r.get("era") == era]
        for row in sorted(rows, key=lambda r: -r["metrics"].get("PC", 0)):
            cells = [f"{row['metrics'].get(c, float('nan')):.4f}" for c, _ in COLUMNS]
            lines.append(
                f"| `{row['method']}` | {era} | "
                + " | ".join(cells)
                + f" | {row.get('seconds', 0):.0f}s |"
            )
    return "\n".join(lines) + "\n"
