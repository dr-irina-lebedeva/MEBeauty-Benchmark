"""Regenerate the README figures from the dataset and the results directory.

Charts only, deliberately. The dataset shows identifiable people who did not
consent to being rated for attractiveness, so no face from it is reproduced
here; the figures describe the distributions instead.

    uv run --with matplotlib python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/figures")
INK, GRID = "#1f2933", "#d9e2ec"
ERA_COLOUR = {
    "baseline": "#9aa5b1",
    "classical": "#7b9acc",
    "deep": "#4a7fb5",
    "foundation": "#22577a",
}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def dataset_figure() -> None:
    from fbp_benchmark.data import DatasetSpec, load_protocol

    protocol = load_protocol(DatasetSpec(config="fbp_extended"))
    labels = np.concatenate(
        [protocol.train.labels, protocol.val.labels, protocol.test.labels]
    )
    counts = np.concatenate(
        [
            s.n_ratings
            for s in (protocol.train, protocol.val, protocol.test)
            if s.n_ratings is not None
        ]
    )
    ethnicity = [
        v
        for s in (protocol.train, protocol.val, protocol.test)
        for v in s.metadata.get("ethnicity", [])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].hist(labels, bins=40, color="#4a7fb5", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("beauty_score")
    axes[0].set_ylabel("images")
    axes[0].set_title(f"Label distribution (n={len(labels):,})", color=INK, fontsize=10)

    axes[1].hist(counts, bins=40, color="#22577a", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("ratings per image")
    axes[1].set_title(
        f"Rater support ({int(counts.min())}–{int(counts.max())} per image)",
        color=INK,
        fontsize=10,
    )

    if ethnicity:
        names, sizes = np.unique(ethnicity, return_counts=True)
        order = np.argsort(sizes)
        axes[2].barh(names[order], sizes[order], color="#7b9acc")
        axes[2].set_xlabel("images")
        axes[2].set_title("Ethnic composition", color=INK, fontsize=10)

    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "dataset.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "dataset.png")


def _load(path: Path) -> dict | None:
    payload = json.loads(path.read_text())
    if "metrics" in payload and "PC" in payload["metrics"]:
        return payload
    return None


def collect(results: Path = Path("results")) -> list[dict]:
    """One entry per method, cross-validated where those runs exist.

    Cross-validation is the protocol the benchmark asks readers to cite, so the
    figure uses it wherever it is available and falls back to the held-out
    split only for methods that were never run under CV. Mixing them silently
    would be dishonest, so each entry records which protocol produced it.
    """
    holdout = {
        r["method"]: r for p in sorted(results.glob("*.json")) if (r := _load(p))
    }

    folds: dict[str, list[dict]] = {}
    for fold_dir in sorted((results / "cv").glob("fold*")):
        for path in sorted(fold_dir.glob("*.json")):
            if run := _load(path):
                folds.setdefault(run["method"], []).append(run)

    entries = []
    for method, run in holdout.items():
        if method in folds:
            values = [f["metrics"]["PC"] for f in folds[method]]
            entries.append(
                {
                    "method": method,
                    "era": run.get("era"),
                    "pc": float(np.mean(values)),
                    "sd": float(np.std(values)),
                    "protocol": "cv",
                }
            )
        else:
            entries.append(
                {
                    "method": method,
                    "era": run.get("era"),
                    "pc": run["metrics"]["PC"],
                    "sd": 0.0,
                    "protocol": "holdout",
                }
            )
    entries.sort(key=lambda e: e["pc"])
    return entries


def results_figure(results: Path = Path("results")) -> None:
    entries = collect(results)
    if not entries:
        print("no results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(entries) + 1.6))
    labels = [
        e["method"] if e["protocol"] == "cv" else f"{e['method']} *" for e in entries
    ]
    ax.barh(
        labels,
        [e["pc"] for e in entries],
        color=[ERA_COLOUR.get(e["era"], "#4a7fb5") for e in entries],
        # A single held-out run has no spread; NaN draws no error bar at all.
        xerr=[e["sd"] if e["protocol"] == "cv" else np.nan for e in entries],
        error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2.5, "alpha": 0.55},
    )
    # The label is noisy, so perfect prediction does not reach 1.0.
    ax.axvline(0.898, color="#c1121f", linestyle="--", linewidth=1)
    ax.text(
        0.898,
        len(entries) - 0.4,
        " reliability ceiling ≈ 0.90",
        color="#c1121f",
        fontsize=8,
        va="top",
        ha="left",
    )
    cv_count = sum(e["protocol"] == "cv" for e in entries)
    ax.set_xlabel(
        "Pearson correlation with beauty_score\n"
        f"5-fold cross-validation, mean ± sd ({cv_count} methods); "
        "* held-out split, single run"
    )
    ax.set_xlim(0, 1.0)
    style(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour) for colour in ERA_COLOUR.values()
    ]
    ax.legend(handles, ERA_COLOUR, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "results.png", f"({len(entries)} methods, {cv_count} with CV)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_figure()
    results_figure()
