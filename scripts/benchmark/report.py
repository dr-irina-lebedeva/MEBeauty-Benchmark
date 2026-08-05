"""Assemble every benchmark artifact into one readable results document.

`summary.md` is a bare ranking. This produces the document a reader actually
needs: the ranking *plus* the things that decide whether the ranking means
anything -- confidence intervals, how long early stopping let each method run,
distribution metrics for the methods that predict one, run-to-run stability,
and the caveats attached to specific entries.

Regenerated from the result files rather than written by hand, so it cannot
drift from the numbers it describes.

    uv run python scripts/benchmark/report.py \\
        --results reports/benchmark \\
        --out docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Caveats that belong beside a specific number, not in a footnote nobody
#: reaches. Keyed by method; printed in the table's own notes column.
CAVEATS = {
    "fpem": "architecture only -- no Swin/FaceNet/CLIP/GPT-2 encoders",
    "transfbp": "TransMix augmentation omitted (the paper's contribution)",
    "aanet": "gates pooled features; paper modulates conv filters",
    "gan2014": "no external unlabelled corpus",
    "ldl-ren2017": "objective only; paper's SLDL is a structural SVM",
    "pi-cnn": "fixed horizontal bands, not landmark-driven boxes",
    "uol": "batch-internal ordering, not the paper's MC comparator",
    "r3cnn": "optimiser inherited from the authors' AaNet paper",
    "dinov2-linear": "NOT a published FBP method",
    "dinov2-partial": "NOT a published FBP method",
    "ensemble": "combination of other entries, not a method",
    "rw-ldl": "**proposed here**",
    "rw-ldl-noweight": "ablation of rw-ldl",
    "rw-ldl-kl": "ablation of rw-ldl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="reports/benchmark")
    parser.add_argument("--out", default="docs/RESULTS.md")
    parser.add_argument(
        "--compare-with",
        default=None,
        help="A second results directory, to report run-to-run stability.",
    )
    return parser.parse_args()


def load(directory: Path) -> dict[str, dict]:
    records = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in {"significance.json"}:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "point" in record:
            records[record["method"]] = record
    return records


def main() -> None:
    args = parse_args()
    results = Path(args.results).expanduser().resolve()
    records = load(results)
    if not records:
        raise SystemExit(f"No results in {results}")

    ranked = sorted(records.values(), key=lambda r: -r["point"]["PC"])
    any_record = ranked[0]
    protocol = any_record["protocol"]
    environment = any_record["environment"]

    lines = [
        "# MEBeauty benchmark-v1 — results",
        "",
        (
            f"Generated from `{results.name}/` by "
            "`scripts/benchmark/report.py`. Do not edit by hand."
        ),
        "",
        "## Protocol",
        "",
        "| | |",
        "|---|---|",
        f"| Protocol | `{protocol['protocol']}` |",
        f"| Label | `{protocol['label']}` — rater-normalised mean, ≥10 ratings/image |",
        f"| Images | `{protocol['images']}` |",
        (
            f"| Splits | train {protocol['sizes']['train']} / "
            f"val {protocol['sizes']['val']} / "
            f"test {protocol['sizes']['test']} |"
        ),
        f"| Seed | {protocol['seed']} |",
        (
            f"| Device | {environment.get('device', '?')}, "
            f"torch {environment.get('torch', '?')} |"
        ),
        (
            f"| Commit | `{environment['git_commit'][:8]}` "
            f"(tree dirty: {environment.get('git_tree_dirty')}) |"
        ),
        "",
        (
            "**Not comparable to published SCUT-FBP5500 or MEBeauty-2022 "
            "numbers.** Different images, labels and splits."
        ),
        "",
        "## Results",
        "",
        "| # | Method | PC | PC 95% CI | SROCC | MAE | RMSE | epochs | train | setup | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, record in enumerate(ranked, start=1):
        point, ci = record["point"], record["pearson_ci95"]
        epochs = record.get("epochs_trained")
        setup = record.get("setup", {}).get("source", "--")
        note = CAVEATS.get(record["method"], "")
        lines.append(
            f"| {rank} | `{record['method']}` | **{point['PC']:.4f}** | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {point['SROCC']:.4f} | "
            f"{point['MAE']:.4f} | {point['RMSE']:.4f} | "
            f"{epochs if epochs is not None else '--'} | "
            f"{record['train_seconds']:.0f}s | {setup} | {note} |"
        )

    # Distribution metrics, for the entries that predict one.
    with_distributions = [r for r in ranked if "distribution" in r]
    if with_distributions:
        lines += [
            "",
            "## Label-distribution metrics",
            "",
            (
                "Only methods that predict a full rating distribution. "
                "Chebyshev, Clark, Canberra and KL are distances (lower is "
                "better); Cosine and Intersection are similarities (higher "
                "is better)."
            ),
            "",
            "| Method | Chebyshev ↓ | Clark ↓ | Canberra ↓ | KL ↓ | Cosine ↑ | Intersection ↑ |",
            "|---|---|---|---|---|---|---|",
        ]
        for record in with_distributions:
            d = record["distribution"]
            lines.append(
                f"| `{record['method']}` | {d['Chebyshev']:.4f} | "
                f"{d['Clark']:.4f} | {d['Canberra']:.4f} | {d['KL']:.4f} | "
                f"{d['Cosine']:.4f} | {d['Intersection']:.4f} |"
            )

    # Early stopping is the largest source of run-to-run movement, so how long
    # each method actually ran belongs in the report, not just in the JSON.
    stopped_early = [
        r
        for r in ranked
        if r.get("epochs_trained")
        and r["setup"].get("epochs")
        and r["epochs_trained"] < 0.4 * r["setup"]["epochs"]
    ]
    if stopped_early:
        lines += [
            "",
            "## Methods that stopped well before their schedule",
            "",
            (
                "Early stopping (patience 5) on a "
                f"{protocol['sizes']['val']}-image validation split is noisy "
                "for runs at high learning rates. An entry that halted in the "
                "first 40% of its schedule is reporting where it stopped, not "
                "what the method can do."
            ),
            "",
            "| Method | epochs run | of | optimizer | lr |",
            "|---|---|---|---|---|",
        ]
        for record in stopped_early:
            setup = record["setup"]
            lines.append(
                f"| `{record['method']}` | {record['epochs_trained']} | "
                f"{setup['epochs']} | {setup['optimizer']} | "
                f"{setup['learning_rate']} |"
            )

    if args.compare_with:
        other = load(Path(args.compare_with).expanduser().resolve())
        shared = sorted(set(records) & set(other))
        if shared:
            lines += [
                "",
                "## Run-to-run stability",
                "",
                (
                    f"Against `{Path(args.compare_with).name}/`. A method "
                    "whose score moves substantially between runs is "
                    "reporting variance, not capability."
                ),
                "",
                "| Method | this run | other run | Δ |",
                "|---|---|---|---|",
            ]
            for name in sorted(
                shared,
                key=lambda n: -abs(records[n]["point"]["PC"] - other[n]["point"]["PC"]),
            ):
                a, b = records[name]["point"]["PC"], other[name]["point"]["PC"]
                flag = " **unstable**" if abs(a - b) > 0.05 else ""
                lines.append(f"| `{name}` | {a:.4f} | {b:.4f} | {a - b:+.4f}{flag} |")

    significance = results / "significance.md"
    if significance.is_file():
        lines += [
            "",
            "## Significance",
            "",
            (
                "See `significance.md` beside these results — paired "
                "bootstrap over all method pairs, Holm-Bonferroni corrected. "
                "**The ranking above is not evidence that one method beats "
                "another until that table says so.**"
            ),
        ]
    else:
        lines += [
            "",
            "## Significance",
            "",
            (
                "Not yet computed. Run `scripts/benchmark/compare.py` before "
                "describing any method as better than another: with "
                f"n = {protocol['sizes']['test']} the leading entries sit "
                "inside each other's confidence intervals."
            ),
        ]

    lines += [
        "",
        "## How to read this",
        "",
        (
            "- **Eleven of the published methods are reimplementations** "
            "(`adapted`). A low score is evidence about this implementation "
            "on this dataset, not about the original work. See "
            "`docs/IMPLEMENTATION_VS_PAPERS.md`."
        ),
        (
            "- **No implementation has been checked against its published "
            "number.** `scripts/benchmark/validate_on_scut.py` exists for "
            "that and needs SCUT-FBP5500, which requires accepting a release "
            "agreement."
        ),
        (
            "- **`mean-baseline` predicts the training mean.** Its PC is 0 by "
            "construction; its MAE is the number every other entry must beat."
        ),
        "",
    ]

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(ranked)} methods)")


if __name__ == "__main__":
    main()
