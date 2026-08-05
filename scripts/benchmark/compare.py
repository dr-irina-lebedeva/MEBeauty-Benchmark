"""Which differences in the results table are real, and which are noise.

A ranked table invites the reader to treat position as meaning. On 434 test
images it usually does not: the top entries of `benchmark-v1` sit within about
0.03 Pearson of each other, which is well inside what resampling the test set
would move them. Reporting that order without saying so is the most likely way
this benchmark gets misread.

**Paired, not independent.** Two methods are compared on the *same* resampled
images, so the shared difficulty of those images cancels. Comparing two
independent confidence intervals instead is far more conservative and would
call real differences insignificant -- which is why the per-image predictions
are saved in the first place; aggregate scores cannot support this test.

Reads every `*_predictions.npz` in the results directory, so it runs on
whatever has finished.

**Multiple comparisons.** With 15 methods there are 105 pairs, and at p<0.05
roughly five would look significant by chance alone. Holm-Bonferroni adjusted
p-values ship beside the raw ones; the "significant" column uses the adjusted
value.

    uv run python scripts/benchmark/compare.py --results reports/benchmark
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from mebeauty_benchmark.benchmark.metrics import paired_bootstrap_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="reports/benchmark")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Adjusted p-values, controlling the family-wise error rate.

    Holm rather than plain Bonferroni: uniformly more powerful, same
    guarantee. Adjusted values are made monotone so a more extreme raw p can
    never end up with a larger adjusted one.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    total = len(p_values)
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        value = (total - rank) * p_values[index]
        running = max(running, min(1.0, value))
        adjusted[index] = running
    return adjusted


def main() -> None:
    args = parse_args()
    results = Path(args.results).expanduser().resolve()

    predictions, actual = {}, None
    for path in sorted(results.glob("*_predictions.npz")):
        data = np.load(path, allow_pickle=True)
        name = path.name.removesuffix("_predictions.npz")
        predictions[name] = data["predicted"]
        if actual is None:
            actual, image_ids = data["actual"], data["image_ids"]
        elif not np.array_equal(data["image_ids"], image_ids):
            raise SystemExit(
                f"{name} was evaluated on different images; a paired test is "
                "meaningless unless every method saw the same test set."
            )
    if len(predictions) < 2:
        raise SystemExit(f"Need at least two methods in {results}")

    def pearson(values: np.ndarray) -> float:
        return float(np.corrcoef(values, actual)[0, 1])

    ranked = sorted(predictions, key=lambda n: -pearson(predictions[n]))
    print(f"{len(ranked)} methods, {len(actual)} test images\n")

    rows = []
    for a, b in combinations(ranked, 2):
        outcome = paired_bootstrap_difference(
            predictions[a], predictions[b], actual, args.iterations, args.seed
        )
        rows.append({"a": a, "b": b, **outcome})

    adjusted = holm_bonferroni([r["p_value"] for r in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_adjusted"] = round(value, 4)
        row["significant"] = bool(value < args.alpha)

    significant = [r for r in rows if r["significant"]]
    print(
        f"{len(significant)} of {len(rows)} pairs differ significantly "
        f"(Holm-adjusted p < {args.alpha})\n"
    )

    # The practical question a reader has: who is actually indistinguishable
    # from the best method?
    best = ranked[0]
    tied = [
        other
        for other in ranked[1:]
        if not next(r["significant"] for r in rows if {r["a"], r["b"]} == {best, other})
    ]
    print(f"Top method: {best} (PC {pearson(predictions[best]):.4f})")
    print(f"Statistically indistinguishable from it: {', '.join(tied) or 'none'}\n")

    lines = [
        "# Which differences are real",
        "",
        (
            f"Paired bootstrap, {args.iterations} resamples of the "
            f"{len(actual)} test images, Holm-Bonferroni adjusted across all "
            f"{len(rows)} pairs. A difference in the results table that does "
            "not appear here as significant should not be described as one "
            "method beating another."
        ),
        "",
        f"**`{best}` leads at PC {pearson(predictions[best]):.4f}**, but is "
        f"statistically indistinguishable from: "
        + (", ".join(f"`{t}`" for t in tied) or "no other method")
        + ".",
        "",
        "| A | B | PC(A) - PC(B) | 95% CI | p | Holm p | significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: -abs(r["difference"])):
        lines.append(
            f"| `{row['a']}` | `{row['b']}` | {row['difference']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['p_value']:.4f} | "
            f"{row['p_adjusted']:.4f} | {'**yes**' if row['significant'] else 'no'} |"
        )
    out = results / "significance.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "significance.json").write_text(
        json.dumps(
            {
                "test": "paired bootstrap on Pearson correlation",
                "iterations": args.iterations,
                "n_test_images": len(actual),
                "alpha": args.alpha,
                "correction": "Holm-Bonferroni",
                "best_method": best,
                "indistinguishable_from_best": tied,
                "pairs": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
