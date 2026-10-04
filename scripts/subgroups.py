"""Per-subgroup metrics for one method, with bootstrap confidence intervals.

    uv run python scripts/subgroups.py --method dinov2-partial

Reads the prediction dumps in `results/` and `results/cv/fold*/`, joins them to
the gender and ethnicity columns the dataset ships, and prints markdown tables.
Cross-validation pools the out-of-fold predictions, so every image is scored
exactly once by a fold that did not train on it -- the held-out split leaves
28-98 images per group, which is too few for a stable correlation and is
reported only as a sanity check.

Metrics come from `fbp_benchmark.metrics`; this script only groups and
resamples. Nothing here trains or writes to `results/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fbp_benchmark import load_protocol
from fbp_benchmark.metrics import point_metrics

#: Bootstrap resamples. 10,000 keeps the interval endpoints stable to ~0.001.
ITERATIONS = 10_000


@dataclass(frozen=True)
class Scored:
    """One method's predictions, with the metadata needed to group them."""

    labels: np.ndarray
    predicted: np.ndarray
    groups: dict[str, np.ndarray]

    def subset(self, mask: np.ndarray) -> Scored:
        return Scored(
            self.labels[mask],
            self.predicted[mask],
            {k: v[mask] for k, v in self.groups.items()},
        )


def _load(method: str, results: Path, protocol: str, fold: int | None) -> Scored:
    spec_protocol = load_protocol(protocol=protocol, fold=fold, seed=0)
    directory = results if fold is None else results / "cv" / f"fold{fold}"
    path = directory / f"{method}_predictions.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Produce it with:\n"
            f"  fbp-benchmark run --method {method}"
            + ("" if fold is None else f" --protocol cv --fold {fold}")
            + f" --out {directory}"
        )
    predicted = np.load(path)["predictions"]
    split = spec_protocol.test
    labels = np.asarray(split.labels, dtype=float)
    if len(predicted) != len(labels):
        raise SystemExit(
            f"{path} holds {len(predicted)} predictions but the split has "
            f"{len(labels)} images -- the file is stale."
        )
    groups = {
        column: split.metadata[column].to_numpy()
        for column in ("ethnicity", "gender")
        if column in split.metadata
    }
    return Scored(labels, predicted, groups)


def pooled_cv(method: str, results: Path, folds: int = 5) -> Scored:
    """Out-of-fold predictions for every image, concatenated across folds."""
    parts = [_load(method, results, "cv", fold) for fold in range(folds)]
    return Scored(
        np.concatenate([p.labels for p in parts]),
        np.concatenate([p.predicted for p in parts]),
        {
            key: np.concatenate([p.groups[key] for p in parts])
            for key in parts[0].groups
        },
    )


def _point(scored: Scored) -> tuple[float, float, float]:
    m = point_metrics(scored.predicted, scored.labels)
    return m.pearson, m.mae, m.rmse


def bootstrap(scored: Scored, rng: np.random.Generator) -> list[tuple[float, float]]:
    """95% percentile intervals for PC, MAE and RMSE."""
    draws = []
    n = len(scored.labels)
    for _ in range(ITERATIONS):
        index = rng.integers(0, n, n)
        sample = scored.subset(index)
        # A resample with no label variance has no defined correlation.
        if np.std(sample.labels) == 0 or np.std(sample.predicted) == 0:
            continue
        draws.append(_point(sample))
    array = np.asarray(draws)
    return [
        (
            float(np.percentile(array[:, k], 2.5)),
            float(np.percentile(array[:, k], 97.5)),
        )
        for k in range(3)
    ]


def gap(
    a: Scored, b: Scored, metric: int, rng: np.random.Generator
) -> dict[str, float]:
    """Difference between two groups, resampling each independently."""
    draws = []
    for _ in range(ITERATIONS):
        left = a.subset(rng.integers(0, len(a.labels), len(a.labels)))
        right = b.subset(rng.integers(0, len(b.labels), len(b.labels)))
        if min(np.std(left.labels), np.std(right.labels)) == 0:
            continue
        draws.append(_point(left)[metric] - _point(right)[metric])
    array = np.asarray(draws)
    low, high = float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))
    return {
        "difference": _point(a)[metric] - _point(b)[metric],
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": float(low > 0 or high < 0),
    }


def table(scored: Scored, column: str, rng: np.random.Generator) -> str:
    """Markdown rows for one grouping column, largest PC first."""
    values = scored.groups[column]
    rows = []
    for name in sorted(set(values)):
        group = scored.subset(values == name)
        pc, mae, rmse = _point(group)
        ci = bootstrap(group, rng)
        rows.append((pc, name, len(group.labels), pc, ci[0], mae, ci[1], rmse))
    rows.sort(reverse=True)

    lines = [
        f"| {column.capitalize()} | n | PC | PC 95% CI | MAE | MAE 95% CI | RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, name, n, pc, pc_ci, mae, mae_ci, rmse in rows:
        lines.append(
            f"| `{name}` | {n:,} | {pc:.4f} | [{pc_ci[0]:.4f}, {pc_ci[1]:.4f}] "
            f"| {mae:.4f} | [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}] | {rmse:.4f} |"
        )
    pc, mae, rmse = _point(scored)
    ci = bootstrap(scored, rng)
    lines.append(
        f"| **all** | **{len(scored.labels):,}** | **{pc:.4f}** "
        f"| [{ci[0][0]:.4f}, {ci[0][1]:.4f}] | **{mae:.4f}** "
        f"| [{ci[1][0]:.4f}, {ci[1][1]:.4f}] | **{rmse:.4f}** |"
    )
    return "\n".join(lines)


def extremes(scored: Scored, column: str, metric: int) -> tuple[str, str]:
    """The best and worst group names for a metric (lower is better for MAE)."""
    values = scored.groups[column]
    names = sorted(set(values))
    scored_by = {n: _point(scored.subset(values == n))[metric] for n in names}
    ranked = sorted(names, key=lambda n: scored_by[n], reverse=metric == 0)
    return ranked[0], ranked[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="dinov2-partial")
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    cv = pooled_cv(args.method, args.results)

    print(f"## {args.method}: 5-fold out-of-fold, n = {len(cv.labels):,}\n")
    for column in cv.groups:
        print(table(cv, column, rng), "\n")
        for metric, label in ((0, "PC"), (1, "MAE")):
            best, worst = extremes(cv, column, metric)
            result = gap(
                cv.subset(cv.groups[column] == best),
                cv.subset(cv.groups[column] == worst),
                metric,
                rng,
            )
            verdict = "excludes zero" if result["excludes_zero"] else "includes zero"
            print(
                f"- largest {label} gap by {column}: `{best}` vs `{worst}` = "
                f"{result['difference']:+.4f}, 95% CI "
                f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] ({verdict})"
            )
        print()

    holdout = _load(args.method, args.results, "holdout", None)
    print(f"## {args.method}: held-out split, n = {len(holdout.labels):,} (small-n)\n")
    print("| Group | n | PC | MAE | RMSE |")
    print("|---|---|---|---|---|")
    for column, values in holdout.groups.items():
        for name in sorted(set(values)):
            group = holdout.subset(values == name)
            if len(group.labels) < 3:
                continue
            pc, mae, rmse = _point(group)
            print(
                f"| `{name}` | {len(group.labels)} | {pc:.4f} | {mae:.4f} | {rmse:.4f} |"
            )
    pc, mae, rmse = _point(holdout)
    print(f"| all | {len(holdout.labels)} | {pc:.4f} | {mae:.4f} | {rmse:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
