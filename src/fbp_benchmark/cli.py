"""The command line. Three verbs: `list`, `run`, `report`.

    fbp-benchmark list
    fbp-benchmark run --method cnn-resnet18
    fbp-benchmark run --era classical
    fbp-benchmark report

Defaults point at MEBeauty on the Hub, so `run` works with no arguments and no
setup. `--dataset configs/your.yaml` points everything at another dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import DatasetSpec, load_protocol
from .registry import ERA_ORDER, available
from .runner import run, save

DEFAULT_RESULTS = Path("results")


def _spec(args: argparse.Namespace) -> DatasetSpec:
    if args.dataset:
        return DatasetSpec.from_yaml(args.dataset)
    return DatasetSpec(config=args.config, label_column=args.label)


def cmd_list(args: argparse.Namespace) -> int:
    entries = available(args.era)
    if not entries:
        print(f"No methods registered for era {args.era!r}")
        return 1
    era = None
    for entry in entries:
        if entry.era != era:
            era = entry.era
            print(f"\n{era.upper()}")
        flags = []
        if entry.requires:
            flags.append("needs " + ", ".join(entry.requires))
        if entry.notes:
            flags.append(entry.notes)
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        print(f"  {entry.name:18s} {entry.reference}{suffix}")
    from .methods import unavailable

    for module, reason in unavailable.items():
        print(f"\n! methods.{module} could not be imported: {reason}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    names = (
        [args.method]
        if args.method
        else [
            e.name for e in available(args.era) if not args.skip_slow or not e.trainable
        ]
    )
    protocol = load_protocol(
        _spec(args), protocol=args.protocol, fold=args.fold, seed=args.seed
    )
    print(f"{protocol.spec.repo_id} [{protocol.spec.config}] -> {protocol.describe()}")

    overrides = {
        k: v
        for k, v in (("epochs", args.epochs), ("patience", args.patience))
        if v is not None
    }
    failures = 0
    for name in names:
        try:
            result = run(name, protocol, seed=args.seed, **overrides)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad. A sweep of twenty methods must not be ended
            # by one that cannot import its backbone or runs out of memory;
            # the failure is reported and the remaining methods still run.
            failures += 1
            print(f"  {name:18s} FAILED: {exc}", file=sys.stderr)
            continue
        path = save(result, args.out)
        headline = {k: result.metrics[k] for k in ("PC", "MAE") if k in result.metrics}
        print(f"  {name:18s} {headline}  {result.seconds:.0f}s -> {path.name}")
    return 1 if failures and not args.keep_going else 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_table

    print(build_table(args.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("fbp-benchmark", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", help="YAML DatasetSpec; defaults to MEBeauty")
        p.add_argument(
            "--config",
            default="fbp_extended",
            help="Hugging Face config name; `fbp` is the minimal surface",
        )
        p.add_argument("--label", default="beauty_score")
        p.add_argument("--protocol", choices=("holdout", "cv"), default="holdout")
        p.add_argument("--fold", type=int, help="required when --protocol cv")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=str(DEFAULT_RESULTS))

    p_list = sub.add_parser("list", help="show registered methods")
    p_list.add_argument("--era", choices=ERA_ORDER)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="train and evaluate")
    shared(p_run)
    p_run.add_argument("--method", help="one method; omit to run a whole era")
    p_run.add_argument("--era", choices=ERA_ORDER)
    p_run.add_argument(
        "--epochs", type=int, help="override every schedule (smoke tests)"
    )
    p_run.add_argument("--patience", type=int)
    p_run.add_argument(
        "--skip-slow", action="store_true", help="classical methods only"
    )
    p_run.add_argument(
        "--keep-going", action="store_true", help="exit 0 despite failures"
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="render the results table")
    p_report.add_argument("--out", default=str(DEFAULT_RESULTS))
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
