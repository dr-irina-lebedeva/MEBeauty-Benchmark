"""Run benchmark methods under `benchmark-v1` and write comparable results.

Every method sees the same splits, the same label and the same images, and is
scored by the same code. Anything a method could vary on its own -- its
evaluation, its split, its metric -- is owned by the harness instead.

Three things happen for every run and are not optional:

- **Seeds are set** before fitting, and recorded with the result.
- **A leak check runs** after fitting: the method re-predicts with the test
  labels shuffled, and the run fails if its output moves. This catches the one
  bug that would silently invalidate an entire table.
- **Per-image predictions are saved**, not only the summary. Paired significance
  testing between two methods needs their predictions on the *same* images, and
  a table of aggregate scores cannot support it afterwards.

    uv run python scripts/benchmark/run.py --methods mean-baseline,kagian2008
    uv run python scripts/benchmark/run.py --methods all --epochs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mebeauty_benchmark.benchmark.base import MeanBaseline, assert_no_test_leak
from mebeauty_benchmark.benchmark.metrics import (
    correlation_ci,
    distribution_metrics,
    point_metrics,
)
from mebeauty_benchmark.benchmark.protocol import (
    load_protocol,
    set_seed,
    write_result,
)
from mebeauty_benchmark.methods.setups import provenance_table, setup_for


def build_registry(seed: int, epochs: int | None = None):
    """Method name -> factory.

    Each deep method is built from *its own paper's* setup
    (`methods/setups.py`), not from one global config -- a benchmark that
    trains every method under the author's chosen schedule measures the
    author's tuning rather than the literature. `--epochs` overrides all of
    them, which is for smoke tests only and is recorded in the result.

    Deep methods import lazily so the classical ones still run without torch.
    """

    def deep(cls_name: str, method: str, module_name: str = "deep"):
        def factory():
            import importlib

            from mebeauty_benchmark.methods.deep import TrainConfig

            module = importlib.import_module(
                f"mebeauty_benchmark.methods.{module_name}"
            )
            config = TrainConfig.from_setup(method, epochs=epochs)
            return getattr(module, cls_name)(config=config, seed=seed)

        return factory

    from mebeauty_benchmark.methods.geometric import (
        Eisenthal2006,
        Fan2012,
        Kagian2008,
    )

    return {
        "mean-baseline": MeanBaseline,
        "eisenthal2006": lambda: Eisenthal2006(seed=seed),
        "kagian2008": lambda: Kagian2008(seed=seed),
        "fan2012": lambda: Fan2012(seed=seed),
        # 2014-2022: the CNN era. Ordered as in the survey table.
        "gan2014": deep("Gan2014", "gan2014", "modern"),
        "cnn-resnet18": deep("CNNRegression", "cnn-resnet18"),
        "pi-cnn": deep("PICNN", "pi-cnn", "modern"),
        "ldl-ren2017": deep("LabelDistributionLearning", "ldl-ren2017"),
        "cnn-resnext50": deep("CNNResNeXt", "cnn-resnext50"),
        "r3cnn": deep("R3CNN", "r3cnn"),
        "aanet": deep("AttributeAware", "aanet"),
        "comboloss": deep("ComboLoss", "comboloss"),
        # 2024-2026: ordinal, fusion and transformer methods.
        "uol": deep("UncertaintyOrderLearning", "uol", "modern"),
        "fpem": deep("FPEM", "fpem", "modern"),
        "transfbp": deep("TransFBP", "transfbp", "modern"),
    }


#: Which published work each entry stands for, so a results table can be read
#: against the literature without guessing. Entries marked `reimplementation`
#: preserve the method's mechanism but not its exact features or weights --
#: see each class's docstring for what was substituted.
PROVENANCE = {
    "mean-baseline": ("--", "floor: predicts the training mean"),
    "eisenthal2006": ("Eisenthal et al. 2006", "reimplementation"),
    "kagian2008": ("Kagian et al. 2008", "reimplementation"),
    "fan2012": ("Fan et al. 2012", "reimplementation"),
    "gan2014": ("Gan et al. 2014", "reimplementation, no external unlabelled corpus"),
    "cnn-resnet18": ("Xie et al. 2015 / MEBeauty 2022", "CNN regression baseline"),
    "pi-cnn": ("Xu et al. 2017", "reimplementation, fixed bands not landmark boxes"),
    "ldl-ren2017": ("Ren & Geng 2017", "label distribution learning"),
    "cnn-resnext50": ("Liang et al. 2018", "SCUT-FBP5500 baseline backbone"),
    "r3cnn": ("Lin et al. 2019/2022", "reimplementation, in-batch pairs"),
    "aanet": ("Lin et al. 2019", "reimplementation, conditions on gender/ethnicity"),
    "comboloss": ("Xu & Xiang 2020", "reimplementation"),
    "uol": ("Liang et al. 2024", "reimplementation, no Bradley-Terry graph"),
    "fpem": ("Li et al. 2025", "architecture only -- no Swin/FaceNet/CLIP encoders"),
    "transfbp": ("Boukhari & Dornaika 2026", "reimplementation, no TransMix aug"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", default="data/mebeauty_v3")
    parser.add_argument(
        "--methods", default="mean-baseline", help="comma-separated, or 'all'"
    )
    parser.add_argument("--label", default="score")
    parser.add_argument("--images", default="cropped_256")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override every method's published epoch count. Smoke tests only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/benchmark")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave methods that already have a result file alone. Deep "
        "methods take longer than a single session, so a run has to be "
        "resumable or partial progress is thrown away.",
    )
    return parser.parse_args()


def rebuild_summary(out: Path) -> list[dict]:
    """Summary from every result on disk, not just this invocation's.

    Results accumulate across resumed runs, so building the table from the
    in-memory list would silently shrink it to whatever the last chunk
    happened to contain.
    """
    rows = []
    for path in sorted(out.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "point" in record:
            rows.append(record)

    lines = [
        "# MEBeauty benchmark-v1 results",
        "",
        "Not comparable to published SCUT-FBP5500 or MEBeauty-2022 numbers:",
        "different images, labels and splits. Entries marked *reimpl.* preserve",
        "the method's mechanism but not its exact features or weights -- see",
        "`docs/IMPLEMENTATION_VS_PAPERS.md` for what was substituted and why.",
        "",
        f"{len(rows)} of {len(PROVENANCE)} methods have results.",
        "",
        "| Method | Reference | PC | PC 95% CI | SROCC | MAE | RMSE | setup | epochs | train (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["point"]["PC"]):
        p, ci = r["point"], r["pearson_ci95"]
        epochs = r.get("epochs_trained", "--")
        lines.append(
            f"| `{r['method']}` | {r['reference']} | {p['PC']:.4f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {p['SROCC']:.4f} | {p['MAE']:.4f} | "
            f"{p['RMSE']:.4f} | {r['setup']['source']} | {epochs} | "
            f"{r['train_seconds']:.0f} |"
        )
    lines += ["", "## Training setups and their provenance", "", provenance_table()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    args = parse_args()
    registry = build_registry(args.seed, args.epochs)
    requested = list(registry) if args.methods == "all" else args.methods.split(",")
    unknown = [m for m in requested if m not in registry]
    if unknown:
        raise SystemExit(f"Unknown method(s) {unknown}; available: {sorted(registry)}")

    protocol = load_protocol(
        args.v3, label=args.label, images=args.images, seed=args.seed
    )
    print(json.dumps(protocol.describe(), indent=2))

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        done = [n for n in requested if (out / f"{n}.json").is_file()]
        if done:
            print(f"Skipping {len(done)} already done: {', '.join(done)}", flush=True)
        requested = [n for n in requested if n not in done]

    for name in requested:
        print(f"\n=== {name} ===", flush=True)
        set_seed(args.seed)
        method = registry[name]()
        started = time.time()
        method.fit(protocol)
        elapsed = time.time() - started

        # Non-negotiable: a method that reads test labels invalidates the table.
        assert_no_test_leak(method, protocol, seed=args.seed)

        prediction = method.predict(protocol.test)
        point = point_metrics(prediction.scores, protocol.test.labels)
        low, high = correlation_ci(
            prediction.scores, protocol.test.labels, seed=args.seed
        )

        reference, note = PROVENANCE.get(name, ("--", ""))
        record: dict[str, object] = {
            "method": name,
            "reference": reference,
            "implementation_note": note,
            "setup": setup_for(name).as_dict(),
            "epochs_overridden": args.epochs,
            "protocol": protocol.describe(),
            "train_seconds": round(elapsed, 1),
            # How long early stopping actually let it run, which is the
            # difference between "converged" and "stopped too soon".
            "epochs_trained": getattr(method, "epochs_trained", None),
            "best_val_mae": (
                round(float(method.best_val_mae), 4)
                if getattr(method, "best_val_mae", None) is not None
                else None
            ),
            "point": point.as_dict(),
            "pearson_ci95": [round(low, 4), round(high, 4)],
        }
        if (
            prediction.distributions is not None
            and protocol.test.distributions is not None
        ):
            record["distribution"] = distribution_metrics(
                prediction.distributions, protocol.test.distributions
            ).as_dict()

        write_result(out / f"{name}.json", record)
        # Per-image predictions, so two methods can be compared with a paired
        # test later. Aggregates alone cannot support that.
        np.savez_compressed(
            out / f"{name}_predictions.npz",
            image_ids=protocol.test.image_ids,
            predicted=prediction.scores,
            actual=protocol.test.labels,
        )
        print(
            f"  {point.as_dict()}  PC 95% CI [{low:.3f}, {high:.3f}]  "
            f"{elapsed:.0f}s  epochs={record['epochs_trained']}",
            flush=True,
        )
        rebuild_summary(out)

    lines = rebuild_summary(out)
    print(f"\n{out / 'summary.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
