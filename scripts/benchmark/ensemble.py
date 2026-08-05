"""Combine finished methods into an ensemble, and score it like any other.

Reads the per-image predictions the harness saves and rank-averages them.
Costs no training: every member has already run.

**Members are chosen on validation, never on test.** The obvious approach --
"take the best few from the results table" -- selects members using test
performance and reports the result on the same test set, which is how an
ensemble comes to look better than it is. There is no validation prediction
saved for each method, so this script instead requires the member list to be
given explicitly, or selects by a rule that does not consult the test labels:
`--top-k` uses each member's *own recorded validation MAE*, which the harness
stores as `best_val_mae`.

Classical methods have no validation MAE (they do not train by epochs), so
`--top-k` considers only methods that recorded one. List them explicitly with
`--members` to include them.

    uv run python scripts/benchmark/ensemble.py --results reports/benchmark --top-k 5
    uv run python scripts/benchmark/ensemble.py --members dinov2-linear,ldl-ren2017,fpem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mebeauty_benchmark.benchmark.metrics import correlation_ci, point_metrics
from mebeauty_benchmark.benchmark.protocol import load_protocol, write_result
from mebeauty_benchmark.methods.foundation import Ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", default="data/mebeauty_v3")
    parser.add_argument("--results", default="reports/benchmark")
    parser.add_argument("--members", default=None, help="comma-separated method names")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Select k members by recorded validation MAE (never test).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results).expanduser().resolve()

    predictions, actual, image_ids = {}, None, None
    for path in sorted(results.glob("*_predictions.npz")):
        data = np.load(path, allow_pickle=True)
        name = path.name.removesuffix("_predictions.npz")
        predictions[name] = data["predicted"]
        if actual is None:
            actual, image_ids = data["actual"], data["image_ids"]
        elif not np.array_equal(data["image_ids"], image_ids):
            raise SystemExit(f"{name} was evaluated on different images")

    if args.members:
        members = args.members.split(",")
        unknown = [m for m in members if m not in predictions]
        if unknown:
            raise SystemExit(f"No predictions for {unknown}")
        selection = "explicit"
    elif args.top_k:
        scored = []
        for name in predictions:
            record_path = results / f"{name}.json"
            if not record_path.is_file():
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            val_mae = record.get("best_val_mae")
            if val_mae is not None:
                scored.append((val_mae, name))
        if len(scored) < args.top_k:
            raise SystemExit(
                f"Only {len(scored)} methods recorded a validation MAE; "
                f"cannot pick {args.top_k} without consulting test scores. "
                "Pass --members explicitly instead."
            )
        members = [name for _, name in sorted(scored)[: args.top_k]]
        selection = f"top {args.top_k} by validation MAE"
    else:
        raise SystemExit("Pass --members or --top-k")

    print(f"Members ({selection}): {', '.join(members)}")

    # The quantile reference is the *training* labels, so no test information
    # enters the mapping.
    protocol = load_protocol(args.v3, seed=args.seed)
    ensemble = Ensemble(members, protocol.train.labels)
    combined = ensemble.predict_from(predictions)

    point = point_metrics(combined, actual)
    low, high = correlation_ci(combined, actual, seed=args.seed)
    print(f"  ensemble  {point.as_dict()}  PC 95% CI [{low:.3f}, {high:.3f}]")
    for name in members:
        member_point = point_metrics(predictions[name], actual)
        print(f"    member {name:16s} PC {member_point.PC:.4f}")

    record = {
        "method": "ensemble",
        "reference": "--",
        "implementation_note": (
            f"rank-average of {len(members)} methods, quantile-mapped to the "
            "training label distribution; members chosen by " + selection
        ),
        "members": members,
        "member_selection": selection,
        "setup": {"source": "default", "method": "ensemble"},
        "protocol": protocol.describe(),
        "train_seconds": 0.0,
        "epochs_trained": None,
        "best_val_mae": None,
        "point": point.as_dict(),
        "pearson_ci95": [round(low, 4), round(high, 4)],
    }
    write_result(results / "ensemble.json", record)
    np.savez_compressed(
        results / "ensemble_predictions.npz",
        image_ids=image_ids,
        predicted=combined,
        actual=actual,
    )
    print(f"Wrote {results / 'ensemble.json'}")


if __name__ == "__main__":
    main()
