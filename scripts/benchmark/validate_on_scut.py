"""Check these implementations against their published numbers, on SCUT-FBP5500.

Every result in `reports/benchmark/` is produced by *this* code on *this*
dataset. If an implementation is subtly wrong -- as `comboloss` was until
2026-07-28, optimising a different objective from its paper -- nothing in that
table would reveal it. A low score reads as "hard dataset" or "weak method"
just as easily as "broken implementation", and there is no way to tell them
apart from inside.

The only real test is to run the same code on the dataset the papers used and
see whether it lands near their reported figures. That is what this does.

**Published reference figures** (SCUT-FBP5500, 60% train / 40% test, Pearson):

    cnn-resnet18    0.8900   Liang et al. 2018
    cnn-resnext50   0.8777   Liang et al. 2018
    comboloss       0.8965   Xu & Xiang 2020 (SE-ResNeXt-50)

A reimplementation landing within roughly 0.02-0.03 of these is evidence the
mechanism is right. Landing far below means the implementation is wrong, and
the MEBeauty number for that method should not be trusted.

**The dataset is not downloaded automatically, deliberately.** SCUT-FBP5500 is
released under an agreement that has to be accepted by a person, and this
project does not fetch data around a licence. Obtain it from

    https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

and point `--scut` at the extracted directory. Expected layout:

    <scut>/Images/*.jpg
    <scut>/train_test_files/split_of_60%training and 40%testing/train.txt
    <scut>/train_test_files/split_of_60%training and 40%testing/test.txt

Each line of train.txt/test.txt is `<filename> <score>`, scores on 1-5.

**What differs from the MEBeauty protocol, necessarily.** SCUT ships no
validation split, so 10% of its training images are held out for early
stopping -- the methods here all use validation-based stopping, and removing it
would change what is being tested. That makes these numbers slightly
pessimistic against the papers, which trained on all 60%.

Labels stay on SCUT's native 1-5 scale rather than being rescaled to 1-10:
rescaling would change every loss's effective weighting and test something
other than the published setup.

    uv run python scripts/benchmark/validate_on_scut.py \\
        --scut ~/data/SCUT-FBP5500 \\
        --methods cnn-resnet18,cnn-resnext50,comboloss
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mebeauty_benchmark.benchmark.base import assert_no_test_leak
from mebeauty_benchmark.benchmark.metrics import correlation_ci, point_metrics
from mebeauty_benchmark.benchmark.protocol import (
    Protocol,
    Split,
    environment,
    set_seed,
    write_result,
)

#: Pearson correlations reported by the papers on SCUT-FBP5500's 60/40 split.
#: These are the targets an honest reimplementation should approach. They are
#: NOT comparable to anything in `reports/benchmark/` -- different dataset.
PUBLISHED_PC = {
    "cnn-resnet18": (0.8900, "Liang et al. 2018, ICPR"),
    "cnn-resnext50": (0.8777, "Liang et al. 2018, ICPR"),
    "comboloss": (0.8965, "Xu & Xiang 2020 (SE-ResNeXt-50)"),
    "ldl-ren2017": (0.9040, "Xu & Xiang 2020, reporting LDL on SCUT-FBP5500"),
}

#: Below this correlation with its published figure, an implementation is
#: judged not to reproduce the method rather than merely to underperform.
SUSPICION_THRESHOLD = 0.05

SPLIT_DIR = "split_of_60%training and 40%testing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scut", required=True, help="Extracted SCUT-FBP5500 root")
    parser.add_argument("--methods", default=",".join(PUBLISHED_PC))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", default="reports/scut_validation")
    return parser.parse_args()


def read_split(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        filename, score = line.rsplit(maxsplit=1)
        rows.append({"file_name": filename, "score": float(score)})
    return pd.DataFrame(rows)


def make_split(name: str, frame: pd.DataFrame, images: Path) -> Split:
    paths = [images / f for f in frame["file_name"]]
    missing = [p for p in paths[:50] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: image not found, e.g. {missing[0]}")
    return Split(
        name=name,
        image_ids=frame["file_name"].to_numpy(),
        labels=frame["score"].to_numpy(dtype=float),
        image_paths=paths,
        distributions=None,
        # Attribute-conditioned methods (aanet) need these columns to exist.
        # SCUT ships gender/ethnicity in its filenames (AF/AM/CF/CM).
        metadata=pd.DataFrame(
            {
                "gender": [
                    "female" if f[1].upper() == "F" else "male"
                    for f in frame["file_name"]
                ],
                "ethnicity": [
                    "asian" if f[0].upper() == "A" else "caucasian"
                    for f in frame["file_name"]
                ],
            }
        ),
    )


def load_scut(root: Path, seed: int, val_fraction: float) -> Protocol:
    images = root / "Images"
    split_dir = root / "train_test_files" / SPLIT_DIR
    if not images.is_dir():
        raise FileNotFoundError(f"No Images/ directory under {root}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No {SPLIT_DIR!r} under {root / 'train_test_files'}")

    train_all = read_split(split_dir / "train.txt")
    test = read_split(split_dir / "test.txt")

    # Carve a validation split out of training, never out of test.
    rng = np.random.default_rng(seed)
    held = rng.random(len(train_all)) < val_fraction
    train, val = train_all[~held], train_all[held]
    print(f"SCUT-FBP5500: train {len(train)}, val {len(val)}, test {len(test)}")

    return Protocol(
        train=make_split("train", train.reset_index(drop=True), images),
        val=make_split("val", val.reset_index(drop=True), images),
        test=make_split("test", test.reset_index(drop=True), images),
        label="score",
        images="scut-native",
        seed=seed,
        root=root,
    )


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run import build_registry

    registry = build_registry(args.seed, args.epochs)
    requested = args.methods.split(",")
    unknown = [m for m in requested if m not in registry]
    if unknown:
        raise SystemExit(f"Unknown method(s) {unknown}")

    protocol = load_scut(
        Path(args.scut).expanduser().resolve(), args.seed, args.val_fraction
    )
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in requested:
        print(f"\n=== {name} on SCUT-FBP5500 ===", flush=True)
        set_seed(args.seed)
        method = registry[name]()
        started = time.time()
        method.fit(protocol)
        elapsed = time.time() - started

        assert_no_test_leak(method, protocol, seed=args.seed)
        prediction = method.predict(protocol.test)
        point = point_metrics(prediction.scores, protocol.test.labels)
        low, high = correlation_ci(
            prediction.scores, protocol.test.labels, seed=args.seed
        )

        published, source = PUBLISHED_PC.get(name, (None, ""))
        gap = None if published is None else round(point.PC - published, 4)
        verdict = "no published figure recorded"
        if gap is not None:
            verdict = (
                "reproduces the published figure"
                if gap > -SUSPICION_THRESHOLD
                else "DOES NOT reproduce -- treat this implementation as suspect"
            )

        record = {
            "method": name,
            "dataset": "SCUT-FBP5500",
            "split": "60/40, paper's own file lists",
            "label_scale": "1-5 (SCUT native, not rescaled)",
            "point": point.as_dict(),
            "pearson_ci95": [round(low, 4), round(high, 4)],
            "published_PC": published,
            "published_source": source,
            "gap_vs_published": gap,
            "verdict": verdict,
            "epochs_trained": getattr(method, "epochs_trained", None),
            "train_seconds": round(elapsed, 1),
            "environment": environment(),
        }
        write_result(out / f"{name}.json", record)
        rows.append(record)
        print(
            f"  PC {point.PC:.4f} vs published {published}  gap {gap}  -> {verdict}",
            flush=True,
        )

    lines = [
        "# Fidelity check: these implementations on SCUT-FBP5500",
        "",
        "Purpose: decide whether a low MEBeauty score means a hard dataset or a",
        "broken implementation. These numbers are on SCUT-FBP5500 and must not",
        "be placed in the same table as any MEBeauty result.",
        "",
        "| Method | PC here | Published PC | Gap | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['method']}` | {r['point']['PC']:.4f} | {r['published_PC']} | "
            f"{r['gap_vs_published']} | {r['verdict']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
