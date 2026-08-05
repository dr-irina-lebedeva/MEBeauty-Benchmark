"""The benchmark protocol: what every method is given, and what it must return.

A benchmark is only meaningful if every method is evaluated on exactly the same
thing. This module is the single definition of that, so a result table cannot
quietly mix protocols.

**One protocol, `benchmark-v1`.** Fixed train/val/test from
`ratings/aggregate/*.parquet`. Not k-fold: the splits were rebuilt so that every
photograph of one person sits in a single split, and re-folding at random would
undo that and reintroduce identity leakage. Cross-validation would give tighter
error bars around a number that had been quietly inflated.

**Labels.** `score` -- the rater-normalised mean over valid raters, on images
with at least 8 such ratings. Alternative labels ship (`score_raw_mean`,
`score_adjusted`, `score_all_raters`) and can be selected explicitly, but the
default is the canonical one, and a result must say which it used.

**`score` is not a published column.** The Hugging Face release ships
`score_adjusted` (recommended) and `score_mean` (= `score_all_raters` here).
`score` correlates 0.99 with `score_adjusted`, but a table produced against it
is not reproducible by someone holding only the release, so any result meant
for publication should name a published label.

**Comparability with published numbers.** Results here are *not* comparable to
figures reported on SCUT-FBP5500 or on the 2022 MEBeauty release. Different
images, different labels, different splits. The original split cannot even be
reconstructed -- it was generated without a random seed (Finding 20). Any table
mixing them is wrong, and this docstring exists so nobody does it by accident.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")

#: Paths whose contents can change a result. Used to decide whether a run is
#: reproducible from its commit; deliberately excludes `reports/`, `data/` and
#: `docs/`, none of which the training path reads.
CODE_PATHS = ("src", "scripts", "tests", "pyproject.toml", "uv.lock")

#: The rating scale the soft labels are defined over.
SCORE_BINS_MIN, SCORE_BINS_MAX = 1, 10

#: The label a method predicts unless told otherwise. See the module docstring.
DEFAULT_LABEL = "score"

#: Labels a method may be evaluated against. Anything else is a typo, and
#: silently scoring against the wrong column would invalidate a whole table.
AVAILABLE_LABELS = (
    "score",
    "score_raw_mean",
    "score_adjusted",
    "score_adjusted_offset",
    "score_all_raters",
)

#: Which soft label belongs with which point label. A distribution-learning
#: method trains on the distribution and is scored on the label, so handing it
#: one whose expectation is a *different* quantity grades it on something it
#: was never asked to predict. Measured: doing that cost the LDL entry ~0.09
#: MAE. Labels absent here get no distribution rather than a mismatched one.
DISTRIBUTION_FOR_LABEL = {
    "score": "distributions_normalised.parquet",
    "score_raw_mean": "distributions.parquet",
}

#: Image configurations a method may consume.
AVAILABLE_IMAGES = ("cropped_256", "standardized_256", "native")


@dataclass(frozen=True)
class Split:
    """One split's images, labels and (optionally) rating distributions."""

    name: str
    image_ids: np.ndarray
    labels: np.ndarray
    image_paths: list[Path]
    distributions: np.ndarray | None = None
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: How many valid raters produced each label, and how much they disagreed.
    #: Shipped to methods because label reliability is *known* here and varies
    #: 4x across the test split -- an image rated 11 times is a far noisier
    #: target than one rated 102 times. A method may use this on train and val;
    #: using it on test would be reading the answer sheet's margin notes, so
    #: the harness never scores against it.
    n_ratings: np.ndarray | None = None
    label_std: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def standard_error(self) -> np.ndarray | None:
        """Standard error of each label's mean: std / sqrt(n)."""
        if self.n_ratings is None or self.label_std is None:
            return None
        return np.asarray(self.label_std) / np.sqrt(np.asarray(self.n_ratings))


@dataclass(frozen=True)
class Protocol:
    """Everything a method is given, and the provenance of it."""

    train: Split
    val: Split
    test: Split
    label: str
    images: str
    seed: int
    root: Path

    def describe(self) -> dict[str, object]:
        return {
            "protocol": "benchmark-v1",
            "label": self.label,
            "images": self.images,
            "seed": self.seed,
            "sizes": {s: len(getattr(self, s)) for s in SPLITS},
            "not_comparable_to": (
                "SCUT-FBP5500 results, and the 2022 MEBeauty release: "
                "different images, labels and splits"
            ),
        }


def _distribution_matrix(
    frame: pd.DataFrame, image_ids: np.ndarray
) -> np.ndarray | None:
    """Rating distributions aligned to `image_ids`, or None if unavailable."""
    if frame.empty:
        return None
    lookup = dict(zip(frame["image_id"], frame["probabilities"]))
    if not all(image_id in lookup for image_id in image_ids):
        return None
    return np.vstack([np.asarray(lookup[i], dtype=float) for i in image_ids])


def load_protocol(
    root: str | Path,
    label: str = DEFAULT_LABEL,
    images: str = "cropped_256",
    seed: int = 0,
) -> Protocol:
    """Load `benchmark-v1` from a built `data/mebeauty_v3` directory."""
    if label not in AVAILABLE_LABELS:
        raise ValueError(f"Unknown label {label!r}; choose from {AVAILABLE_LABELS}")
    if images not in AVAILABLE_IMAGES:
        raise ValueError(f"Unknown images {images!r}; choose from {AVAILABLE_IMAGES}")

    root = Path(root).expanduser().resolve()
    metadata = pd.read_parquet(root / "images" / "metadata.parquet")
    image_dir = root / "images" if images == "native" else root / images / "images"

    distribution_file = DISTRIBUTION_FOR_LABEL.get(label)
    if distribution_file is None:
        distributions = pd.DataFrame()
    else:
        try:
            distributions = pd.read_parquet(root / "ratings" / distribution_file)
        except FileNotFoundError:
            distributions = pd.DataFrame()

    file_names = dict(zip(metadata["image_id"], metadata["file_name"]))
    splits = {}
    for name in SPLITS:
        frame = pd.read_parquet(root / "ratings" / "aggregate" / f"{name}.parquet")
        if label not in frame.columns:
            raise ValueError(
                f"Label {label!r} is not in {name}.parquet; "
                f"available: {sorted(set(frame.columns) & set(AVAILABLE_LABELS))}"
            )
        # Sorting makes the protocol deterministic: two runs must present the
        # same images in the same order, or per-image predictions cannot be
        # compared between methods.
        frame = frame.sort_values("image_id").reset_index(drop=True)
        missing = frame[label].isna()
        if missing.any():
            raise ValueError(
                f"{name}: {int(missing.sum())} rows have no {label!r}. "
                "A benchmark cannot score against a missing label."
            )

        image_ids = frame["image_id"].to_numpy()
        paths = []
        for image_id in image_ids:
            name_on_disk = (
                file_names[image_id] if images == "native" else f"{image_id}.jpg"
            )
            path = image_dir / name_on_disk
            if not path.is_file():
                raise FileNotFoundError(f"{name}: image missing on disk: {path}")
            paths.append(path)

        labels = frame[label].to_numpy(dtype=float)
        matrix = _distribution_matrix(distributions, image_ids)
        if matrix is not None:
            # Checked on every load rather than trusted: if the soft label and
            # the point label ever disagree again, it must surface here and
            # not as an unexplained penalty in one method's MAE.
            bins = np.arange(SCORE_BINS_MIN, SCORE_BINS_MAX + 1)
            drift = np.abs((matrix * bins).sum(axis=1) - labels).max()
            if drift > 1e-6:
                raise ValueError(
                    f"{name}: the distribution's expectation differs from "
                    f"{label!r} by up to {drift:.4f}. A method trained on the "
                    "distribution would be scored against a different target."
                )

        splits[name] = Split(
            name=name,
            image_ids=image_ids,
            labels=labels,
            image_paths=paths,
            distributions=matrix,
            metadata=metadata.set_index("image_id").loc[image_ids].reset_index(),
            n_ratings=(
                frame["n_ratings"].to_numpy(dtype=float)
                if "n_ratings" in frame
                else None
            ),
            label_std=(frame["std"].to_numpy(dtype=float) if "std" in frame else None),
        )

    return Protocol(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        label=label,
        images=images,
        seed=seed,
        root=root,
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:  # pragma: no cover - git absent
        return ""


def environment() -> dict[str, str]:
    """Record what produced a result, so it can be reproduced or discounted."""
    commit = _git("rev-parse", "HEAD")
    # A commit hash alone is a false promise when the code is modified: the
    # result did not come from that commit and cannot be reproduced by
    # checking it out. Recorded so a reader can discount the run rather than
    # trust a hash that does not describe it.
    #
    # Scoped to the paths that can change a result. A whole-tree check is
    # useless here and was: a run writes its own results into `reports/`, so
    # the first result file makes the tree dirty and every subsequent record
    # claims to be irreproducible because of its own output.
    dirty = bool(_git("status", "--porcelain", "--", *CODE_PATHS))

    info = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit or "unknown",
        "git_tree_dirty": dirty,
        "reproducible_from_commit": not dirty,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    except ImportError:
        pass
    return info


def set_seed(seed: int) -> None:
    """Seed every generator a method might use."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_result(path: str | Path, payload: dict) -> None:
    """Persist a result with its environment attached."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**payload, "environment": environment()}, indent=2) + "\n",
        encoding="utf-8",
    )
