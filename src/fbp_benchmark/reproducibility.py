"""Seeding, environment capture, and writing a result you can trust later.

A results table is only worth reading if each row says what produced it. Every
result written by this benchmark carries the commit, whether the tree was
modified, the Python and torch versions, and the device -- so a number can be
reproduced, or knowingly discounted.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

#: Paths whose contents can change a result. Deliberately excludes `results/`:
#: a run writes its own output there, so a whole-tree check would mark every
#: run after the first as irreproducible because of its own results file.
CODE_PATHS = ("src", "scripts", "tests", "configs", "pyproject.toml", "uv.lock")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:  # pragma: no cover - git absent
        return ""


def environment() -> dict[str, object]:
    """Record what produced a result, so it can be reproduced or discounted."""
    commit = _git("rev-parse", "HEAD")
    # A commit hash alone is a false promise when the code is modified: the
    # result did not come from that commit and cannot be reproduced by
    # checking it out. Recorded so a reader can discount the run rather than
    # trust a hash that does not describe it.
    dirty = bool(_git("status", "--porcelain", "--", *CODE_PATHS))

    info: dict[str, object] = {
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
