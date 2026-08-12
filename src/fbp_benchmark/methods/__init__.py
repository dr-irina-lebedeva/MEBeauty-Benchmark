"""Importing this package registers every method.

The registry is populated by decoration, so a method only exists once its
module has been imported. Doing that here -- rather than expecting each caller
to remember -- is what makes `fbp-benchmark list` complete.

Optional dependencies are tolerated. `classical` needs only numpy and
scikit-learn; `deep` and `foundation` need torch, and `foundation` also needs
transformers. A missing one removes those methods from the listing instead of
breaking the whole import, so the classical baselines stay runnable on a
machine with no GPU stack installed.
"""

from __future__ import annotations

import importlib
import logging

_LOG = logging.getLogger(__name__)

#: Every module that registers methods, in era order.
ERA_MODULES = ("classical", "deep", "foundation", "proposed")

unavailable: dict[str, str] = {}

for _module in ERA_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_module}")
    except ImportError as exc:  # pragma: no cover - depends on the install
        unavailable[_module] = str(exc)
        _LOG.debug("methods.%s unavailable: %s", _module, exc)

__all__ = ["ERA_MODULES", "unavailable"]
