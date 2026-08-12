"""The method registry: one name, one implementation, one era.

Methods register by decoration, so there is no second list to keep in sync:

    @register("my-method", era="deep", reference="Author et al., 2027")
    class MyMethod: ...

`era` groups the results table: `classical` (landmark geometry), `deep`
(convolutional networks trained end to end), `foundation` (large pretrained
backbones) and `baseline` (reference points, not methods).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .methods.base import Method

Era = Literal["baseline", "classical", "deep", "foundation"]

ERA_ORDER: tuple[Era, ...] = ("baseline", "classical", "deep", "foundation")

T = TypeVar("T")


@dataclass(frozen=True)
class Entry:
    """One registered method."""

    name: str
    factory: Callable[..., Method]
    era: Era
    reference: str
    #: URL of the publication. Empty when no stable link was verified -- an
    #: invented DOI is worse than none, so the citation still appears in the
    #: methods table, just without a hyperlink.
    paper: str = ""
    #: What the method needs from the dataset beyond images and a label.
    #: Checked before a run so a missing column fails immediately with a
    #: readable message rather than deep inside a training loop.
    requires: tuple[str, ...] = ()
    #: True when the method is constructed from its paper's schedule in
    #: `setups.py`. Classical methods and baselines take only a seed.
    trainable: bool = False
    notes: str = ""


_REGISTRY: dict[str, Entry] = {}


def register(
    name: str,
    *,
    era: Era,
    reference: str,
    paper: str = "",
    requires: tuple[str, ...] = (),
    trainable: bool = False,
    notes: str = "",
) -> Callable[[T], T]:
    """Add a method to the registry under `name`."""
    if era not in ERA_ORDER:
        raise ValueError(f"Unknown era {era!r}; choose from {ERA_ORDER}")

    def decorate(factory: T) -> T:
        if name in _REGISTRY:
            raise ValueError(
                f"Method {name!r} is already registered by "
                f"{_REGISTRY[name].factory!r}. Names must be unique -- two "
                "entries under one name would silently overwrite a result."
            )
        _REGISTRY[name] = Entry(
            name=name,
            factory=factory,  # type: ignore[arg-type]
            era=era,
            reference=reference,
            paper=paper,
            requires=requires,
            trainable=trainable,
            notes=notes,
        )
        return factory

    return decorate


def available(era: Era | None = None) -> list[Entry]:
    """Every registered method, ordered by era then name."""
    import fbp_benchmark.methods  # noqa: F401  (import triggers registration)

    entries = [e for e in _REGISTRY.values() if era is None or e.era == era]
    return sorted(entries, key=lambda e: (ERA_ORDER.index(e.era), e.name))


def get(name: str) -> Entry:
    """Look up one method, with a useful error if the name is wrong."""
    import fbp_benchmark.methods  # noqa: F401

    if name not in _REGISTRY:
        close = [n for n in _REGISTRY if name.lower() in n.lower()]
        hint = f" Did you mean: {', '.join(sorted(close))}?" if close else ""
        raise KeyError(
            f"No method named {name!r}. Run `fbp-benchmark list` to see "
            f"the {len(_REGISTRY)} registered methods.{hint}"
        )
    return _REGISTRY[name]


def create(name: str, seed: int = 0, **overrides) -> Method:
    """Instantiate a registered method."""
    entry = get(name)
    if not entry.trainable:
        return entry.factory(seed=seed)
    from .methods.training import TrainConfig

    return entry.factory(config=TrainConfig.from_setup(name, **overrides), seed=seed)
