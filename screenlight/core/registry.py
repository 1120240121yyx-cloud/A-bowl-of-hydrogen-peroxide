"""Tiny plugin registry: `register(kind, name, factory)` / `resolve(kind, order)`.

Deliberately boring. The point is that new modules (light engines, scene
classifiers, KWS, brightness backends) plug in without app.py knowing them.
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[tuple[str, str], Callable[..., Any]] = {}


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    _REGISTRY[(kind, name)] = factory


def get(kind: str, name: str) -> Callable[..., Any] | None:
    return _REGISTRY.get((kind, name))


def names(kind: str) -> list[str]:
    return sorted(name for k, name in _REGISTRY if k == kind)


def fallback_chain(kind: str, requested: str, preferred: list[str]) -> list[str]:
    """Build a fallback order for a requested engine name.

    ``auto`` / unknown -> the preferred chain.  An explicit, registered name is
    moved to the front but the preferred chain is kept behind it, so a broken
    engine still degrades instead of crashing.
    """
    chain = [n for n in preferred if get(kind, n) is not None]
    if requested and requested != "auto" and get(kind, requested) is not None:
        chain = [requested] + [n for n in chain if n != requested]
    if requested and requested != "auto" and requested not in chain:
        chain = [requested] + chain   # let the caller see the error, then fall back
    return chain or list(preferred)


def factories(kind: str, chain: list[str]) -> list[tuple[str, Callable[..., Any]]]:
    out: list[tuple[str, Callable[..., Any]]] = []
    for name in chain:
        factory = get(kind, name)
        if factory is not None:
            out.append((name, factory))
    return out
