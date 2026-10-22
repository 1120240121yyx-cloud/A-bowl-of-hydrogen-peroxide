"""Engine construction with a real fallback chain.

`SafeEngine` builds the first working analyzer, smoke-tests it, and silently
switches to the next candidate if it raises later.  The panel can show
`engine.active_name` and `engine.failures` so a broken AI plugin is visible
instead of being swallowed.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from . import registry
from .frame import AnalysisFrame

OFFLINE_KIND = "lightengine:offline"
LIVE_KIND = "lightengine:live"
OFFLINE_PREFERRED = ["perceptual", "classic"]
LIVE_PREFERRED = ["perceptual", "classic"]


class SafeEngine:
    """Wraps a chain of engine factories; degrades on construction/frame errors."""

    def __init__(self, kind: str, requested: str, builder, *args, **kwargs) -> None:
        self.kind = kind
        self.requested = requested or "auto"
        self._builder = builder
        self._args = args
        self._kwargs = kwargs
        self._engine: Any | None = None
        self.active_name = "none"
        self.failures: list[str] = []
        self._alive = True
        self._build()

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        preferred = OFFLINE_PREFERRED if self.kind == OFFLINE_KIND else LIVE_PREFERRED
        chain = registry.fallback_chain(self.kind, self.requested, preferred)
        while chain:
            name = chain.pop(0)
            factory = registry.get(self.kind, name)
            if factory is None:
                self.failures.append(f"{name}: not registered")
                continue
            try:
                engine = self._builder(factory, self._args, self._kwargs)
                self._engine = engine
                self.active_name = name
                return
            except Exception as exc:  # noqa: BLE001 - fallback is the feature
                self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
        self._engine = None
        self.active_name = "none"
        self._alive = False

    # ---------------------------------------------------------------- frames
    def _degrade(self, exc: BaseException) -> None:
        self.failures.append(f"{self.active_name}: {type(exc).__name__}: {exc}")
        self._engine = None
        self.active_name = "none"
        self._build()

    def frame_at(self, t: float, policy, now: float | None = None) -> AnalysisFrame:
        if self._engine is None:
            return AnalysisFrame(engine="none")
        try:
            return self._engine.frame_at(t, policy, now)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            if self._engine is None:
                return AnalysisFrame(engine="none")
            try:
                return self._engine.frame_at(t, policy, now)
            except Exception as exc2:  # noqa: BLE001
                self._degrade(exc2)
                return AnalysisFrame(engine="none")

    # live engines -------------------------------------------------------
    def push(self, block: np.ndarray) -> None:
        if self._engine is not None and hasattr(self._engine, "push"):
            try:
                self._engine.push(block)
            except Exception as exc:  # noqa: BLE001
                self._degrade(exc)

    def frame(self, now: float, policy) -> AnalysisFrame:
        if self._engine is None:
            return AnalysisFrame(engine="none")
        try:
            return self._engine.frame(now, policy)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            if self._engine is None:
                return AnalysisFrame(engine="none")
            try:
                return self._engine.frame(now, policy)
            except Exception as exc2:  # noqa: BLE001
                self._degrade(exc2)
                return AnalysisFrame(engine="none")

    # diagnostics --------------------------------------------------------
    @property
    def healthy(self) -> bool:
        return self._engine is not None

    def __getattr__(self, item: str):
        """Delegate engine-specific metadata (bpm / duration / beat_confidence ...)."""
        engine = self.__dict__.get("_engine")
        if engine is not None:
            return getattr(engine, item)
        raise AttributeError(item)

    def describe(self) -> str:
        if not self.failures:
            return self.active_name
        short = self.failures[-1]
        if len(short) > 90:
            short = short[:87] + "..."
        return f"{self.active_name}（回退 {len(self.failures)} 次：{short}）"


def _instantiate(factory, args, kwargs):
    engine = factory(*args, **kwargs)
    if hasattr(engine, "healthy") and not getattr(engine, "healthy"):
        raise RuntimeError("engine reported unhealthy at construction")
    return engine


def _ensure_plugins() -> None:
    """Import plugin modules for their registration side effects (lazy: avoids
    a circular import and keeps a missing plugin from breaking startup)."""
    try:
        from ..ai import light_engine  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - classic remains available
        import sys
        print(f"[screenlight] AI engine unavailable, using classic: {exc}", file=sys.stderr)


def build_offline(samples: np.ndarray, sr: int, policy) -> SafeEngine:
    _ensure_plugins()
    return SafeEngine(OFFLINE_KIND, getattr(policy, "engine", "auto"),
                      _instantiate, samples, sr, policy)


def build_live(sr: int, policy) -> SafeEngine:
    _ensure_plugins()
    return SafeEngine(LIVE_KIND, getattr(policy, "engine", "auto"),
                      _instantiate, sr, policy)
