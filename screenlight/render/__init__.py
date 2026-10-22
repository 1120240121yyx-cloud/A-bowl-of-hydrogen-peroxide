"""Renderer sub-package: animated full-screen light-show scenes."""
from __future__ import annotations

from .scene_canvas import CanvasScene

try:  # DJ mode is optional: a broken scene must not break canvas mode
    from .scene_dj import DJScene
except Exception:  # pragma: no cover - defensive
    DJScene = None  # type: ignore[assignment]

__all__ = ["CanvasScene", "DJScene"]
