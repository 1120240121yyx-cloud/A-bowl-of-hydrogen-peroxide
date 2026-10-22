"""The one data contract shared by analysis -> policy -> render."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_BANDS = 24


def _zeros(n: int = N_BANDS) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


@dataclass
class AnalysisFrame:
    """Everything the renderer needs for one display frame.

    The renderer must not look at raw audio or policy internals; if it needs a
    new signal, add it here (and to every engine) instead of reaching into an
    engine.
    """

    t: float = 0.0
    level: float = 0.0          # 0..1 target brightness, already policy-shaped
    bands: np.ndarray = field(default_factory=_zeros)   # (24,) 0..1 canvas bars
    bpm: float = 0.0
    beat_phase: float = 0.0     # 0..1 position inside the current beat
    energy: float = 0.0         # 0..1 slow continuous energy (breathing)
    accent: float = 0.0         # 0..1 beat accent / punch
    beat_confidence: float = 0.0  # 0..1; low => smooth/beatless material
    scene: str = "music"
    scene_confidence: float = 0.0
    wake: bool = False
    engine: str = "classic"
    # continuous musical emotion (0..1) + coarse label for the UI
    valence: float = 0.5
    arousal: float = 0.5
    emotion: str = "balanced"

    def clamped(self) -> "AnalysisFrame":
        self.level = min(max(float(self.level), 0.0), 1.0)
        self.energy = min(max(float(self.energy), 0.0), 1.0)
        self.accent = min(max(float(self.accent), 0.0), 1.0)
        self.beat_confidence = min(max(float(self.beat_confidence), 0.0), 1.0)
        self.beat_phase = float(self.beat_phase) % 1.0
        self.bpm = max(float(self.bpm), 0.0)
        self.valence = min(max(float(self.valence), 0.0), 1.0)
        self.arousal = min(max(float(self.arousal), 0.0), 1.0)
        if not isinstance(self.emotion, str) or not self.emotion:
            self.emotion = "balanced"
        b = np.asarray(self.bands, dtype=np.float32).reshape(-1)
        if b.size != N_BANDS:
            b = np.resize(b, N_BANDS) if b.size else _zeros()
        self.bands = np.clip(b, 0.0, 1.0)
        return self
