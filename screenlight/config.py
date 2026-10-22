"""Policy / config objects shared by every layer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PRESETS = ROOT / "presets"

SAFETY_LEVELS = ("off", "comfort", "strict")
MODES = ("dim", "canvas", "dj")


@dataclass
class Policy:
    mode: str = "dim"
    base: float = 0.08
    gain: float = 0.92
    attack_ms: float = 6.0
    decay_ms: float = 130.0
    gamma: float = 0.95
    max_dim: float = 0.96
    palette: list[str] = field(default_factory=lambda: ["#FF7A18", "#FF3D00", "#FFD166"])
    text: str = "看见声音"
    safety: str = "comfort"
    # --- v3 light-engine selection (all optional, old presets keep working) ---
    engine: str = "auto"          # auto | perceptual | classic
    beat_weight: float = -1.0     # -1 = auto by safety level
    curve_contrast: float = -1.0  # -1 = auto by safety level
    # v2.1 user light controls (replace the old safety cycle in the UI)
    sensitivity: float = -1.0     # -1 = derive from legacy safety preset
    base_brightness: float = 0.08  # brightness when the music is silent

    ENGINE_CHOICES = ("auto", "perceptual", "classic")

    def clamp(self) -> "Policy":
        self.base = float(min(max(self.base, 0.0), 1.0))
        self.gain = float(min(max(self.gain, 0.0), 1.0))
        self.attack_ms = float(min(max(self.attack_ms, 1.0), 2000.0))
        self.decay_ms = float(min(max(self.decay_ms, 1.0), 4000.0))
        self.gamma = float(min(max(self.gamma, 0.2), 3.0))
        self.max_dim = float(min(max(self.max_dim, 0.0), 0.98))
        if self.mode not in MODES:
            self.mode = "dim"
        if self.safety not in SAFETY_LEVELS:
            self.safety = "comfort"
        if not self.palette:
            self.palette = ["#FF7A18", "#FF3D00", "#FFD166"]
        if str(self.engine) not in self.ENGINE_CHOICES:
            self.engine = "auto"
        self.beat_weight = float(min(max(float(self.beat_weight), -1.0), 1.0))
        self.curve_contrast = float(min(max(float(self.curve_contrast), -1.0), 1.0))
        self.sensitivity = float(min(max(float(self.sensitivity), -1.0), 1.0))
        self.base_brightness = float(min(max(float(self.base_brightness), 0.0), 1.0))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known}).clamp()

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        p = Path(path)
        if not p.exists():
            return cls().clamp()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


DEFAULT_POLICY = Policy().clamp()
