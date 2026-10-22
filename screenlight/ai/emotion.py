"""Emotion / energy curve (P2 in the guide: ''温柔 / 炸裂 两种响应曲线'').

CLAP zero-shot would be the heavyweight option; here we derive a compact
*valence / arousal* estimate from the light engine's own analysis and map it to
two response curves.  The optional ``CLAP`` backend hook is kept so a model can
be dropped in later without touching app.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionProfile:
    label: str            # gentle | balanced | explosive
    valence: float        # 0 sad .. 1 happy
    arousal: float        # 0 calm .. 1 intense
    beat_weight: float    # suggested engine beat weight
    curve_contrast: float # suggested S-curve contrast
    accent_gain: float    # multiplier for canvas / dj accents
    source: str = "heuristic"

    def describe(self) -> str:
        return (f"{self.label} v={self.valence:.2f} a={self.arousal:.2f} "
                f"({self.source})")


# curves keyed by label; values tuned against the demo sources
CURVES = {
    "gentle": dict(beat_weight=0.22, curve_contrast=-0.18, accent_gain=0.55),
    "balanced": dict(beat_weight=0.0, curve_contrast=0.0, accent_gain=1.0),
    "explosive": dict(beat_weight=0.58, curve_contrast=0.22, accent_gain=1.45),
    # per-frame quadrants (v2, used by the emotion-driven brightness contour)
    "bright": dict(beat_weight=0.45, curve_contrast=0.15, accent_gain=1.20),
    "warm": dict(beat_weight=0.10, curve_contrast=-0.05, accent_gain=0.80),
    "intense": dict(beat_weight=0.60, curve_contrast=0.25, accent_gain=1.50),
    "sad": dict(beat_weight=0.18, curve_contrast=-0.22, accent_gain=0.45),
}


def estimate(features: dict) -> EmotionProfile:
    """``features`` may contain: level_mean/std, flux_mean, bpm, low_ratio,
    flatness, transient_density (all optional, 0..1 except bpm)."""
    level = float(features.get("level_mean", 0.4))
    level_std = float(features.get("level_std", 0.1))
    flux = float(features.get("flux_mean", 0.2))
    bpm = float(features.get("bpm", 100.0))
    low = float(features.get("low_ratio", 0.3))
    flat = float(features.get("flatness", 0.3))
    density = float(features.get("transient_density", 0.1))

    arousal = np.clip(0.35 * level + 0.25 * flux + 0.15 * density
                      + 0.15 * min(bpm / 180.0, 1.0) + 0.10 * level_std * 4.0,
                      0.0, 1.0)
    # major-ish brightness (low-band energy + tonal flatness inverted) as valence
    valence = np.clip(0.55 - 0.35 * flat + 0.25 * low + 0.10 * level,
                      0.0, 1.0)
    if arousal < 0.34:
        label = "gentle"
    elif arousal > 0.66:
        label = "explosive"
    else:
        label = "balanced"
    curve = CURVES[label]
    return EmotionProfile(label=label, valence=float(valence),
                          arousal=float(arousal), source="heuristic",
                          **curve)


def from_track_curve(curve) -> EmotionProfile:
    """Estimate from a :class:`TrackCurve`-like object (offline analysis)."""
    try:
        levels = np.asarray(getattr(curve, "energy", []), dtype=np.float32)
        accent = np.asarray(getattr(curve, "accent", []), dtype=np.float32)
        bands = np.asarray(getattr(curve, "bands", np.zeros((0, 1))),
                           dtype=np.float32)
    except Exception:  # noqa: BLE001
        levels, accent, bands = np.zeros(0), np.zeros(0), np.zeros((0, 1))
    low_ratio = 0.3
    if bands.size and bands.ndim == 2 and bands.shape[1] >= 8:
        # first bands are the low end in the v3 feature set
        low_ratio = float(bands[:, :4].sum() / (bands.sum() + 1e-9))
    feats = {
        "level_mean": float(levels.mean()) if levels.size else 0.4,
        "level_std": float(levels.std()) if levels.size else 0.1,
        "flux_mean": float(accent.mean()) if accent.size else 0.2,
        "low_ratio": float(np.clip(low_ratio, 0.0, 1.0)),
        "bpm": float(getattr(curve, "bpm", 100.0) or 100.0),
    }
    return estimate(feats)


class ClapEmotion:
    """Placeholder for an optional CLAP ONNX zero-shot classifier.

    Instantiate only when a model is present; ``available`` stays False here so
    the heuristic is always used.  Kept intentionally tiny: the rest of the app
    only needs ``estimate`` to return an :class:`EmotionProfile`.
    """

    available = False

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("CLAP backend not installed in this build")


# per-quadrant response curves used by app.py every frame.
# brightness contour shape + accent multiplier, on top of the user policy.
QUADRANT_CURVES = {
    "bright":  dict(attack_ms=14.0, decay_ms=150.0, gain=1.06, accent_gain=1.20),
    "warm":    dict(attack_ms=90.0, decay_ms=520.0, gain=0.94, accent_gain=0.85),
    "intense": dict(attack_ms=5.0,  decay_ms=95.0,  gain=1.10, accent_gain=1.50),
    "sad":     dict(attack_ms=220.0, decay_ms=1200.0, gain=0.70, accent_gain=0.45),
    "balanced": dict(attack_ms=40.0, decay_ms=300.0, gain=1.00, accent_gain=1.00),
}


def curve_for(valence: float, arousal: float) -> dict:
    labels = CURVES  # keep the summary table in sync
    label = ("bright" if valence >= 0.52 and arousal >= 0.48 else
             "warm" if valence >= 0.52 else
             "intense" if arousal >= 0.48 else
             "sad" if valence < 0.43 else "balanced")
    out = dict(QUADRANT_CURVES[label])
    out["label"] = label
    return out


def profile_for(valence: float, arousal: float) -> EmotionProfile:
    label = curve_for(valence, arousal)["label"]
    curve = CURVES.get(label, CURVES["balanced"])
    return EmotionProfile(label=label, valence=float(valence),
                          arousal=float(arousal), source="per-frame",
                          **curve)


# Natural-language mood intent used by the Agent text box / voice commands.
# This is deliberately local and instant: saying "悲伤一点" must change the
# brightness contour immediately, without waiting for an LLM round-trip.
_INTENTS = (
    ("sad", ("悲伤", "难过", "忧伤", "丧", "低落", "emo", "孤独", "别太亮"),
     dict(label="sad", valence=0.30, arousal=0.32)),
    ("warm", ("温柔", "治愈", "安静", "平静", "放松", "睡前", "呼吸", "暖"),
     dict(label="warm", valence=0.62, arousal=0.32)),
    ("bright", ("开心", "快乐", "明亮", "阳光", "活泼", "甜", "元气"),
     dict(label="bright", valence=0.72, arousal=0.62)),
    ("intense", ("激烈", "炸", "燃", "愤怒", "狂", "嗨", "强烈", "躁"),
     dict(label="intense", valence=0.46, arousal=0.82)),
)


def intent_for(text: str) -> dict | None:
    low = (text or "").lower()
    for _name, words, target in _INTENTS:
        if any(w in low for w in words):
            return dict(target)
    return None
