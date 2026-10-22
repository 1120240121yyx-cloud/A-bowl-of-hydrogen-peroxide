"""Brightness mapping + user-controlled safety limiter."""
from __future__ import annotations

import math


SAFETY_PRESETS = {
    # max_hz: 0 == unlimited     slew: max level change per second
    "off":     {"max_hz": 0.0, "max_delta": 1.00, "min_ramp_ms": 0,   "slew": 80.0},
    "comfort": {"max_hz": 3.0, "max_delta": 0.40, "min_ramp_ms": 80,  "slew": 5.0},
    "strict":  {"max_hz": 1.0, "max_delta": 0.20, "min_ramp_ms": 200, "slew": 1.0},
}

SAFETY_LABELS = {"off": "关闭", "comfort": "舒适", "strict": "严格"}


class SafetyLimiter:
    """Smoothly rate-limits the brightness signal.

    Two stages:
      1. one-pole low-pass with cutoff = max_hz (0 disables it)
      2. slew-rate limiter (max level change per second)

    The user owns the switch: `off` removes both stages, while the
    emergency stop (Esc) always works.
    """

    def __init__(self, level: str = "comfort") -> None:
        self.level = level if level in SAFETY_PRESETS else "comfort"
        self.value = 0.0
        self._lp = 0.0
        self.last_t: float | None = None
        self.clamped_events = 0

    @property
    def params(self) -> dict:
        return SAFETY_PRESETS[self.level]

    def set_level(self, level: str) -> None:
        if level in SAFETY_PRESETS:
            self.level = level

    def reset(self, value: float = 0.0) -> None:
        self.value = value
        self._lp = value
        self.last_t = None

    def step(self, target: float, now: float) -> float:
        target = min(max(target, 0.0), 1.0)
        p = self.params
        if self.last_t is None:
            self.last_t = now
            self.value = target
            self._lp = target
            return self.value

        dt = max(now - self.last_t, 1e-4)
        self.last_t = now

        # stage 1: low-pass. Cutoff = max_hz (0 -> bypass)
        if p["max_hz"] > 0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * p["max_hz"] * dt)
            self._lp += alpha * (target - self._lp)
            target = self._lp
        else:
            self._lp = target

        # stage 2: slew rate
        max_step = p["slew"] * dt
        delta = target - self.value
        if abs(delta) > max_step:
            delta = math.copysign(max_step, delta)
            self.clamped_events += 1
        self.value = min(max(self.value + delta, 0.0), 1.0)
        return self.value


# ---------------------------------------------------------------- color helpers

def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        return (255, 255, 255)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(c))) for c in rgb))


def mix(a: str, b: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return rgb_to_hex((ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t))


def palette_color(palette: list[str], t: float) -> str:
    if not palette:
        return "#FFFFFF"
    if len(palette) == 1:
        return palette[0]
    t = min(max(t, 0.0), 1.0) * (len(palette) - 1)
    i = min(int(t), len(palette) - 2)
    return mix(palette[i], palette[i + 1], t - i)


def scale_color(color: str, level: float) -> str:
    """Dim a color without changing the window alpha."""
    r, g, b = hex_to_rgb(color)
    k = min(max(level, 0.0), 1.0)
    k = k ** 0.8
    return rgb_to_hex((r * k, g * k, b * k))


# ---------------------------------------------------------------- emotion colour
WARM_TARGET = "#FFB36B"
COOL_TARGET = "#6EA8FF"


def emotion_palette(palette: list[str], valence: float,
                    arousal: float = 0.5, strength: float = 0.38
                    ) -> list[str]:
    """Shift a palette warm/cool with musical valence (0..1).

    ``arousal`` only scales saturation a little: intense tracks get a slightly
    richer shift, sad tracks a colder and softer one.  Kept in sRGB because the
    renderer works with plain hex strings.
    """
    v = min(max(float(valence), 0.0), 1.0)
    a = min(max(float(arousal), 0.0), 1.0)
    amount = min(1.0, abs(v - 0.5) * 2.0) * strength * (0.75 + 0.5 * a)
    target = WARM_TARGET if v >= 0.5 else COOL_TARGET
    try:
        return [mix(c, target, amount) for c in palette]
    except Exception:  # noqa: BLE001 - palette override must never crash
        return list(palette)


BURST_THRESHOLD = 0.90


def burst_is_white(level: float, threshold: float = BURST_THRESHOLD) -> bool:
    """Exact threshold for the optional strobe module (90% brightness)."""
    return float(level) >= float(threshold)


def apply_light_controls(response: float, base: float,
                         sensitivity: float, silence: float = 0.0) -> float:
    """User brightness controls without touching the response shape.

    - ``response`` is the already-shaped engine/emotion level (timing intact);
    - ``silence`` is the engine output during silence (estimated externally);
    - the dynamic part above silence is scaled by ``sensitivity``;
    - output is clamped to ``[base, 1]`` so silence reads exactly ``base``.

    The mapping is monotone: peaks/crossings happen at exactly the same time;
    only the amplitude changes.  sensitivity=0 => constant base;
    sensitivity=1 and base=0 => response normalized from its silent floor.
    """
    r = min(max(float(response), 0.0), 1.0)
    b = min(max(float(base), 0.0), 1.0)
    s = min(max(float(sensitivity), 0.0), 1.0)
    q = min(max(float(silence), 0.0), 0.99)
    dyn = min(max((r - q) / max(1e-6, 1.0 - q), 0.0), 1.0)
    return float(min(max(b + s * (1.0 - b) * dyn, 0.0), 1.0))
