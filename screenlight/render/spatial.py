"""Spatial light masks for OpenRay.

A single transparent full-screen canvas draws regional darkness with Tk
stipple halftones, so different parts of the desktop can dim at different
rates while keeping the rest of the screen visible.  Patterns:

* ``columns`` — four tall bands driven by low/mid/high/full spectrum energy;
* ``wave``    — a travelling sea of 16 light/dark strips;
* ``window``  — dark window panes with three diagonal light shafts.

``auto`` is AI-side: YAMNet scene + emotion choose the language and drive its
energy.  No extra runtime dependency and no per-pixel Win32 compositing.
"""
from __future__ import annotations

import ctypes
import math
import tkinter as tk

BG = "#010203"  # transparent colour key; never used by artwork
INK = "#000000"
WARM_LIGHT = "#FFF4D7"
COOL_LIGHT = "#EAF7FF"

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

PATTERN_LABELS = {"columns": "四列", "wave": "海浪", "window": "窗光",
                  "spotlight": "追光"}
PATTERNS = ("columns", "wave", "window", "spotlight")
# Tk's built-in stipple bitmaps are ordered by foreground (black) density.
STIPPLES = ("gray12", "gray25", "gray50", "gray75")


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def choose_pattern(scene: str, emotion: str, valence: float,
                   arousal: float) -> str:
    """AI-side mapping from scene/emotion to a spatial language."""
    if scene in ("applause", "cheer"):
        return "columns"
    if scene in ("speech",):
        return "window"
    if emotion in ("intense", "bright") and arousal > 0.62:
        return "spotlight"
    if scene in ("music",) and arousal > 0.70:
        return "spotlight"
    return "wave"


def pattern_alphas(pattern: str, level: float, bands, phase: float,
                   arousal: float = 0.5) -> list[float]:
    """Pure, testable darkness values (0=clear/bright, 1=black) per region."""
    level = _clamp(level)
    arr = []
    try:
        arr = [float(x) for x in bands]
    except Exception:  # noqa: BLE001
        arr = []
    if pattern == "columns":
        n = 4
        out = []
        for i in range(n):
            seg = arr[int(i * len(arr) / n):max(int((i + 1) * len(arr) / n), 1)] if arr else []
            band = sum(seg) / len(seg) if seg else level
            energy = _clamp(0.20 + 0.85 * band + 0.18 * level + 0.10 * arousal)
            out.append(_clamp(0.94 - 0.95 * energy))
        return out
    if pattern == "wave":
        n = 16
        out = []
        for i in range(n):
            wave = 0.5 + 0.5 * math.sin(phase * math.tau * 1.6 + i * 0.62)
            base = 0.72 - 0.45 * level
            out.append(_clamp(base + 0.20 * (wave - 0.5) * (0.5 + 0.5 * level)))
        return out
    # window: one darkness value for the pane (the shafts are holes)
    return [_clamp(0.86 - 0.62 * level)]


def _stipple(alpha: float) -> str | None:
    """Map a darkness value to a stipple bitmap, or None for clear."""
    if alpha <= 0.06:
        return None
    if alpha < 0.28:
        return STIPPLES[0]
    if alpha < 0.48:
        return STIPPLES[1]
    if alpha < 0.70:
        return STIPPLES[2]
    return STIPPLES[3]

class SpatialOverlay:
    """One transparent, click-through, topmost window with a Canvas."""

    def __init__(self, root: tk.Misc, width: int, height: int,
                 x: int = 0, y: int = 0) -> None:
        self.root = root
        self.width, self.height = width, height
        self.x, self.y = x, y
        self.enabled = False
        self.pattern = ""
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{width}x{height}+{x}+{y}")
        self.win.configure(bg=BG)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-transparentcolor", BG)
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(self.win, width=width, height=height, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self._clickthrough()
        self.win.withdraw()

    def _clickthrough(self) -> None:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.win.winfo_id())
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- render
    def show(self, pattern: str, level: float, bands, palette,
             valence: float = 0.5, arousal: float = 0.5,
             beat_phase: float = 0.0, emotion: str = "balanced") -> None:
        pattern = pattern if pattern in PATTERNS else "wave"
        if not self.enabled or self.pattern != pattern:
            self.pattern = pattern
            self.enabled = True
            self.canvas.delete("all")
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.win.lift()
        if pattern == "columns":
            self._columns(level, bands, arousal)
        elif pattern == "wave":
            self._wave(level, bands, arousal, beat_phase)
        elif pattern == "window":
            self._window(level, palette, valence, beat_phase)
        elif pattern == "spotlight":
            self._spotlight(level, bands, palette, valence, beat_phase, emotion)
        else:
            self._wave(level, bands, arousal, beat_phase)

    def _rect_dark(self, x0: float, y0: float, x1: float, y1: float,
                   alpha: float) -> None:
        st = _stipple(alpha)
        if st is None:
            return
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=INK, stipple=st, outline="")

    def _columns(self, level: float, bands, arousal: float) -> None:
        self.canvas.delete("all")
        vals = pattern_alphas("columns", level, bands, 0.0, arousal)
        for i, a in enumerate(vals):
            x0 = int(i * self.width / len(vals))
            x1 = int((i + 1) * self.width / len(vals))
            self._rect_dark(x0, 0, x1, self.height, a)
            if a < 0.42:
                self.canvas.create_line(x1, 0, x1, self.height,
                                        fill="#2A1E12", width=1)

    def _wave(self, level: float, bands, arousal: float, phase: float) -> None:
        self.canvas.delete("all")
        vals = pattern_alphas("wave", level, bands, phase, arousal)
        for i, a in enumerate(vals):
            x0 = int(i * self.width / len(vals))
            x1 = int((i + 1) * self.width / len(vals))
            self._rect_dark(x0, 0, x1, self.height, a)
            if a < 0.42:
                self.canvas.create_rectangle(x0, 0, x1, self.height, fill="#FFE0A6",
                                             stipple="gray12", outline="")
            if a < 0.32:
                self.canvas.create_line(x0, 0, x0, self.height,
                                        fill="#3A2A18", width=1)

    def _window(self, level: float, palette, valence: float, phase: float) -> None:
        self.canvas.delete("all")
        w, h = self.width, self.height
        shaft_w = w * 0.13
        slant = w * 0.23
        tops = []
        for i in range(3):
            base = (0.05 + i * 0.33 + 0.06 * math.sin(phase * math.tau + i * 0.9)) * w
            base = max(-0.10 * w, min(0.94 * w, base))
            tops.append(base)
        # dark complement panes, with the shafts left transparent
        left = tops[0]
        self._poly_dark([(0, 0), (left, 0), (left + slant, h), (0, h)], level)
        for i in range(3):
            a0 = tops[i] + shaft_w
            a0b = a0 + slant
            if i + 1 < 3:
                a1 = tops[i + 1]
                a1b = a1 + slant
                self._poly_dark([(a0, 0), (a1, 0), (a1b, h), (a0b, h)], level)
            else:
                self._poly_dark([(a0, 0), (w, 0), (w, h), (a0b, h)], level)
        # soft light inside the shafts
        for t in tops:
            pts = [(t, 0), (t + shaft_w, 0), (t + shaft_w + slant, h),
                   (t + slant, h)]
            self.canvas.create_polygon(pts, fill=WARM_LIGHT if valence >= 0.45 else COOL_LIGHT,
                                       stipple="gray12", outline="")
            self.canvas.create_line(t, 0, t + slant, h,
                                    fill="#FFE4B0" if valence >= 0.45 else "#CFEFFF",
                                    width=1)

    def _poly_dark(self, pts, level: float) -> None:
        a = _clamp(0.82 - 0.58 * level)
        st = _stipple(a)
        if st is None:
            return
        self.canvas.create_polygon(pts, fill=INK, stipple=st, outline="")

    # ------------------------------------------------------------ spotlight
    SPOT_PALETTES = {
        "bright": ("#FFD36B", "#FF9AC2", "#C084FC"),
        "warm": ("#FFB36B", "#FF7EA8", "#B78CFF"),
        "intense": ("#C084FC", "#EC4899", "#60A5FA"),
        "sad": ("#60A5FA", "#A78BFA", "#7FE3E7"),
        "balanced": ("#B78CFF", "#7FE3E7", "#FF9AC2"),
    }

    def _spotlight(self, level: float, bands, palette, valence: float,
                   phase: float, emotion: str = "balanced") -> None:
        """One translucent, colour-shifting stage beam follows the melody."""
        self.canvas.delete("all")
        w, h = self.width, self.height
        arr = [float(x) for x in bands] if bands is not None else []
        if arr:
            total = sum(arr) or 1.0
            centroid = sum(i * v for i, v in enumerate(arr)) / (total * max(len(arr) - 1, 1))
        else:
            centroid = 0.5
        colors = self.SPOT_PALETTES.get(
            emotion or "balanced",
            self.SPOT_PALETTES["balanced" if valence >= 0.45 else "sad"])
        idx = int((phase * 2.0 + self._t * 0.05) % len(colors)) if hasattr(self, "_t") else int(phase * 2) % len(colors)
        c1 = colors[idx % len(colors)]
        c2 = colors[(idx + 1) % len(colors)]
        c3 = colors[(idx + 2) % len(colors)]
        target = w * (0.12 + 0.76 * (centroid * 0.82 + 0.09))
        target += w * 0.035 * math.sin(phase * math.tau)
        target = _clamp(target, w * 0.10, w * 0.90)
        source = w * 0.5 + w * 0.10 * math.sin(phase * math.tau * 0.35)
        base = h * 0.97
        top_half = w * (0.012 + 0.010 * level)
        bot_half = w * (0.11 + 0.045 * level)

        def points(extra: float) -> list[tuple[float, float]]:
            return [(source - top_half - extra, 0), (source + top_half + extra, 0),
                    (target + bot_half + extra, base), (target - bot_half - extra, base)]

        # wide, very transparent outer glow; narrow colour core
        self.canvas.create_polygon(points(w * 0.055), fill=c1, stipple="gray12", outline="")
        self.canvas.create_polygon(points(w * 0.018), fill=c2, stipple="gray25", outline="")
        self.canvas.create_line(source - top_half, 0, target - bot_half, base,
                                fill=c3, width=1)
        self.canvas.create_line(source + top_half, 0, target + bot_half, base,
                                fill=c3, width=1)
        self.canvas.create_line(source, 0, target, base, fill=c2, width=2,
                                stipple="gray25")
        # soft pool on the floor
        self.canvas.create_oval(target - w * 0.13, base - h * 0.035,
                                target + w * 0.13, base + h * 0.035,
                                fill=c1, stipple="gray25", outline="")
        self.canvas.create_oval(target - w * 0.075, base - h * 0.018,
                                target + w * 0.075, base + h * 0.018,
                                fill=c2, stipple="gray50", outline="")
        # a few dust motes inside the beam
        for i in range(8):
            t = (i / 8 + phase * 0.7) % 1.0
            x = source + (target - source) * t
            y = base * t
            x += math.sin(phase * math.tau + i * 1.3) * w * 0.018
            self.canvas.create_line(x - 1, y, x + 2, y, fill=c3, width=1)

    def hide(self) -> None:
        self.enabled = False
        self.pattern = ""
        try:
            self.win.withdraw()
        except tk.TclError:
            pass

    def lift(self) -> None:
        try:
            self.win.lift()
        except tk.TclError:
            pass

    def close(self) -> None:
        self.hide()
        try:
            self.win.destroy()
        except tk.TclError:
            pass
