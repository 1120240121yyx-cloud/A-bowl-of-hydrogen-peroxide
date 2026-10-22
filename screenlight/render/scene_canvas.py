"""Animated light-show scene for OpenRay's Tk canvas mode.

Tk canvas items have no alpha channel, so every "translucent" layer is faked by
mixing its colour towards the backdrop colour it sits on.  The scene creates a
fixed set of canvas items once, then only moves / recolours them in place.

Layers (back to front):
  * vertical gradient backdrop (``_BG_STEPS`` full-width rectangles)
  * stage glow (nested ovals) + beat pulse oval
  * two expanding beat rings (thin outlined ovals)
  * thin stage floor line
  * 28 spectrum capsules, three layers each: floor reflection, wide glow,
    main rounded bar (``create_line`` + ``capstyle="round"``)
  * slow-falling VU peak caps (thin bright horizontal capsules)
  * small rounded caption text

Performance notes:
  * coordinates are quantised to ``int`` and only written when they change
  * per-item colours are cached as strings and only written when they change
  * bar colours are derived from a 64-step brightness quantisation cache
  * per-frame state is EMA-smoothed; nothing allocates per item
"""
from __future__ import annotations

import math
import time
import tkinter as tk
import tkinter.font as tkfont

import numpy as np

from ..mapping import mix, palette_color, scale_color

# ---------------------------------------------------------------- tuning
_ATTACK_S = 0.040          # bar attack time constant (s)
_RELEASE_S = 0.180         # bar release time constant (s)
_LEVEL_ATTACK_S = 0.030    # backdrop brightness attack
_LEVEL_RELEASE_S = 0.120   # backdrop brightness release
_RING_ATTACK_S = 0.045     # beat-ring brightness attack
_RING_RELEASE_S = 0.240    # beat-ring brightness release
_PEAK_FALL = 0.50          # VU peak fall speed, screen heights per second
_RING_DELTA_CAP = 0.12     # max ring brightness change per frame (anti-strobe)
_BG_STEPS = 40             # gradient rectangles (smooth luminance ramp)
_GLOW_LAYERS = 8           # nested stage-glow ovals (soft radial falloff)
_RING_COUNT = 2            # expanding beat rings
_BRIGHT_STEPS = 63         # bar colour quantisation steps

_DEFAULT_PALETTE = ["#FF7A18", "#FF3D00", "#FFD166"]

# every item created by the scene carries this tag so the overlay can clean up
# a half-built scene if construction ever fails
_SCENE_TAG = "sl_scene"


def _clamp01(value: float) -> float:
    """Clamp to [0, 1]; NaN becomes 0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not (value > 0.0):  # also catches NaN
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _capsule_span(floor: int, height: int, width: int) -> tuple[int, int]:
    """Vertical endpoint pair for a round-capped line whose far edge is at
    ``floor - height`` and whose near cap ends at ``floor``."""
    half = width // 2
    y_near = floor - half
    if height < width:
        return y_near, y_near
    return floor - height + half, y_near


class CanvasScene:
    """Plain-Tk animated spectrum scene (no alpha, no images, no PIL)."""

    def __init__(self, canvas: "tk.Canvas", width: int, height: int,
                 n_bars: int = 28) -> None:
        self.canvas = canvas
        self.width = max(2, int(width))
        self.height = max(2, int(height))
        self.n_bars = max(4, int(n_bars))

        w, h, n = self.width, self.height, self.n_bars
        self._floor = int(h * 0.86)
        self._slot = w / float(n)
        self._x = [int((i + 0.5) * self._slot) for i in range(n)]
        self._pal_t = [i / float(max(n - 1, 1)) for i in range(n)]

        self._bar_w = max(5, int(self._slot * 0.40))
        self._glow_w = max(self._bar_w + 6, int(self._slot * 0.90))
        self._refl_w = max(3, int(self._bar_w * 0.55))
        self._cap_w = max(2, int(self._bar_w * 0.20))
        self._cap_half = max(3, int(self._bar_w * 0.45))

        self._min_dim = min(w, h)
        self._ring_r0 = max(8, int(self._min_dim * 0.10))
        self._ring_span = max(16, int(self._min_dim * 0.44))
        self._ring_width = max(2, int(self._min_dim / 700.0))

        # static glow geometry (outer -> inner)
        self._glow_cx = w // 2
        self._glow_cy = self._floor - int(h * 0.05)
        self._glow_rx = [max(8, int(w * 0.40 * (1.0 - 0.12 * k))) for k in range(_GLOW_LAYERS)]
        self._glow_ry = [max(8, int(h * 0.32 * (1.0 - 0.12 * k))) for k in range(_GLOW_LAYERS)]

        # ---- current palette / derived colours --------------------------
        self._pal_key: tuple[str, ...] | None = None
        self._pal_base: list[str] = [_DEFAULT_PALETTE[0]] * n
        self._wash = "#202020"
        self._ring_hot = "#FFD166"
        self._glow_hot = "#5A4A3A"
        self._hint_base = "#C0C0C0"

        # ---- animation state --------------------------------------------
        self._vals: list[float] = [0.0] * n
        self._peaks: list[float] = [0.0] * n
        self._raw: list[float] = [0.0] * n
        self._heights: list[int] = [0] * n
        self._peak_h: list[int] = [0] * n
        self._level = 0.0
        self._ring_env = 0.0
        self._last_t: float | None = None
        self._visible = True

        # colour cache: last string written per item
        self._last_q: list[int] = [-1] * n
        self._bar_colors: list[tuple[str, str, str, str]] = [
            ("#000000", "#000000", "#000000", "#000000") for _ in range(n)
        ]
        self._last_bar: list[str | None] = [None] * n
        self._last_glowline: list[str | None] = [None] * n
        self._last_refl: list[str | None] = [None] * n
        self._last_cap: list[str | None] = [None] * n
        self._last_bg: list[str | None] = [None] * _BG_STEPS
        self._last_glowo: list[str | None] = [None] * _GLOW_LAYERS
        self._last_accent: str | None = None
        self._last_ringc: list[str | None] = [None] * _RING_COUNT
        self._last_floor = None
        self._last_hint_color = None
        self._last_hint_text: str | None = None

        # coordinate cache
        self._p_bar: list[tuple[int, int, int] | None] = [None] * n
        self._p_glowline: list[tuple[int, int, int] | None] = [None] * n
        self._p_refl: list[tuple[int, int, int] | None] = [None] * n
        self._p_cap: list[tuple[int, int, int, int] | None] = [None] * n
        self._p_glowo: list[tuple[int, int, int, int] | None] = [None] * _GLOW_LAYERS
        self._p_accent: tuple[int, int, int, int] | None = None
        self._p_ring: list[tuple[int, int, int, int] | None] = [None] * _RING_COUNT
        self._ring_vis: list[bool] = [False] * _RING_COUNT
        self._accent_vis = False

        self._resample_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # -------------------------------------------------- build items
        c = canvas
        tag = _SCENE_TAG
        self._bg_items: list[int] = []
        for j in range(_BG_STEPS):
            y0 = int(h * j / _BG_STEPS)
            y1 = int(h * (j + 1) / _BG_STEPS)
            self._bg_items.append(
                c.create_rectangle(0, y0, w, y1, fill="#000000", outline="", tags=tag)
            )

        self._glow_items: list[int] = []
        for _ in range(_GLOW_LAYERS):
            self._glow_items.append(
                c.create_oval(0, 0, 0, 0, fill="#000000", outline="", tags=tag)
            )
        self._accent_item = c.create_oval(0, 0, 0, 0, fill="#000000", outline="",
                                          state="hidden", tags=tag)

        self._ring_items: list[int] = [
            c.create_oval(0, 0, 0, 0, fill="", outline="#000000",
                          width=self._ring_width, state="hidden", tags=tag)
            for _ in range(_RING_COUNT)
        ]

        self._floor_item = c.create_line(
            0, self._floor, w, self._floor, fill="#000000", width=1, tags=tag
        )

        self._refl: list[int] = []
        self._glow_lines: list[int] = []
        self._bars: list[int] = []
        self._caps: list[int] = []
        for i in range(n):
            x = self._x[i]
            self._refl.append(
                c.create_line(x, self._floor, x, self._floor, width=self._refl_w,
                              capstyle="round", fill="#000000", tags=tag)
            )
            self._glow_lines.append(
                c.create_line(x, self._floor, x, self._floor, width=self._glow_w,
                              capstyle="round", fill="#000000", tags=tag)
            )
            self._bars.append(
                c.create_line(x, self._floor, x, self._floor, width=self._bar_w,
                              capstyle="round", fill="#000000", tags=tag)
            )
            hw = self._cap_half
            self._caps.append(
                c.create_line(x - hw, self._floor, x + hw, self._floor,
                              width=self._cap_w, capstyle="round", fill="#000000",
                              tags=tag)
            )

        try:
            self._hint_font = tkfont.Font(
                family="Microsoft YaHei", size=-max(10, int(h * 0.017))
            )
        except Exception:  # pragma: no cover - font backend fallback
            self._hint_font = ("TkDefaultFont",)
        hint_y = int(h * 0.045)
        self._hint_y = hint_y
        self._hint_shadow = c.create_text(
            w // 2 + 1, hint_y + 1, text="", fill="#000000", font=self._hint_font,
            state="hidden", tags=tag,
        )
        self._hint_text = c.create_text(
            w // 2, hint_y, text="", fill="#000000", font=self._hint_font,
            state="hidden", tags=tag,
        )

        self._all_items: list[int] = (
            self._bg_items
            + self._glow_items
            + [self._accent_item]
            + self._ring_items
            + [self._floor_item]
            + self._refl
            + self._glow_lines
            + self._bars
            + self._caps
            + [self._hint_shadow, self._hint_text]
        )

    # ------------------------------------------------------------ public API
    def set_visible(self, visible: bool) -> None:
        """Show or hide every item owned by this scene."""
        visible = bool(visible)
        if visible == self._visible:
            return
        self._visible = visible
        state = "normal" if visible else "hidden"
        itemconfig = self.canvas.itemconfig
        for item in self._all_items:
            itemconfig(item, state=state)
        # rings / beat pulse only show while a beat is actually decaying;
        # re-decide on the next frame
        for k, item in enumerate(self._ring_items):
            if visible:
                itemconfig(item, state="hidden")
                self._ring_vis[k] = False
        if visible:
            itemconfig(self._accent_item, state="hidden")
            self._accent_vis = False
        if not visible:
            self._ring_vis = [False] * _RING_COUNT
            self._accent_vis = False

    def reset(self) -> None:
        """Drop all smoothed state and colour caches (next frame redraws all)."""
        n = self.n_bars
        self._vals = [0.0] * n
        self._peaks = [0.0] * n
        self._heights = [0] * n
        self._peak_h = [0] * n
        self._level = 0.0
        self._ring_env = 0.0
        self._last_t = None
        self._last_q = [-1] * n
        self._bar_colors = [("#000000", "#000000", "#000000", "#000000") for _ in range(n)]
        self._last_bar = [None] * n
        self._last_glowline = [None] * n
        self._last_refl = [None] * n
        self._last_cap = [None] * n
        self._last_bg = [None] * _BG_STEPS
        self._last_glowo = [None] * _GLOW_LAYERS
        self._last_accent = None
        self._last_ringc = [None] * _RING_COUNT
        self._last_floor = None
        self._last_hint_color = None
        self._last_hint_text = None
        self._p_bar = [None] * n
        self._p_glowline = [None] * n
        self._p_refl = [None] * n
        self._p_cap = [None] * n
        self._p_glowo = [None] * _GLOW_LAYERS
        self._p_accent = None
        self._p_ring = [None] * _RING_COUNT

    # ------------------------------------------------------------ frame
    def update(self, level: float, bands, palette: list[str], hint: str,
               accent: float = 0.0, beat_phase: float = 0.0) -> None:
        """Draw one frame.  ``bands`` may be any 1-D array-like (NaN safe)."""
        c = self.canvas
        itemconfig = c.itemconfig
        coords = c.coords
        w, h, n = self.width, self.height, self.n_bars

        now = time.perf_counter()
        if self._last_t is None:
            dt = 1.0 / 60.0
        else:
            dt = now - self._last_t
            if dt < 1e-4:
                dt = 1e-4
            elif dt > 0.25:
                dt = 0.25
        self._last_t = now

        level = _clamp01(level)
        accent = _clamp01(accent)
        try:
            phase = float(beat_phase) % 1.0
        except (TypeError, ValueError):
            phase = 0.0
        if not (0.0 <= phase <= 1.0):
            phase = 0.0

        # ------------------------------------------------ palette switch
        if not palette:
            palette = list(_DEFAULT_PALETTE)
        pkey = tuple(palette)
        if pkey != self._pal_key:
            self._pal_key = pkey
            self._pal_base = [palette_color(palette, t) for t in self._pal_t]
            wash = mix(palette[0], "#FFFFFF", 0.32)
            self._wash = wash
            self._ring_hot = mix(palette_color(palette, 0.78), "#FFFFFF", 0.32)
            self._glow_hot = mix(wash, "#FFFFFF", 0.12)
            self._hint_base = mix(wash, "#FFFFFF", 0.45)
            # force every cached colour to be rewritten next frame
            self._last_q = [-1] * n
            self._last_bar = [None] * n
            self._last_glowline = [None] * n
            self._last_refl = [None] * n
            self._last_cap = [None] * n
            self._last_bg = [None] * _BG_STEPS
            self._last_glowo = [None] * _GLOW_LAYERS
            self._last_accent = None
            self._last_ringc = [None] * _RING_COUNT
            self._last_floor = None
            self._last_hint_color = None
        wash = self._wash

        # ------------------------------------------------ smooth level
        target = level
        tau = _LEVEL_ATTACK_S if target > self._level else _LEVEL_RELEASE_S
        self._level += (1.0 - math.exp(-dt / tau)) * (target - self._level)
        lv = self._level

        # -------------------------------------------- beat-ring envelope
        tau = _RING_ATTACK_S if accent > self._ring_env else _RING_RELEASE_S
        nv = self._ring_env + (1.0 - math.exp(-dt / tau)) * (accent - self._ring_env)
        delta = nv - self._ring_env
        if delta > _RING_DELTA_CAP:            # hard cap => no strobing
            delta = _RING_DELTA_CAP
        elif delta < -_RING_DELTA_CAP:
            delta = -_RING_DELTA_CAP
        self._ring_env = _clamp01(self._ring_env + delta)
        renv = self._ring_env

        # ------------------------------------------- backdrop gradient
        top = mix("#000000", wash, 0.03 + 0.12 * lv)
        bottom = mix("#000000", wash, 0.10 + 0.62 * lv)
        bg_mid = mix(top, bottom, 0.70)
        bg_floor = mix(top, bottom, 0.86)
        last_bg = self._last_bg
        bg_items = self._bg_items
        if _BG_STEPS > 1:
            step = 1.0 / (_BG_STEPS - 1)
            for j in range(_BG_STEPS):
                color = mix(top, bottom, (j * step) ** 1.7)
                if color != last_bg[j]:
                    itemconfig(bg_items[j], fill=color)
                    last_bg[j] = color

        # ------------------------------------------------ spectrum state
        raw = self._resample(bands)
        vals = self._vals
        att = 1.0 - math.exp(-dt / _ATTACK_S)
        rel = 1.0 - math.exp(-dt / _RELEASE_S)
        if n == 1:
            vals[0] += (att if raw[0] > vals[0] else rel) * (raw[0] - vals[0])
        elif n > 1:
            for i in range(n):
                if i == 0:
                    target_i = 0.72 * raw[0] + 0.28 * raw[1]
                elif i == n - 1:
                    target_i = 0.72 * raw[n - 1] + 0.28 * raw[n - 2]
                else:
                    target_i = 0.16 * raw[i - 1] + 0.68 * raw[i] + 0.16 * raw[i + 1]
                cur = vals[i]
                vals[i] = cur + (att if target_i > cur else rel) * (target_i - cur)

        fall = _PEAK_FALL * h * dt
        peaks = self._peaks
        heights = self._heights
        peak_h = self._peak_h
        amp = 0.64 * h * (0.50 + 0.50 * lv)
        for i in range(n):
            p = peaks[i] - fall
            v = vals[i]
            if p < v:                     # peak caps and bars both quantised later
                p = v
            peaks[i] = p
            heights[i] = v * amp
            peak_h[i] = p * amp

        # ---------------------------------------------- spectrum capsules
        floor_y = self._floor
        xs = self._x
        bar_w = self._bar_w
        glow_w = self._glow_w
        refl_w = int(max(3, bar_w * 0.55))
        refl_room = min(h - floor_y - 1, max(2, int(h * 0.09)))
        bar_items = self._bars
        glow_items = self._glow_lines
        refl_items = self._refl
        cap_items = self._caps
        pal = self._pal_base
        last_bar = self._last_bar
        last_glow = self._last_glowline
        last_refl = self._last_refl
        last_cap = self._last_cap
        last_q = self._last_q
        bar_colors = self._bar_colors
        p_bar = self._p_bar
        p_glow = self._p_glowline
        p_refl = self._p_refl
        p_cap = self._p_cap
        level_gain = 0.34 + 0.66 * lv
        accent_gain = 1.0 + 0.18 * renv
        for i in range(n):
            x = xs[i]
            hgt = int(heights[i])
            if hgt < 0:
                hgt = 0
            bright = level_gain * (0.60 + 0.40 * vals[i]) * accent_gain
            if bright > 1.0:
                bright = 1.0
            q = int(bright * _BRIGHT_STEPS)
            if q != last_q[i]:
                main_c = scale_color(pal[i], bright)
                colors = (
                    main_c,
                    scale_color(pal[i], bright * 0.45),
                    scale_color(pal[i], bright * 0.14),
                    mix(main_c, "#FFFFFF", 0.18 + 0.50 * bright),
                )
                bar_colors[i] = colors
                last_q[i] = q

            main_c, glow_c, refl_c, cap_c = bar_colors[i]
            if main_c != last_bar[i]:
                itemconfig(bar_items[i], fill=main_c)
                last_bar[i] = main_c
            if glow_c != last_glow[i]:
                itemconfig(glow_items[i], fill=glow_c)
                last_glow[i] = glow_c
            if refl_c != last_refl[i]:
                itemconfig(refl_items[i], fill=refl_c)
                last_refl[i] = refl_c
            if cap_c != last_cap[i]:
                itemconfig(cap_items[i], fill=cap_c)
                last_cap[i] = cap_c

            # main capsule
            ytop, ybot = _capsule_span(floor_y, hgt, bar_w)
            pos = (x, ytop, ybot)
            if pos != p_bar[i]:
                coords(bar_items[i], x, ytop, x, ybot)
                p_bar[i] = pos

            # wide dim glow halo
            gtop, gbot = _capsule_span(floor_y, hgt, glow_w)
            pos = (x, gtop, gbot)
            if pos != p_glow[i]:
                coords(glow_items[i], x, gtop, x, gbot)
                p_glow[i] = pos

            # mirrored floor reflection (short, dim, clipped to the floor gap)
            rh = int(hgt * 0.22)
            if rh > refl_room:
                rh = refl_room
            half_r = refl_w // 2
            r_top = floor_y + half_r
            r_bot = r_top if rh < refl_w else floor_y + rh - half_r
            pos = (x, r_top, r_bot)
            if pos != p_refl[i]:
                coords(refl_items[i], x, r_top, x, r_bot)
                p_refl[i] = pos

            # slow-falling VU peak cap
            ghost = int(peak_h[i])
            ycap = floor_y - ghost
            if ycap < 2:
                ycap = 2
            pos = (x - self._cap_half, ycap, x + self._cap_half, ycap)
            if pos != p_cap[i]:
                coords(cap_items[i], pos[0], pos[1], pos[2], pos[3])
                p_cap[i] = pos

        # ------------------------------------------------ stage floor line
        floor_color = mix(bg_floor, wash, 0.05 + 0.10 * lv)
        if floor_color != self._last_floor:
            itemconfig(self._floor_item, fill=floor_color)
            self._last_floor = floor_color

        # ------------------------------------------------- stage glow
        glow_level = lv ** 0.85
        pulse = 0.80 + 0.16 * lv + 0.08 * renv
        gcx = self._glow_cx
        gcy = self._glow_cy
        glow_items_o = self._glow_items
        glow_hot = self._glow_hot
        p_glowo = self._p_glowo
        last_o = self._last_glowo
        for k in range(_GLOW_LAYERS):
            rx = int(self._glow_rx[k] * pulse)
            ry = int(self._glow_ry[k] * pulse)
            pos = (gcx - rx, gcy - ry, gcx + rx, gcy + ry)
            if pos != p_glowo[k]:
                coords(glow_items_o[k], pos[0], pos[1], pos[2], pos[3])
                p_glowo[k] = pos
            # outer layer nearly invisible, inner layers build a soft falloff;
            # strength scales with level so quiet sections keep only a warm haze
            gl = glow_level
            strength = 0.008 + 0.11 * gl + 0.04 * renv + 0.042 * k * gl
            color = mix(bg_mid, glow_hot, strength)
            if color != last_o[k]:
                itemconfig(glow_items_o[k], fill=color)
                last_o[k] = color

        # beat pulse oval behind the bars (only while a beat is decaying)
        if renv < 0.03:
            if self._accent_vis:
                itemconfig(self._accent_item, state="hidden")
                self._accent_vis = False
        else:
            if not self._accent_vis:
                itemconfig(self._accent_item, state="normal")
                self._accent_vis = True
            arx = max(6, int(w * (0.12 + 0.08 * renv)))
            ary = max(6, int(h * (0.10 + 0.06 * renv)))
            acy = int(h * 0.60)
            apos = (w // 2 - arx, acy - ary, w // 2 + arx, acy + ary)
            if apos != self._p_accent:
                coords(self._accent_item, apos[0], apos[1], apos[2], apos[3])
                self._p_accent = apos
            acc_color = mix(bg_mid, glow_hot, 0.03 + 0.22 * renv)
            if acc_color != self._last_accent:
                itemconfig(self._accent_item, fill=acc_color)
                self._last_accent = acc_color

        # ------------------------------------------------- expanding rings
        rcx = w // 2
        rcy = int(h * 0.54)
        ring_items = self._ring_items
        ring_hot = self._ring_hot
        r0 = self._ring_r0
        span = self._ring_span
        p_ring = self._p_ring
        last_ring = self._last_ringc
        for k in range(_RING_COUNT):
            ph = (phase + 0.5 * k) % 1.0
            strength = renv * (1.0 - ph) ** 1.4
            if strength < 0.02:
                if self._ring_vis[k]:
                    itemconfig(ring_items[k], state="hidden")
                    self._ring_vis[k] = False
                continue
            if not self._ring_vis[k]:
                itemconfig(ring_items[k], state="normal")
                self._ring_vis[k] = True
            r = r0 + int(ph * span)
            pos = (rcx - r, rcy - r, rcx + r, rcy + r)
            if pos != p_ring[k]:
                coords(ring_items[k], pos[0], pos[1], pos[2], pos[3])
                p_ring[k] = pos
            color = mix(bg_mid, ring_hot, 0.05 + 0.80 * strength)
            if color != last_ring[k]:
                itemconfig(ring_items[k], outline=color)
                last_ring[k] = color

        # ------------------------------------------------- caption
        hint = hint or ""
        if hint != self._last_hint_text:
            if hint:
                itemconfig(self._hint_text, text=hint, state="normal")
                itemconfig(self._hint_shadow, text=hint, state="normal")
            else:
                itemconfig(self._hint_text, text="", state="hidden")
                itemconfig(self._hint_shadow, text="", state="hidden")
            self._last_hint_text = hint
        hint_color = scale_color(self._hint_base, 0.28 + 0.40 * lv)
        if hint_color != self._last_hint_color:
            itemconfig(self._hint_text, fill=hint_color)
            self._last_hint_color = hint_color

    # ------------------------------------------------------------ helpers
    def _resample(self, bands) -> list[float]:
        """Linearly resample arbitrary length bands to ``n_bars``, NaN-safe."""
        n = self.n_bars
        out = self._raw
        if bands is None:
            for i in range(n):
                out[i] = 0.0
            return out
        try:
            m = len(bands)
        except TypeError:
            for i in range(n):
                out[i] = 0.0
            return out
        if m == 0:
            for i in range(n):
                out[i] = 0.0
            return out
        try:
            arr = bands if isinstance(bands, np.ndarray) else np.asarray(bands, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if arr.size != m:
                m = arr.size
                if m == 0:
                    for i in range(n):
                        out[i] = 0.0
                    return out
            if m == n:
                for i in range(n):
                    v = float(arr[i])
                    if not (v > 0.0):
                        v = 0.0
                    elif v > 1.0:
                        v = 1.0
                    out[i] = v
                return out
            entry = self._resample_cache.get(m)
            if entry is None:
                entry = (np.arange(m, dtype=np.float64),
                         np.linspace(0.0, m - 1, n, dtype=np.float64))
                self._resample_cache[m] = entry
            sampled = np.interp(entry[1], entry[0], arr)
        except Exception:  # pragma: no cover - defensive, never crash the demo
            for i in range(n):
                out[i] = 0.0
            return out
        for i in range(n):
            v = float(sampled[i])
            if not (v > 0.0):
                v = 0.0
            elif v > 1.0:
                v = 1.0
            out[i] = v
        return out
