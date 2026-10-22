"""DJ mode — seven genuinely different full-screen canvases.

Each canvas has its own composition, play-cell layout, background system,
motion language and press reaction (not just a palette swap on one skeleton):

  0 Aurora Ink       ink sky, drifting aurora sheets, orbital arcs, stars
  1 Porcelain Silk   porcelain gradient, silk ribbons, glints, white caps
  2 Impressionist    lily pond: lily pads, water ripples, drifting paint dabs,
                     brush-stroke lines; jittered watercolour cells
  3 Carnival         night parade: two sweeping spotlights, falling confetti,
                     bokeh, twisted candy cells, neon marquee arcs
  4 Tech Metal       command deck: converging perspective grid, radar sweep &
                     rings, scanline, HUD ticks; offset notched metal cells
  5 Blood            ritual wall: tapered drips, stains, rising embers; jagged
                     shard cells
  6 Sacred           rose window: stained-glass pieslices, gold tracery, light
                     beams, dust, candles; cells on two radial rings as petals

Canvas changes use a wiping light curtain instead of a muddy colour cross-fade.

Keyboard: any key maps to a cell (see ``key_to_cell``); no legends are drawn.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ helpers


def _mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _scale(c, k):
    return (c[0] * k, c[1] * k, c[2] * k)


def _hex(c):
    return "#{:02X}{:02X}{:02X}".format(
        int(max(0, min(255, c[0]))), int(max(0, min(255, c[1]))),
        int(max(0, min(255, c[2]))))


def _ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rounded(r, per=5):
    pts = []
    corners = ((0.5 - r, -0.5 + r, -90.0, 0.0),
               (0.5 - r, 0.5 - r, 0.0, 90.0),
               (-0.5 + r, 0.5 - r, 90.0, 180.0),
               (-0.5 + r, -0.5 + r, 180.0, 270.0))
    for cx, cy, a0, a1 in corners:
        for k in range(per + 1):
            a = math.radians(a0 + (a1 - a0) * k / per)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return tuple(pts)


def _rotate(points, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return tuple((x * ca - y * sa, x * sa + y * ca) for x, y in points)


ROUND = _rounded(0.30)
SOFT = _rounded(0.44)
NOTCH = ((-0.50, -0.20), (-0.30, -0.50), (0.30, -0.50), (0.50, -0.20),
         (0.50, 0.20), (0.30, 0.50), (-0.30, 0.50), (-0.50, 0.20))
SHARD = ((-0.50, -0.12), (-0.16, -0.50), (0.22, -0.40), (0.50, -0.08),
         (0.34, 0.30), (0.10, 0.50), (-0.28, 0.42), (-0.50, 0.18))


def _petal(n=14, inner=0.62):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 0.5 * (inner + (1.0 - inner) * abs(math.cos(a)) ** 1.4)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return tuple(pts)


PETAL = _petal()

# ------------------------------------------------------------------ metadata

STYLES = [
    dict(name="Aurora Ink", light=False),
    dict(name="Porcelain Silk", light=True),
    dict(name="Impressionist", light=True),
    dict(name="Carnival", light=False),
    dict(name="Tech Metal", light=False),
    dict(name="Blood", light=False),
    dict(name="Sacred", light=False),
]
N_STYLES = len(STYLES)

# full keyboard -> cell (r, c); any key works
_KEY_ROWS = ("`1234567890-=", "qwertyuiop", "asdfghjkl", "zxcvbnm")
_SPECIAL = {
    "space": (2, 1), "return": (3, 2), "kp_enter": (3, 2), "tab": (0, 1),
    "backspace": (0, 3), "caps_lock": (0, 0),
    "up": (0, 2), "down": (3, 1), "left": (2, 0), "right": (2, 3),
    "home": (0, 0), "end": (3, 3), "prior": (0, 2), "next": (3, 2),
    "insert": (0, 1), "delete": (0, 2),
}


def key_to_cell(keysym, char=""):
    ks = (keysym or "").lower()
    ch = (char or "").lower()
    if not ch and len(ks) == 1:
        # synthetic Tk events (and some IMEs) only provide keysym
        ch = ks
    if ch == " ":
        return (2, 1)
    for row_idx, row in enumerate(_KEY_ROWS):
        if ch and ch in row:
            col = int(round(row.index(ch) / max(len(row) - 1, 1) * 3))
            return (row_idx, col)
    if ks in _SPECIAL:
        return _SPECIAL[ks]
    if not ch and not ks:
        return None
    h = sum(ord(c) for c in (ks or ch))
    return (h % 4, (h // 4) % 4)


@dataclass
class Cell:
    r: int
    c: int
    x: float
    y: float
    size: float
    angle: float = 0.0
    unit: tuple = ROUND
    act_t: float = -1e9
    amp: float = 1.0
    tint: int = 0


@dataclass
class Particle:
    item: int
    x: float
    y: float
    vx: float
    vy: float
    gravity: float
    age: float
    life: float
    size: float
    color: tuple
    drift: float = 0.0


# ------------------------------------------------------------------ styles

class BaseStyle:
    name = ""
    light = False
    bg = ((6, 7, 12), (22, 13, 30))
    base = (30, 25, 42)
    text = (240, 236, 248)
    palettes = ()
    chip = dict(shape=ROUND, glow=0.45, border=0.0, idle=0.08,
                glow_off=(0.0, 0.0), border_w=1)

    def __init__(self, scene):
        self.s = scene
        self.items = []
        self.build()
        for it in self.items:
            self.s.c.itemconfig(it, state="hidden")

    def item(self, kind, *args, **kw):
        item = getattr(self.s.c, f"create_{kind}")(*args, **kw)
        self.items.append(item)
        return item

    def build(self):
        pass

    def set_visible(self, on):
        st = "normal" if on else "hidden"
        for it in self.items:
            self.s.c.itemconfig(it, state=st)
        # the scene tracks its own shown items; a direct group toggle resets it
        self.s._visible.difference_update(self.items)

    def cells(self):
        s = self.s
        gap = min((s.w - 2 * s.margin_x) / 4, (s.h * 0.56) / 4)
        base = gap * 0.42
        gx0 = (s.w - gap * 4) * 0.5
        gy0 = s.h * 0.20 + max(0.0, (s.h * 0.56 - gap * 4) * 0.5)
        out = []
        for r in range(4):
            for c in range(4):
                out.append(Cell(r, c, gx0 + (c + 0.5) * gap,
                                gy0 + (r + 0.5) * gap, base,
                                unit=self.chip["shape"]))
        return out

    def draw(self, t, k, pal, level, accent):
        pass

    def on_beat(self, t, pal):
        pass

    def burst(self, x, y, pal, n=10):
        self.s.burst(x, y, pal, n, kind="spark")


def _grid_cells(style, jitter=0.0, size_jitter=0.0, angle_jit=0.0,
                seed=1, size_mul=1.0):
    s = style.s
    gap = min((s.w - 2 * s.margin_x) / 4, (s.h * 0.56) / 4)
    base = gap * 0.42 * size_mul
    gx0 = (s.w - gap * 4) * 0.5
    gy0 = s.h * 0.20 + max(0.0, (s.h * 0.56 - gap * 4) * 0.5)
    rng = random.Random(seed)
    out = []
    for r in range(4):
        for c in range(4):
            jx = rng.uniform(-jitter, jitter) * gap if jitter else 0.0
            jy = rng.uniform(-jitter, jitter) * gap if jitter else 0.0
            sz = base * (1.0 + rng.uniform(-size_jitter, size_jitter))
            ang = rng.uniform(-angle_jit, angle_jit) if angle_jit else 0.0
            out.append(Cell(r, c, gx0 + (c + 0.5) * gap + jx,
                            gy0 + (r + 0.5) * gap + jy, sz, ang,
                            unit=_rotate(style.chip["shape"], ang)))
    return out


class AuroraStyle(BaseStyle):
    name = "Aurora Ink"
    bg = ((5, 6, 11), (21, 12, 29))
    base = (30, 25, 42)
    palettes = (
        ("Sunset", ((255, 178, 96), (255, 98, 142), (188, 132, 255))),
        ("Ocean", ((90, 242, 216), (78, 142, 255), (164, 128, 255))),
        ("Ember", ((255, 198, 82), (255, 102, 58), (255, 68, 100))),
    )
    chip = dict(shape=ROUND, glow=0.48, border=0.0, idle=0.09,
                glow_off=(0.0, 0.0), border_w=1)

    def build(self):
        self.clouds = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(5)]
        self.stars = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                      for _ in range(16)]
        self.arcs = [self.item("arc", 0, 0, 1, 1, start=0, extent=140,
                               style="arc", outline="#000", width=1)
                     for _ in range(2)]

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        for i, it in enumerate(self.clouds):
            rad = s.h * (0.16 + 0.055 * (i % 3)) * (1.0 + 0.05 * accent)
            cx = s.w * (0.12 + 0.19 * i) + math.sin(t * 0.08 + i) * s.w * 0.03
            cy = s.h * (0.26 + 0.11 * (i % 4)) + math.cos(t * 0.06 + i) * s.h * 0.03
            s.coords(it, (cx - rad, cy - rad * 0.82, cx + rad, cy + rad * 0.82), 2.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], bg, 0.87), k),
                       ("c", i, round(cx, 1), round(cy, 1), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.stars):
            cx = s.w * (0.06 + 0.88 * ((i * 37) % 100) / 100.0)
            cy = s.h * (0.08 + 0.7 * ((i * 53) % 100) / 100.0)
            tw = 0.5 + 0.5 * math.sin(t * (0.8 + (i % 5) * 0.2) + i)
            sz = s.h * (0.0016 + 0.0022 * tw)
            s.coords(it, (cx - sz, cy - sz, cx + sz, cy + sz), 1.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], (255, 255, 255), 0.5),
                                  k * (0.25 + 0.6 * tw)),
                       ("star", i, round(tw, 1), round(k, 2)))
            s.show(it)
        for j, it in enumerate(self.arcs):
            r = min(s.w, s.h) * (0.40 + 0.05 * j)
            cx, cy = s.w * 0.5, s.h * 0.47
            s.coords(it, (cx - r, cy - r, cx + r, cy + r), 2.0)
            s.c.itemconfig(it, start=(t * (6 + 3 * j)) % 360, extent=150 - j * 40)
            s.set_outline(it, _scale(_mix(pal[j], bg, 0.62), k * 0.9),
                          ("arc", j, int(t * 0.5), round(k, 2)), 1)
            s.show(it)

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.47, 0.22)


class PorcelainStyle(BaseStyle):
    name = "Porcelain Silk"
    light = True
    bg = ((252, 249, 243), (240, 236, 248))
    base = (255, 253, 249)
    text = (66, 60, 82)
    palettes = (
        ("Blossom", ((255, 150, 110), (246, 170, 205), (176, 156, 255))),
        ("Mist", ((128, 212, 198), (158, 186, 255), (206, 224, 255))),
        ("Dune", ((214, 158, 96), (232, 198, 160), (178, 152, 132))),
    )
    chip = dict(shape=ROUND, glow=0.30, border=0.18, idle=0.05,
                glow_off=(0.0, 7.0), border_w=1)

    def build(self):
        self.ribbons = [self.item("line", 0, 0, 1, 1, width=30, fill="#000",
                                  capstyle="round") for _ in range(5)]
        self.washes = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(3)]
        self.glints = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(8)]

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        for i, it in enumerate(self.washes):
            rad = s.h * (0.28 + 0.03 * i)
            cx = s.w * (0.18 + 0.32 * i) + math.cos(t * 0.03 + i) * s.w * 0.01
            cy = s.h * (0.25 + 0.24 * i)
            s.coords(it, (cx - rad, cy - rad * 0.8, cx + rad, cy + rad * 0.8), 2.0)
            s.set_fill(it, _scale(_mix(pal[i], bg, 0.94), k),
                       ("w", i, round(k, 2), round(cx, 0)))
            s.show(it)
        for i, it in enumerate(self.ribbons):
            ang = math.radians(-26 + i * 9 + math.sin(t * 0.05 + i) * 2.0)
            cx = s.w * 0.5 + math.cos(t * 0.04 + i * 1.3) * s.w * 0.03
            cy = s.h * (0.18 + 0.16 * i)
            length = s.w * 1.5
            dx, dy = math.cos(ang) * length * 0.5, math.sin(ang) * length * 0.5
            s.coords(it, (cx - dx, cy - dy, cx + dx, cy + dy), 2.0)
            s.c.itemconfig(it, width=max(12, int(s.h * 0.030)))
            s.set_fill(it, _scale(_mix(pal[i % 3], (255, 255, 255), 0.55), k * 0.9),
                       ("r", i, round(cx, 0), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.glints):
            cx = s.w * (0.08 + 0.11 * i) + math.sin(t * 0.2 + i) * 8
            cy = s.h * (0.24 + 0.5 * ((i * 7) % 13) / 13.0)
            tw = 0.5 + 0.5 * math.sin(t * 1.1 + i)
            sz = s.h * (0.0015 + 0.002 * tw)
            s.coords(it, (cx - sz, cy - sz, cx + sz, cy + sz), 1.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], (255, 255, 255), 0.8),
                                  k * (0.3 + 0.4 * tw)),
                       ("g", i, round(tw, 1)))
            s.show(it)

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.5, 0.18)


class ImpressionistStyle(BaseStyle):
    name = "Impressionist"
    light = True
    bg = ((246, 242, 229), (233, 228, 246))
    base = (255, 252, 244)
    text = (70, 62, 78)
    palettes = (
        ("Water Lilies", ((150, 132, 205), (226, 148, 176), (134, 184, 158))),
        ("Sunrise", ((250, 158, 108), (252, 202, 150), (178, 154, 218))),
        ("Wheat Field", ((190, 172, 104), (160, 190, 144), (144, 172, 214))),
    )
    chip = dict(shape=SOFT, glow=0.55, border=0.12, idle=0.22,
                glow_off=(0.0, 0.0), border_w=1)

    def build(self):
        self.pads = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                     for _ in range(6)]
        self.dabs = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                     for _ in range(20)]
        self.strokes = [self.item("line", 0, 0, 1, 1, width=10, fill="#000",
                                  capstyle="round") for _ in range(6)]
        self.ripples = [self.item("arc", 0, 0, 1, 1, start=0, extent=200,
                                  style="arc", outline="#000", width=1)
                        for _ in range(4)]

    def cells(self):
        return _grid_cells(self, jitter=0.10, size_jitter=0.10,
                           angle_jit=0.10, seed=22, size_mul=1.02)

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        for i, it in enumerate(self.pads):
            rad = s.h * (0.045 + 0.022 * ((i * 5) % 7) / 7.0)
            cx = s.w * (0.10 + 0.155 * i) + math.sin(t * 0.07 + i) * s.w * 0.02
            cy = s.h * (0.16 + 0.11 * (i % 5)) + math.cos(t * 0.05 + i) * s.h * 0.02
            s.coords(it, (cx - rad * 1.5, cy - rad, cx + rad * 1.5, cy + rad), 2.0)
            col = _mix(pal[(i + 1) % 3], (255, 255, 255), 0.55)
            s.set_fill(it, _scale(_mix(col, bg, 0.45), k),
                       ("pad", i, round(cx, 0), round(cy, 0)))
            s.show(it)
        for i, it in enumerate(self.dabs):
            cx = s.w * (0.04 + 0.92 * ((i * 37) % 101) / 101.0) \
                + math.sin(t * (0.09 + 0.01 * (i % 5)) + i * 1.7) * s.w * 0.012
            cy = s.h * (0.10 + 0.80 * ((i * 59) % 97) / 97.0) \
                + math.cos(t * (0.06 + 0.01 * (i % 3)) + i) * s.h * 0.012
            sz = s.h * (0.006 + 0.010 * ((i * 11) % 9) / 9.0)
            s.coords(it, (cx - sz * 1.3, cy - sz, cx + sz * 1.3, cy + sz), 1.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], (255, 255, 255), 0.42),
                                  k * (0.7 + 0.3 * math.sin(t * 0.4 + i))),
                       ("dab", i, round(cx, 0), round(cy, 0), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.strokes):
            cx = s.w * (0.08 + 0.16 * i) + math.sin(t * 0.05 + i) * s.w * 0.01
            cy = s.h * (0.74 + 0.015 * (i % 3)) + math.cos(t * 0.07 + i * 2) * s.h * 0.01
            dx = s.w * 0.035
            s.coords(it, (cx - dx, cy - dx * 0.18, cx, cy + dx * 0.10,
                          cx + dx, cy - dx * 0.05), 2.0)
            s.c.itemconfig(it, width=max(7, int(s.h * 0.011)))
            s.set_fill(it, _scale(_mix(pal[(i + 2) % 3], bg, 0.55), k * 0.9),
                       ("stroke", i, round(cx, 0), round(cy, 0)))
            s.show(it)
        for j, it in enumerate(self.ripples):
            cyc = (t * 0.12 + j * 0.25) % 1.0
            rad = s.h * (0.02 + 0.11 * cyc)
            cx = s.w * (0.25 + 0.16 * j)
            cy = s.h * (0.34 + 0.14 * ((j * 3) % 4))
            s.coords(it, (cx - rad, cy - rad * 0.55, cx + rad, cy + rad * 0.55), 2.0)
            alpha = (1.0 - cyc) ** 1.3
            s.set_outline(it, _scale(_mix(pal[j % 3], bg, 1.0 - 0.45 * alpha), k),
                          ("rip", j, round(cyc, 2)), 1)
            s.show(it)

    def burst(self, x, y, pal, n=10):
        self.s.burst(x, y, pal, n, kind="dab")

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * (0.3 + 0.4 * random.random()),
                      self.s.h * (0.35 + 0.3 * random.random()), 0.24)


class CarnivalStyle(BaseStyle):
    name = "Carnival"
    bg = ((13, 6, 22), (34, 8, 34))
    base = (38, 22, 48)
    palettes = (
        ("Neon", ((255, 64, 180), (64, 240, 255), (255, 232, 64))),
        ("Candy", ((255, 120, 190), (140, 255, 120), (80, 220, 255))),
        ("Rave", ((180, 80, 255), (180, 255, 40), (255, 140, 40))),
    )
    chip = dict(shape=ROUND, glow=0.65, border=0.55, idle=0.30,
                glow_off=(0.0, 0.0), border_w=2)

    def build(self):
        self.beams = [self.item("line", 0, 0, 1, 1, width=80, fill="#000",
                                capstyle="round") for _ in range(2)]
        self.strips = [self.item("line", 0, 0, 1, 1, width=8, fill="#000",
                                 capstyle="round") for _ in range(10)]
        self.confetti = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                         for _ in range(20)]
        self.bokeh = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                      for _ in range(6)]
        self.arches = [self.item("arc", 0, 0, 1, 1, start=0, extent=120,
                                 style="arc", outline="#000", width=2)
                       for _ in range(2)]

    def cells(self):
        return _grid_cells(self, jitter=0.20, size_jitter=0.16,
                           angle_jit=0.28, seed=33, size_mul=1.04)

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        bx, by = s.w * 0.5, s.h * 1.04
        for i, it in enumerate(self.beams):
            ang = math.radians(-108 + math.sin(t * (0.16 + 0.05 * i) + i * 2.1) * 34)
            length = s.h * 1.5
            x1, y1 = bx + math.cos(ang) * length, by + math.sin(ang) * length
            s.coords(it, (bx, by, x1, y1), 2.0)
            s.c.itemconfig(it, width=max(30, int(s.h * 0.075)))
            s.set_fill(it, _scale(_mix(pal[i % 3], bg, 0.86 - 0.05 * accent), k),
                       ("beam", i, round(ang, 2), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.strips):
            cy = (t * (70 + i * 9) + i * 61) % (s.h * 1.05) - s.h * 0.03
            cx = s.w * (0.06 + 0.088 * i) + math.sin(t * 0.7 + i) * 14
            ang = t * 3.0 + i
            dx, dy = math.cos(ang) * s.h * 0.016, math.sin(ang) * s.h * 0.016
            s.coords(it, (cx - dx, cy - dy, cx + dx, cy + dy), 2.0)
            s.c.itemconfig(it, width=max(3, int(s.h * 0.006)))
            s.set_fill(it, _scale(pal[(i + 1) % 3], k * 0.95),
                       ("strip", i, round(cy / 24), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.confetti):
            cy = (t * (46 + i * 5) + i * 43) % (s.h * 1.1) - s.h * 0.05
            cx = s.w * (0.03 + 0.93 * ((i * 29) % 101) / 101.0) \
                + math.sin(t * 0.9 + i) * 22
            sz = s.h * (0.004 + 0.004 * ((i * 3) % 5) / 5.0)
            s.coords(it, (cx - sz, cy - sz, cx + sz, cy + sz), 1.0)
            s.set_fill(it, _scale(pal[i % 3], k * (0.8 + 0.2 * math.sin(t + i))),
                       ("conf", i, round(cy / 24), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.bokeh):
            cx = s.w * (0.10 + 0.16 * i)
            cy = s.h * (0.20 + 0.12 * ((i * 3) % 4)) + math.sin(t * 0.3 + i) * 10
            rad = s.h * (0.02 + 0.015 * math.sin(t * 0.5 + i * 2) ** 2)
            s.coords(it, (cx - rad, cy - rad, cx + rad, cy + rad), 2.0)
            s.set_fill(it, _scale(_mix(pal[(i + 1) % 3], bg, 0.80), k),
                       ("bok", i, round(rad, 1)))
            s.show(it)
        for j, it in enumerate(self.arches):
            r = min(s.w, s.h) * (0.30 + 0.08 * j)
            cx, cy = s.w * 0.5, s.h * 0.44
            s.coords(it, (cx - r, cy - r, cx + r, cy + r), 2.0)
            s.c.itemconfig(it, start=(t * (18 + 9 * j)) % 360, extent=110)
            s.set_outline(it, _scale(_mix(pal[j], bg, 0.45), k * (0.7 + 0.3 * accent)),
                          ("arch", j, int(t * 2)), 2)
            s.show(it)

    def burst(self, x, y, pal, n=12):
        self.s.burst(x, y, pal, n, kind="confetti")

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.45, 0.30)


class TechStyle(BaseStyle):
    name = "Tech Metal"
    bg = ((12, 15, 19), (27, 32, 39))
    base = (42, 47, 56)
    palettes = (
        ("Titanium", ((140, 196, 224), (198, 214, 230), (84, 168, 210))),
        ("Copper", ((214, 146, 96), (196, 186, 166), (104, 196, 196))),
        ("Graphite", ((146, 156, 176), (106, 126, 156), (74, 196, 218))),
    )
    chip = dict(shape=NOTCH, glow=0.32, border=0.75, idle=0.20,
                glow_off=(0.0, 0.0), border_w=2)

    def build(self):
        self.grid = [self.item("line", 0, 0, 1, 1, width=1, fill="#000")
                     for _ in range(8)]
        self.rings = [self.item("arc", 0, 0, 1, 1, start=0, extent=120,
                                style="arc", outline="#000", width=2)
                      for _ in range(3)]
        self.sweep = self.item("line", 0, 0, 1, 1, width=2, fill="#000",
                               capstyle="round")
        self.scan = self.item("line", 0, 0, 1, 1, width=3, fill="#000",
                              capstyle="round")
        self.ticks = [self.item("line", 0, 0, 1, 1, width=6, fill="#000",
                                capstyle="round") for _ in range(12)]

    def cells(self):
        s = self.s
        gap = min((s.w - 2 * s.margin_x) / 4, (s.h * 0.56) / 4)
        base = gap * 0.40
        gx0 = (s.w - gap * 4) * 0.5
        gy0 = s.h * 0.20 + max(0.0, (s.h * 0.56 - gap * 4) * 0.5)
        out = []
        for r in range(4):
            for c in range(4):
                off = gap * (0.16 if r % 2 else -0.16)
                out.append(Cell(r, c, gx0 + (c + 0.5) * gap + off,
                                gy0 + (r + 0.5) * gap, base, unit=NOTCH))
        return out

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        vx = s.w * 0.5
        for i in range(8):
            if i < 4:
                bx = s.w * (0.10 + 0.26 * i)
                tx = vx + (bx - vx) * 0.18
                s.coords(self.grid[i], (bx, s.h, tx, s.h * 0.10), 2.0)
            else:
                f = (i - 3) / 5.0
                y = s.h * (0.30 + 0.16 * f) ** 1.6 + s.h * 0.10
                s.coords(self.grid[i], (s.w * 0.04, y, s.w * 0.96, y), 2.0)
            s.c.itemconfig(self.grid[i], width=1)
            s.set_fill(self.grid[i],
                       _scale(_mix(pal[i % 3], bg, 0.78 - 0.05 * accent), k),
                       ("grid", i, round(k, 2)))
            s.show(self.grid[i])
        for j, it in enumerate(self.rings):
            r = min(s.w, s.h) * (0.24 + 0.09 * j)
            cx, cy = s.w * 0.5, s.h * 0.45
            s.coords(it, (cx - r, cy - r, cx + r, cy + r), 2.0)
            s.c.itemconfig(it, start=(t * (26 + 12 * j) * (1 if j else -1)) % 360,
                           extent=100 + 30 * j)
            s.set_outline(it, _scale(_mix(pal[j], bg, 0.42), k * 1.1),
                          ("ring", j, int(t * 3)), 1)
            s.show(it)
        cx, cy = s.w * 0.5, s.h * 0.45
        rad = min(s.w, s.h) * 0.42
        ang = t * 1.4
        s.coords(self.sweep, (cx, cy, cx + math.cos(ang) * rad,
                              cy + math.sin(ang) * rad), 2.0)
        s.c.itemconfig(self.sweep, width=2)
        s.set_fill(self.sweep, _scale(_mix(pal[0], (255, 255, 255), 0.35),
                                      k * 0.9), ("sweep", int(t * 8)))
        s.show(self.sweep)
        y = s.h * (0.06 + ((t * 0.09) % 1.0) * 0.86)
        s.coords(self.scan, (s.w * 0.03, y, s.w * 0.97, y), 2.0)
        s.c.itemconfig(self.scan, width=max(2, int(s.h * 0.003)))
        s.set_fill(self.scan, _scale(pal[0], k * 0.5), ("scan", int(y)))
        s.show(self.scan)
        for i, it in enumerate(self.ticks):
            x = s.w * (0.05 + 0.08 * i)
            yy = s.h * (0.90 + math.sin(t * 0.7 + i) * 0.012)
            s.coords(it, (x, yy, x, yy - s.h * (0.01 + 0.02 * ((i * 5) % 7) / 7.0)), 2.0)
            s.c.itemconfig(it, width=max(2, int(s.h * 0.0025)))
            s.set_fill(it, _scale(pal[i % 3], k * (0.4 + 0.3 * accent)),
                       ("tick", i, round(yy, 1)))
            s.show(it)

    def burst(self, x, y, pal, n=10):
        self.s.burst(x, y, pal, n, kind="spark")

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.45, 0.20)


class BloodStyle(BaseStyle):
    name = "Blood"
    bg = ((17, 5, 7), (43, 7, 11))
    base = (48, 16, 18)
    palettes = (
        ("Crimson", ((206, 26, 38), (134, 18, 28), (255, 88, 76))),
        ("Rust", ((176, 78, 38), (116, 48, 28), (216, 126, 66))),
        ("Wine", ((146, 18, 68), (96, 18, 48), (216, 86, 136))),
    )
    chip = dict(shape=SHARD, glow=0.58, border=0.35, idle=0.21,
                glow_off=(0.0, 0.0), border_w=1)

    def build(self):
        self.stains = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(4)]
        self.drips = [self.item("polygon", 0, 0, 1, 0, 1, 1, 0, 1,
                                fill="#000", outline="") for _ in range(9)]
        self.splats = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(12)]
        self.embers = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                       for _ in range(10)]
        self.arcs = [self.item("arc", 0, 0, 1, 1, start=0, extent=140,
                               style="arc", outline="#000", width=1)
                     for _ in range(2)]

    def cells(self):
        return _grid_cells(self, jitter=0.16, size_jitter=0.18,
                           angle_jit=0.20, seed=55, size_mul=1.0)

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        for i, it in enumerate(self.stains):
            rad = s.h * (0.10 + 0.03 * i)
            cx = s.w * (0.16 + 0.23 * i) + math.sin(t * 0.02 + i) * 10
            cy = s.h * (0.24 + 0.16 * (i % 3)) + math.cos(t * 0.02 + i) * 8
            s.coords(it, (cx - rad, cy - rad * 0.85, cx + rad, cy + rad * 0.85), 2.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], bg, 0.90), k),
                       ("stain", i, round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.drips):
            x = s.w * (0.04 + 0.105 * i) + math.sin(t * 0.02 + i) * 4
            top = s.h * (0.05 + 0.03 * ((i * 3) % 5) / 5.0)
            length = s.h * (0.16 + 0.16 * ((i * 7) % 9) / 9.0)
            length *= 0.82 + 0.18 * math.sin(t * 0.23 + i * 1.3)
            wdt = s.h * (0.006 + 0.004 * ((i * 5) % 5) / 5.0)
            pts = (x - wdt, top, x + wdt, top, x + wdt * 0.35, top + length,
                   x - wdt * 0.35, top + length)
            s.coords(it, pts, 2.0)
            s.set_fill(it, _scale(_mix(pal[i % 3], bg, 0.28), k * 0.92),
                       ("drip", i, round(length, 0), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.splats):
            cx = s.w * (0.05 + 0.9 * ((i * 41) % 97) / 97.0)
            cy = s.h * (0.10 + 0.85 * ((i * 67) % 89) / 89.0)
            sz = s.h * (0.003 + 0.006 * ((i * 7) % 5) / 5.0)
            s.coords(it, (cx - sz, cy - sz, cx + sz, cy + sz), 1.0)
            s.set_fill(it, _scale(_mix(pal[(i + 1) % 3], bg, 0.45), k * 0.8),
                       ("splat", i, round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.embers):
            cy = s.h * (0.95 - ((t * (18 + i * 2) + i * 37) % (s.h * 0.85)) / s.h)
            cx = s.w * (0.05 + 0.08 * i) + math.sin(t * 0.6 + i) * 12
            sz = s.h * (0.0018 + 0.0018 * math.sin(t + i) ** 2)
            s.coords(it, (cx - sz, cy - sz, cx + sz, cy + sz), 1.0)
            s.set_fill(it, _scale(_mix(pal[0], (255, 180, 120), 0.3),
                                  k * (0.5 + 0.4 * math.sin(t * 2 + i))),
                       ("ember", i, round(cy / 10)))
            s.show(it)
        for j, it in enumerate(self.arcs):
            r = min(s.w, s.h) * (0.26 + 0.05 * j)
            cx, cy = s.w * 0.5, s.h * 0.42
            s.coords(it, (cx - r, cy - r, cx + r, cy + r), 2.0)
            s.set_outline(it, _scale(_mix(pal[j], bg, 0.55),
                                     k * (0.5 + 0.3 * accent)),
                          ("barc", j, round(accent, 1)), 1)
            s.show(it)

    def burst(self, x, y, pal, n=10):
        self.s.burst(x, y, pal, n, kind="drip")

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.42, 0.22)


class SacredStyle(BaseStyle):
    name = "Sacred"
    bg = ((9, 13, 32), (30, 21, 42))
    base = (44, 44, 74)
    text = (246, 230, 190)
    palettes = (
        ("Byzantine", ((226, 184, 92), (150, 182, 226), (255, 226, 150))),
        ("Rose Window", ((230, 148, 168), (226, 192, 118), (136, 196, 196))),
        ("Ivory", ((242, 230, 204), (214, 194, 146), (164, 184, 226))),
    )
    chip = dict(shape=PETAL, glow=0.42, border=0.85, idle=0.36,
                glow_off=(0.0, 0.0), border_w=2)

    def build(self):
        self.glass = []
        self.glass_specs = []
        ring_specs = ((0.34, 12, 30.0), (0.24, 8, 45.0), (0.14, 4, 90.0))
        for ri, (rr, count, extent) in enumerate(ring_specs):
            for i in range(count):
                it = self.item("arc", 0, 0, 1, 1, start=0, extent=extent - 4,
                               style="pieslice", fill="#000", outline="#000",
                               width=1)
                self.glass.append(it)
                self.glass_specs.append((ri, i, count, rr))
        self.tracery = [self.item("line", 0, 0, 1, 1, width=2, fill="#000")
                        for _ in range(8)]
        self.beams = [self.item("polygon", 0, 0, 1, 0, 1, 1, 0, 1,
                                fill="#000", outline="") for _ in range(5)]
        self.dust = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                     for _ in range(14)]
        self.candles = [self.item("oval", 0, 0, 2, 2, fill="#000", outline="")
                        for _ in range(3)]

    def cells(self):
        s = self.s
        cx, cy = s.w * 0.5, s.h * 0.475
        size = min(s.w, s.h) * 0.058
        out = []
        inner_r = min(s.w, s.h) * 0.170
        outer_r = min(s.w, s.h) * 0.292
        for i in range(8):
            ang = -math.pi / 2 + 2 * math.pi * i / 8
            out.append(Cell(i // 4, i % 4,
                            cx + math.cos(ang) * inner_r,
                            cy + math.sin(ang) * inner_r,
                            size * 0.92,
                            angle=ang + math.pi / 2,
                            unit=_rotate(PETAL, ang + math.pi / 2)))
        for i in range(8):
            ang = -math.pi / 2 + 2 * math.pi * (i + 0.5) / 8
            out.append(Cell(2 + i // 4, i % 4,
                            cx + math.cos(ang) * outer_r,
                            cy + math.sin(ang) * outer_r,
                            size,
                            angle=ang + math.pi / 2,
                            unit=_rotate(PETAL, ang + math.pi / 2)))
        return out

    def draw(self, t, k, pal, level, accent):
        s = self.s
        bg = _scale(self.bg[0], k)
        cx, cy = s.w * 0.5, s.h * 0.475
        m = min(s.w, s.h)
        rot = t * 1.5
        for it, (ri, i, count, rr) in zip(self.glass, self.glass_specs):
            r = m * rr
            s.coords(it, (cx - r, cy - r, cx + r, cy + r), 2.0)
            extent = 360.0 / count - 5
            start_off = (rot * (1 if ri % 2 == 0 else -1)) % 360
            s.c.itemconfig(it, start=(i * 360.0 / count + start_off),
                           extent=extent)
            col = _mix(pal[(i + ri) % 3], bg, 0.24)
            s.set_fill(it, _scale(col, k * (0.92 + 0.25 * accent)),
                       ("glass", ri, i, round(k, 2), round(start_off, 0)))
            s.c.itemconfig(it, outline=_hex(
                _scale(_mix(pal[0], (255, 246, 214), 0.5), k * 0.95)))
            s.show(it)
        for i, it in enumerate(self.tracery):
            ang = math.radians(i * 45 + rot * 0.5)
            rr = m * 0.42
            s.coords(it, (cx + math.cos(ang) * m * 0.10,
                          cy + math.sin(ang) * m * 0.10,
                          cx + math.cos(ang) * rr,
                          cy + math.sin(ang) * rr), 2.0)
            s.c.itemconfig(it, width=2)
            s.set_fill(it, _scale(_mix(pal[0], (255, 246, 210), 0.35),
                                  k * 0.85), ("trace", i, round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.beams):
            ang = math.radians(-90 + (i - 2) * 20 + math.sin(t * 0.12 + i) * 3)
            length = m * 0.8
            width = 0.055
            x0, y0 = s.w * 0.5, s.h * 0.10
            pts = (x0, y0,
                   x0 + math.cos(ang) * length, y0 + math.sin(ang) * length,
                   x0 + math.cos(ang + width) * length,
                   y0 + math.sin(ang + width) * length,
                   x0 + math.cos(ang - width) * length * 0.98,
                   y0 + math.sin(ang - width) * length * 0.98)
            s.coords(it, pts, 2.0)
            s.set_fill(it, _scale(_mix(pal[0], bg, 0.64), k * (0.60 + 0.40 * accent)),
                       ("beam", i, round(ang, 2), round(k, 2)))
            s.show(it)
        for i, it in enumerate(self.dust):
            cyc = (t * (0.03 + 0.006 * i) + i * 0.37) % 1.0
            cx2 = s.w * (0.08 + 0.84 * ((i * 31) % 89) / 89.0)
            cy2 = s.h * (0.92 - 0.72 * cyc)
            tw = 0.5 + 0.5 * math.sin(t * (0.9 + 0.1 * i) + i * 2)
            sz = s.h * (0.0018 + 0.002 * tw)
            s.coords(it, (cx2 - sz, cy2 - sz, cx2 + sz, cy2 + sz), 1.0)
            s.set_fill(it, _scale(pal[i % 3], k * (0.3 + 0.6 * tw)),
                       ("dust", i, round(cyc, 2)))
            s.show(it)
        for i, it in enumerate(self.candles):
            cx2 = s.w * (0.18 + 0.32 * i)
            cy2 = s.h * 0.90
            rad = s.h * (0.022 + 0.006 * math.sin(t * 1.3 + i * 2) ** 2)
            s.coords(it, (cx2 - rad, cy2 - rad, cx2 + rad, cy2 + rad), 2.0)
            s.set_fill(it, _scale(_mix(pal[0], (255, 240, 200), 0.5),
                                  k * (0.35 + 0.25 * accent)),
                       ("candle", i, round(rad, 1)))
            s.show(it)

    def burst(self, x, y, pal, n=10):
        self.s.burst(x, y, pal, n, kind="holy")

    def on_beat(self, t, pal):
        self.s.ripple(self.s.w * 0.5, self.s.h * 0.45, 0.22)


STYLE_CLASSES = [AuroraStyle, PorcelainStyle, ImpressionistStyle,
                 CarnivalStyle, TechStyle, BloodStyle, SacredStyle]


# ------------------------------------------------------------------ scene

class DJScene:
    """A full-screen DJ instrument that cycles between seven canvases."""

    def __init__(self, canvas, width, height, n=4):
        self.c = canvas
        self.w = max(640, int(width))
        self.h = max(360, int(height))
        self.n = n
        self.margin_x = self.w * 0.13
        self.time = 0.0
        self._last_t = time.perf_counter()
        self.level = 0.0
        self.accent = 0.0
        self.beat_phase = 0.0
        self._prev_phase = 0.0
        self.bands = np.zeros(24, dtype=np.float32)
        self.visible = False
        self._fill_cache = {}
        self._coord_cache = {}
        self._text_cache = {}
        self._visible: set[int] = set()
        self._chip_last = 0.0
        self.hint_until = 8.0

        self.style_index = 0
        self.variant_pos = 0.0
        self.variant_target = 0.0
        self.particles: list[Particle] = []
        self.rings: list[tuple] = []
        self.pointer = None
        self.pointer_trail = []
        self._wipe_t = -1e9
        self._wipe_dir = 1.0
        self._wipe_committed = True

        # background gradient bands (shared)
        self.n_bands = 32
        self.bg_items = [self.c.create_rectangle(
            0, i * self.h / self.n_bands, self.w,
            i * self.h / self.n_bands + self.h / self.n_bands + 1.2,
            fill="#000000", outline="") for i in range(self.n_bands)]
        self.particle_items = [self.c.create_oval(0, 0, 2, 2, fill="#000",
                                                  outline="")
                               for _ in range(30)]
        self.ring_items = [self.c.create_oval(0, 0, 1, 1, outline="", width=2)
                           for _ in range(14)]
        for it in self.particle_items + self.ring_items:
            self.c.itemconfig(it, state="hidden")

        # style groups (their items sit above the bands)
        self.styles = [cls(self) for cls in STYLE_CLASSES]
        self.cells = self.styles[0].cells()
        self.styles[0].set_visible(False)

        # play cells: glow + body polygons
        self.chips = []
        for i in range(16):
            glow = self.c.create_polygon(0, 0, 1, 0, 1, 1, 0, 1,
                                         fill="#000", outline="")
            body = self.c.create_polygon(0, 0, 1, 0, 1, 1, 0, 1,
                                         fill="#000", outline="")
            self.chips.append([glow, body])
            for it in (glow, body):
                self.c.itemconfig(it, state="hidden")

        self.pointer_items = [self.c.create_oval(0, 0, 1, 1, outline="", width=2),
                              self.c.create_line(0, 0, 1, 1, width=3,
                                                 capstyle="round", fill="#000")]
        for it in self.pointer_items:
            self.c.itemconfig(it, state="hidden")

        self.title = self.c.create_text(
            self.w * 0.5, self.h * 0.085, text="OPENRAY · DJ",
            font=("Segoe UI", max(11, int(self.h * 0.019)), "bold"),
            fill="#FFFFFF")
        self.subtitle = self.c.create_text(
            self.w * 0.5, self.h * 0.085 + max(17, int(self.h * 0.030)),
            text="", font=("Segoe UI", max(8, int(self.h * 0.0125))),
            fill="#888888")
        self.hint = self.c.create_text(
            self.w * 0.62, self.h * 0.945,
            text="SLIDE  ↔  TO CHANGE CANVAS          SLIDE  ↕  TO CHANGE PALETTE",
            font=("Segoe UI", max(8, int(self.h * 0.0125))), fill="#777777")
        # wipe curtain on top of everything
        self.wipe_items = [self.c.create_line(0, 0, 1, 1, width=10,
                                              capstyle="round", fill="#000"),
                           self.c.create_line(0, 0, 1, 1, width=4,
                                              capstyle="round", fill="#FFF")]
        for it in self.wipe_items:
            self.c.itemconfig(it, state="hidden")
        for it in (self.title, self.subtitle, self.hint):
            self.c.itemconfig(it, state="hidden")
        self.set_visible(False)

    # ------------------------------------------------------------ helpers
    def set_fill(self, item, color, key):
        hexed = _hex(color)
        if self._fill_cache.get(item) == (key, hexed):
            return
        self._fill_cache[item] = (key, hexed)
        self.c.itemconfig(item, fill=hexed)

    def set_outline(self, item, color, key, width=1):
        hexed = _hex(color)
        if self._fill_cache.get(item) == (key, hexed, width):
            return
        self._fill_cache[item] = (key, hexed, width)
        self.c.itemconfig(item, outline=hexed, width=width)

    def coords(self, item, pts, q=1.0):
        key = tuple(int(round(v / q)) for v in pts)
        if self._coord_cache.get(item) == key:
            return
        self._coord_cache[item] = key
        self.c.coords(item, *pts)

    def show(self, item):
        if item not in self._visible:
            self._visible.add(item)
            self.c.itemconfig(item, state="normal")

    def hide(self, item):
        if item in self._visible:
            self._visible.discard(item)
            self.c.itemconfig(item, state="hidden")

    def _text(self, item, color, key):
        hexed = _hex(color)
        if self._text_cache.get(item) == (key, hexed):
            return
        self._text_cache[item] = (key, hexed)
        self.c.itemconfig(item, fill=hexed)

    # ------------------------------------------------------------ palette
    @property
    def style(self) -> BaseStyle:
        return self.styles[self.style_index]

    def _palette(self):
        style = self.style
        vp = self.variant_pos
        v0 = int(math.floor(vp)) % 3
        v1 = (v0 + 1) % 3
        vf = _smooth(vp - math.floor(vp))
        p0 = style.palettes[v0][1]
        p1 = style.palettes[v1][1]
        return [_mix(p0[i], p1[i], vf) for i in range(3)]

    def _variant_label(self):
        vp = self.variant_pos
        v = int(round(vp)) % 3
        return self.style.palettes[v][0]

    # ------------------------------------------------------------ lifecycle
    def set_visible(self, visible):
        self.visible = bool(visible)
        st = "normal" if visible else "hidden"
        for it in self.bg_items + self.particle_items + self.ring_items + \
                self.pointer_items + [self.title, self.subtitle, self.hint]:
            self.c.itemconfig(it, state=st)
        for pair in self.chips:
            for it in pair:
                self.c.itemconfig(it, state=st)
        self.style.set_visible(visible)
        if not visible:
            self._visible.clear()

    def reset_theme(self):
        self.style_index = 0
        self.variant_pos = 0.0
        self.variant_target = 0.0
        self.cells = self.styles[0].cells()
        for i, st in enumerate(self.styles):
            st.set_visible(i == 0)

    # ------------------------------------------------------------ styles
    def set_style(self, index, immediate=False):
        index = int(index) % N_STYLES
        if index == self.style_index and not immediate:
            return
        if immediate:
            self.style.set_visible(False)
            self.style_index = index
            self.styles[index].set_visible(self.visible)
            self.cells = self.styles[index].cells()
            self._wipe_committed = True
            for it in self.wipe_items:
                self.hide(it)
            return
        if self._wipe_t < 0 or self.time - self._wipe_t > 0.55:
            self._start_wipe(+1 if index > self.style_index else -1)

    def next_style(self, direction=1):
        if self.time - self._wipe_t < 0.55:
            return
        self._start_wipe(direction)

    def _start_wipe(self, direction):
        self._wipe_t = self.time
        self._wipe_dir = 1.0 if direction >= 0 else -1.0
        self._wipe_committed = False
        self.hint_until = self.time + 7.0

    def _commit_style(self):
        old = self.style_index
        self.style_index = (old + (1 if self._wipe_dir > 0 else -1)) % N_STYLES
        self.styles[old].set_visible(False)
        self.styles[self.style_index].set_visible(True)
        self.cells = self.styles[self.style_index].cells()
        self._wipe_committed = True
        # wipe flash particles from centre
        pal = self._palette()
        for i in range(10):
            self.burst(self.w * (0.3 + 0.4 * random.random()),
                       self.h * (0.3 + 0.4 * random.random()), pal, 1,
                       kind="spark")

    # ------------------------------------------------------------ input
    def press_index(self, idx):
        idx = int(idx) % 16
        self.press_cell(idx // self.n, idx % self.n)

    def press_cell(self, r, c):
        r = min(max(int(r), 0), self.n - 1)
        c = min(max(int(c), 0), self.n - 1)
        target = next((cell for cell in self.cells
                       if cell.r == r and cell.c == c), None)
        if target is None:
            return
        now = self.time
        for cell in self.cells:
            dist = (abs(cell.x - target.x) + abs(cell.y - target.y)) / \
                max(cell.size * 1.7, 1.0)
            rings = int(dist)
            cell.act_t = now + rings * 0.075
            cell.tint = (rings + cell.r + cell.c) % 3
            cell.amp = max(0.42, 1.0 - 0.09 * rings)
        self.ripple(target.x, target.y, 1.0)
        self.style.burst(target.x, target.y, self._palette(), 12)
        self.hint_until = self.time + 6.0

    def _cell_at(self, x, y):
        best = None
        for cell in self.cells:
            d = math.hypot(x - cell.x, y - cell.y)
            if d <= cell.size * 0.95 and (best is None or d < best[0]):
                best = (d, cell)
        return best[1] if best else None

    def tap(self, x, y):
        cell = self._cell_at(x, y)
        if cell is not None:
            self.press_cell(cell.r, cell.c)
            return True
        self.ripple(x, y, 0.7)
        self.burst(x, y, self._palette(), 6, kind="dab")
        return False

    def swipe(self, dx, dy):
        self.pointer = None
        self.pointer_trail.clear()
        self.hint_until = self.time + 7.0
        if abs(dx) >= abs(dy):
            self.next_style(1 if dx >= 0 else -1)
            return "style"
        self.variant_target = (self.variant_target + 1) % 3
        return "palette"

    def pointer_press(self, x, y):
        self.pointer = (x, y, self.time)
        self.pointer_trail = [(x, y, self.time)]

    def pointer_move(self, x, y):
        if self.pointer is None:
            return
        self.pointer_trail.append((x, y, self.time))
        if len(self.pointer_trail) > 16:
            self.pointer_trail.pop(0)

    def pointer_release(self, x, y):
        if self.pointer is None:
            return "none"
        x0, y0, t0 = self.pointer
        dx, dy = x - x0, y - y0
        dt = self.time - t0
        self.pointer = None
        self.pointer_trail.clear()
        if math.hypot(dx, dy) >= max(60.0, 0.06 * min(self.w, self.h)) and dt < 1.2:
            return self.swipe(dx, dy)
        self.tap(x0, y0)
        return "tap"

    # ------------------------------------------------------------ effects
    def ripple(self, x, y, strength):
        pal = self._palette()
        now = self.time
        for i in range(3):
            self.rings.append((x, y, now + i * 0.09, pal[i % 3], strength))
        self.rings = self.rings[-14:]

    def burst(self, x, y, pal, n=10, kind="spark"):
        free = [it for it in self.particle_items
                if it not in {p.item for p in self.particles}]
        for _ in range(min(n, len(free))):
            item = free.pop()
            ang = random.uniform(0, 2 * math.pi)
            if kind == "confetti":
                speed = random.uniform(160, 360)
                vx, vy = math.cos(ang) * speed, math.sin(ang) * speed - 160
                g, life = 420.0, random.uniform(0.7, 1.4)
            elif kind == "drip":
                vx, vy = random.uniform(-40, 40), random.uniform(30, 150)
                g, life = 260.0, random.uniform(0.9, 1.6)
            elif kind == "holy":
                vx, vy = math.cos(ang) * 40, -random.uniform(20, 80)
                g, life = -20.0, random.uniform(1.0, 1.8)
            elif kind == "dab":
                vx, vy = math.cos(ang) * 55, math.sin(ang) * 55
                g, life = 8.0, random.uniform(1.1, 2.0)
            else:
                speed = random.uniform(90, 260)
                vx, vy = math.cos(ang) * speed, math.sin(ang) * speed
                g, life = 0.0, random.uniform(0.3, 0.65)
            self.particles.append(Particle(
                item=item, x=x, y=y, vx=vx, vy=vy, gravity=g, age=0.0,
                life=life, size=random.uniform(0.004, 0.014) * self.h,
                color=pal[random.randrange(3)],
                drift=random.uniform(-1, 1)))

    # ------------------------------------------------------------ frame
    def update(self, level, bands, palette, hint="", accent=0.0, beat_phase=0.0):
        now = time.perf_counter()
        dt = min(max(now - self._last_t, 1e-4), 0.12)
        self._last_t = now
        self.time += dt
        t = self.time
        self.level = min(max(float(level), 0.0), 1.0)
        self.accent = min(max(float(accent), 0.0), 1.0)
        self.beat_phase = float(beat_phase) % 1.0
        if bands is not None:
            b = np.asarray(bands, dtype=np.float32).reshape(-1)
            if b.size:
                self.bands = b

        self.variant_pos += (self.variant_target - self.variant_pos) * (
            1.0 - math.exp(-dt / 0.22))

        # horizontal wipe transition (may commit the style change, so the
        # active style must be resolved *after* this block)
        if self._wipe_t >= 0:
            p = (t - self._wipe_t) / 0.52
            if not self._wipe_committed and p >= 0.45:
                self._commit_style()
            if p >= 1.0:
                self._wipe_t = -1e9
                for it in self.wipe_items:
                    self.hide(it)
            else:
                es = _ease_out(min(max(p, 0.0), 1.0))
                x = self.w * (-0.25 + 1.5 * es) * self._wipe_dir \
                    + (self.w * 0.5) * (1 - abs(self._wipe_dir))
                wk = 0.32 + 0.9 * self.level
                col = _mix(self._palette()[0], (255, 255, 255), 0.35)
                band = self.wipe_items[0]
                self.coords(band, (x, -self.h * 0.2, x, self.h * 1.2), 2.0)
                self.c.itemconfig(band, width=max(60, int(self.w * 0.20)))
                self.set_fill(band, _scale(col, wk * 0.55), ("wipe", round(x, 0)))
                self.show(band)
                edge = self.wipe_items[1]
                ex = x - self._wipe_dir * self.w * 0.06
                self.coords(edge, (ex, -self.h * 0.2, ex, self.h * 1.2), 2.0)
                self.c.itemconfig(edge, width=max(4, int(self.w * 0.006)))
                self.set_fill(edge, _scale(_mix(col, (255, 255, 255), 0.6), wk),
                              ("wipe2", round(ex, 0)))
                self.show(edge)

        style = self.style
        pal = self._palette()
        light = 1.0 if style.light else 0.0
        k_dark = min(max(0.30 + 0.95 * self.level, 0.20), 1.25)
        k_light = 0.88 + 0.13 * self.level
        k = k_dark * (1 - light) + k_light * light
        kb = round(k, 2)
        vi = int(round(self.variant_pos)) % 3
        bg_top = style.bg[0]
        bg_bot = style.bg[1]

        # background gradient
        for i, it in enumerate(self.bg_items):
            f = i / (self.n_bands - 1)
            col = _mix(bg_top, bg_bot, f ** 1.35)
            kk = kb * (1.0 - 0.14 * (1 - light) * f)
            self.set_fill(it, _scale(col, kk), ("bg", i, kb, vi))
            self.show(it)

        # style background
        style.draw(t, k, pal, self.level, self.accent)

        # play cells
        if t - self._chip_last >= 0.025:
            self._chip_last = t
            self._draw_chips(t, k, kb, vi, pal, style)

        # rings
        live = 0
        max_r = math.hypot(self.w, self.h) * 0.5
        for (rx, ry, t0, color, strength) in self.rings:
            age = t - t0
            if age < 0 or age > 1.05 or live >= len(self.ring_items):
                continue
            it = self.ring_items[live]
            live += 1
            p = _ease_out(age / 0.95)
            radius = max(4.0, p * max_r * 0.9)
            self.coords(it, (rx - radius, ry - radius,
                             rx + radius, ry + radius), 2.0)
            fade = max(0.0, 1.0 - age / 1.05) * strength
            col = _mix(color, bg_top, 1.0 - 0.62 * fade)
            self.set_outline(it, _scale(col, k * (0.85 + 0.35 * fade)),
                             ("ring", round(radius, 1), round(fade, 2), vi),
                             max(1, int(2.6 * fade + 0.6)))
            self.show(it)
        self.rings = [r for r in self.rings if t - r[2] <= 1.1]
        for it in self.ring_items[live:]:
            self.hide(it)

        # particles
        used = set()
        for p in self.particles:
            p.age += dt
            if p.age >= p.life:
                continue
            p.vy += p.gravity * dt
            p.x += p.vx * dt + math.sin(t * 2.0 + p.drift * 5.0) * p.drift * 18 * dt
            p.y += p.vy * dt
            fl = 1.0 - p.age / p.life
            col = _mix(p.color, bg_top, 1.0 - fl)
            size = p.size * (0.35 + 0.65 * fl)
            used.add(p.item)
            self.coords(p.item, (p.x - size, p.y - size,
                                 p.x + size, p.y + size), 1.0)
            self.set_fill(p.item, _scale(col, k * (0.5 + 0.6 * fl)),
                          ("part", round(p.age, 2), round(size, 1), vi))
            self.show(p.item)
        self.particles = [p for p in self.particles if p.age < p.life]
        for it in self.particle_items:
            if it not in used:
                self.hide(it)

        # beat ambience
        if self.beat_phase < self._prev_phase - 0.5:
            style.on_beat(t, pal)
        self._prev_phase = self.beat_phase

        # pointer
        if self.pointer is not None:
            x, y, _ = self.pointer
            r = max(16.0, self.h * 0.03)
            self.coords(self.pointer_items[0], (x - r, y - r, x + r, y + r), 2.0)
            self.set_outline(self.pointer_items[0], _scale(pal[0], k),
                             ("ptr", round(x, 0), round(y, 0), vi))
            self.show(self.pointer_items[0])
            if len(self.pointer_trail) > 1:
                x0, y0, _ = self.pointer_trail[0]
                self.coords(self.pointer_items[1], (x0, y0, x, y), 2.0)
                self.set_fill(self.pointer_items[1], _scale(pal[1], k * 0.8),
                              ("ptr2", round(x, 0), round(y, 0), vi))
                self.show(self.pointer_items[1])
        else:
            for it in self.pointer_items:
                self.hide(it)

        # text
        text_col = style.text
        self._text(self.title, _scale(text_col, k), ("title", kb, vi))
        self._text(self.subtitle,
                   _scale(_mix(text_col, pal[0], 0.18 * (1 - light)), kb * 0.88),
                   ("sub", kb, vi))
        self.c.itemconfig(self.subtitle,
                          text=f"{style.name}   ·   {self._variant_label()}")
        show_hint = t < self.hint_until
        if show_hint:
            self._text(self.hint, _scale(_mix(text_col, bg_top, 0.55), kb),
                       ("hint", kb, vi))
            self.show(self.hint)
        else:
            self.hide(self.hint)

    # ------------------------------------------------------------ caps
    def _draw_chips(self, t, k, kb, vi, pal, style):
        chip = style.chip
        offx, offy = chip["glow_off"]
        for i, cell in enumerate(self.cells):
            age = t - cell.act_t
            if age < 0:
                env = 0.0
            elif age < 0.07:
                env = _ease_out(age / 0.07)
            else:
                env = math.exp(-(age - 0.07) / 0.42)
            press = env * cell.amp
            bob = math.sin(t * 0.7 + i * 0.9) * (1.2 + 1.8 * self.level)
            band = 0.0
            if self.bands.size:
                band = float(self.bands[i % self.bands.size])
            lift = 7.0 * press + bob
            size = cell.size * (1.0 + 0.12 * press)
            cx, cy = cell.x, cell.y - lift
            pk = round(press, 2)
            tint = pal[(cell.tint + i) % 3]
            glow_item, body_item = self.chips[i]

            gsize = size * (1.34 - 0.32 * (1.0 if style.light else 0.0))
            pts = []
            for ux, uy in cell.unit:
                pts.extend((cx + offx + ux * gsize, cy + offy + uy * gsize))
            self.coords(glow_item, pts, 1.0)
            pts = []
            for ux, uy in cell.unit:
                pts.extend((cx + ux * size, cy + uy * size))
            self.coords(body_item, pts, 1.0)

            body_col = _mix(style.base, _mix(tint, (255, 255, 255), 0.10),
                            min(0.90, chip["idle"]
                                + (0.90 - chip["idle"]) * press ** 0.8
                                + 0.012 * band))
            body_col = _scale(body_col, k * (0.95 + 0.26 * press))
            self.set_fill(body_item, body_col,
                          ("body", pk, round(band, 2), kb, vi))
            halo = _mix(tint, style.bg[0], 0.80 - 0.55 * press)
            self.set_fill(glow_item, _scale(halo, k * chip["glow"]
                                            * (1.0 + 1.1 * press)),
                          ("glow", pk, round(band, 2), kb, vi))
            border = chip["border"]
            if border > 0.05:
                edge = _mix(tint, style.base, 0.45)
                self.set_outline(body_item, _scale(edge, k),
                                 ("edge", vi, kb, pk), chip["border_w"])
            elif style.light:
                self.set_outline(body_item,
                                 _scale(_mix(style.bg[0], style.text, 0.18), k),
                                 ("edge_l", vi, kb), 1)
            else:
                self.set_outline(body_item,
                                 _scale(_mix(style.bg[0], style.base, 0.6), k),
                                 ("edge_d", vi, kb), 1)
            self.show(glow_item)
            self.show(body_item)

    # ------------------------------------------------------------ info
    def status(self):
        return f"{self.style.name} · {self._variant_label()}"

    @property
    def style_count(self):
        return N_STYLES
