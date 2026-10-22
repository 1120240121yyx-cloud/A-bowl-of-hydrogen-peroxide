"""Creative light-control slider for the Tk panel.

Not a rectangular scale: a slowly breathing "light string" (sine ribbon), a
luminous bead knob, a comet trail while dragging and a ripple on release.
Dragging is eased (the bead lags the pointer) so it feels smooth, and the
value is reported continuously so the screen reacts while the user drags.
"""
from __future__ import annotations

import math
import tkinter as tk


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _hex(c):
    return "#{:02X}{:02X}{:02X}".format(*[max(0, min(255, int(v))) for v in c])


class LightSlider(tk.Canvas):
    def __init__(self, parent, label: str, value: float = 0.4,
                 accents=((86, 225, 255), (197, 107, 255)),
                 bg: str = "#0E0E10", on_change=None, width: int = 608,
                 height: int = 42) -> None:
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.w = int(width)
        self.h = int(height)
        self.label = label
        self.c1 = tuple(accents[0])
        self.c2 = tuple(accents[1])
        self.bg_rgb = (14, 14, 16)
        self.on_change = on_change
        self.value = max(0.0, min(1.0, float(value)))
        self.target = self.value
        self._phase = 0.0
        self._dragging = False
        self._last = None
        self._trail: list[list] = []
        self._ripples: list[list] = []
        self._pad_l = 86 if self.w < 400 else 104
        self._pad_r = 42 if self.w < 400 else 76
        self._cy = self.h / 2.0
        self._build()
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self._tick()

    # ------------------------------------------------------------ items
    def _build(self) -> None:
        self.lbl = self.create_text(4, self._cy, text=self.label, anchor="w",
                                    fill="#C9C9D4",
                                    font=("Microsoft YaHei", 9))
        self.val = self.create_text(self.w - 4, self._cy, text="0%", anchor="e",
                                    fill=_hex(self.c2),
                                    font=("Consolas", 10, "bold"))
        # ghost / glow / ribbon / active core / knob halo / ring / core
        self.shadow = self.create_line(0, 0, 1, 0, width=15,
                                       fill=_hex(_mix(self.bg_rgb, self.c1, 0.22)),
                                       capstyle="round", smooth=True)
        self.ribbon = self.create_line(0, 0, 1, 0, width=5,
                                       fill=_hex(_mix(self.bg_rgb, self.c1, 0.55)),
                                       capstyle="round", smooth=True)
        self.hot = self.create_line(0, 0, 1, 0, width=5,
                                    fill=_hex(self.c2), capstyle="round",
                                    smooth=True)
        self.halo = self.create_oval(0, 0, 1, 1, outline="", fill="")
        self.ring = self.create_oval(0, 0, 1, 1, outline=_hex(self.c2),
                                     width=2)
        self.core = self.create_oval(0, 0, 1, 1, outline="", fill="#FFFFFF")
        self.trail_items = [self.create_oval(0, 0, 1, 1, outline="", fill="")
                            for _ in range(7)]
        self.ripple_items = [self.create_oval(0, 0, 1, 1, outline="", width=2)
                             for _ in range(3)]

    # ------------------------------------------------------------ geometry
    def _x_for(self, v: float) -> float:
        return self._pad_l + v * (self.w - self._pad_l - self._pad_r)

    def _v_for(self, x: float) -> float:
        span = max(1.0, self.w - self._pad_l - self._pad_r)
        return min(max((x - self._pad_l) / span, 0.0), 1.0)

    def _wave(self, x0: float, x1: float, n: int = 26) -> list[float]:
        pts = []
        for i in range(n + 1):
            f = i / n
            x = x0 + (x1 - x0) * f
            y = self._cy + math.sin(self._phase + f * 5.2) * 2.2
            pts.extend((x, y))
        return pts

    # ------------------------------------------------------------ events
    def _press(self, event) -> None:
        self._dragging = True
        self._set_target(self._v_for(event.x))

    def _motion(self, event) -> None:
        if self._dragging:
            self._set_target(self._v_for(event.x))

    def _release(self, event) -> None:
        self._dragging = False
        x = self._x_for(self.value)
        self._ripples.append([x, self._cy, 0.0])
        self._set_target(self._v_for(event.x))

    def _set_target(self, v: float) -> None:
        self.target = min(max(float(v), 0.0), 1.0)
        if self.on_change is not None:
            try:
                self.on_change(self.target)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ animation
    def _tick(self) -> None:
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self.value += (self.target - self.value) * 0.32
        if abs(self.target - self.value) < 0.001:
            self.value = self.target
        self._phase += 0.04
        self._draw()
        try:
            self.after(33, self._tick)
        except tk.TclError:
            pass

    def _draw(self) -> None:
        x0, x1 = self._pad_l, self.w - self._pad_r
        vx = self._x_for(self.value)
        base_pts = self._wave(x0, x1)
        self.coords(self.shadow, *base_pts)
        self.coords(self.ribbon, *base_pts)
        hot_pts = self._wave(x0, vx) if vx > x0 + 2 else [x0, self._cy, x0 + 1, self._cy]
        self.coords(self.hot, *hot_pts)
        # bright core of the track is a second, thinner pass
        self.coords(self.shadow, *base_pts)

        ky = self._cy + math.sin(self._phase + (vx - x0) / max(1, x1 - x0) * 5.2) * 2.2
        pulse = 1.0 + 0.08 * math.sin(self._phase * 3.0)
        r = 13.0 * pulse
        self.coords(self.halo, vx - r * 1.9, ky - r * 1.9, vx + r * 1.9, ky + r * 1.9)
        self.itemconfig(self.halo, fill=_hex(_mix(self.bg_rgb, self.c2, 0.30)))
        self.coords(self.ring, vx - r, ky - r, vx + r, ky + r)
        self.coords(self.core, vx - 3.2, ky - 3.2, vx + 3.2, ky + 3.2)

        # comet trail
        self._trail.append([vx, ky, 0.0])
        if len(self._trail) > 28:
            self._trail = self._trail[-28:]
        for i, item in enumerate(self.trail_items):
            idx = len(self._trail) - 1 - (i + 1) * 3
            if idx >= 0:
                tx, ty, _ = self._trail[idx]
                rr = max(1.0, 6.5 - i * 0.7)
                self.coords(item, tx - rr, ty - rr, tx + rr, ty + rr)
                col = _mix(self.c1, self.bg_rgb, 0.25 + i * 0.11)
                self.itemconfig(item, fill=_hex(col))
            else:
                self.coords(item, -10, -10, -9, -9)

        # release ripples
        for rip in self._ripples:
            rip[2] += 0.07
        self._ripples = [r for r in self._ripples if r[2] < 1.0]
        for i, item in enumerate(self.ripple_items):
            if i < len(self._ripples):
                rx, ry, age = self._ripples[i]
                rr = 8 + age * 42
                self.coords(item, rx - rr, ry - rr, rx + rr, ry + rr)
                self.itemconfig(item, outline=_hex(_mix(self.c2, self.bg_rgb, age)),
                                width=max(1, int(2.2 * (1 - age))))
            else:
                self.coords(item, -10, -10, -9, -9)

        self.itemconfig(self.val, text=f"{int(round(self.value * 100))}%")
        # active ribbon brightens toward the bead
        self.itemconfig(self.hot, fill=_hex(_mix(self.c2, (255, 255, 255), 0.18)))
        self.itemconfig(self.ribbon, fill=_hex(_mix(self.bg_rgb, self.c1, 0.55)))

    # ------------------------------------------------------------ API
    def set_value(self, value: float, notify: bool = False) -> None:
        self.value = min(max(float(value), 0.0), 1.0)
        self.target = self.value
        if notify and self.on_change is not None:
            self.on_change(self.value)
