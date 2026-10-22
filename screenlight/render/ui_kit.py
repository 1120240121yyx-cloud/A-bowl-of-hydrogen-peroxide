"""Small Tk canvas kit used by the polished control panel.

Tk's native widgets are functional but look dated.  These are lightweight
canvas widgets with rounded shapes, warm accents and quiet hover states, so the
panel feels like a product rather than a debug console.  They keep the minimal
API the app already uses (``configure(text=...)``, ``set``, ``invoke``).
"""
from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont

PANEL = "#101216"
CARD = "#15181D"
CARD2 = "#12151A"
HAIR = "#242830"
FG = "#F2F3F5"
MUTED = "#8B909A"
FAINT = "#5E646E"
AMBER = "#F2A33A"
CORAL = "#FF6B4A"
MINT = "#6EE7B7"


def hexmix(a: str, b: str, t: float) -> str:
    a = a.lstrip("#")
    b = b.lstrip("#")
    ca = [int(a[i:i + 2], 16) for i in (0, 2, 4)]
    cb = [int(b[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(ca[i] + (cb[i] - ca[i]) * t):02X}" for i in range(3))


def rr(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float,
       r: float, **kw) -> int:
    """Rounded rectangle as a smooth polygon."""
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, splinesteps=18, **kw)


class PillButton(tk.Canvas):
    def __init__(self, parent, text: str, command, width: int = 130,
                 height: int = 32, *, bg: str = PANEL, accent: bool = False,
                 font=("Microsoft YaHei UI", 9), radius: int = 15) -> None:
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.w, self.h = width, height
        self.command = command
        self.accent = accent
        self._bg = CARD
        self._hover = "#1C2027"
        self._text = text
        self._font = font
        self._active = False
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        fill = hexmix(AMBER, CARD, 0.82) if self.accent else self._bg
        edge = AMBER if self.accent else HAIR
        self._shape = rr(self, 1, 1, self.w - 1, self.h - 1, self.h * 0.42,
                         fill=fill, outline=edge if self.accent else "",
                         width=1)
        self._label = self.create_text(self.w / 2, self.h / 2, text=self._text,
                                       fill="#1B1206" if self.accent else FG,
                                       font=self._font)
        self.tag_bind(self._shape, "<Enter>", self._enter)
        self.tag_bind(self._shape, "<Leave>", self._leave)
        self.tag_bind(self._shape, "<ButtonPress-1>", self._press)
        self.tag_bind(self._shape, "<ButtonRelease-1>", self._release)
        self.tag_bind(self._label, "<Enter>", self._enter)
        self.tag_bind(self._label, "<Leave>", self._leave)
        self.tag_bind(self._label, "<ButtonPress-1>", self._press)
        self.tag_bind(self._label, "<ButtonRelease-1>", self._release)

    def _enter(self, _=None) -> None:
        if not self.accent:
            self.itemconfigure(self._shape, fill=self._hover)
            self.itemconfigure(self._label, fill="#FFFFFF")

    def _leave(self, _=None) -> None:
        if not self.accent:
            self.itemconfigure(self._shape, fill=self._bg)
            self.itemconfigure(self._label, fill=FG)

    def _press(self, _=None) -> None:
        self._active = True
        self.itemconfigure(self._shape, fill=hexmix(AMBER, CARD, 0.72))

    def _release(self, _=None) -> None:
        if not self._active:
            return
        self._active = False
        self._leave()
        try:
            self.command()
        except Exception:  # noqa: BLE001 - UI should never crash
            pass

    def configure(self, **kw):  # noqa: D401 - Tk-compatible API
        if "text" in kw:
            self._text = str(kw.pop("text"))
            self.itemconfigure(self._label, text=self._text)
        if kw:
            super().configure(**kw)

    config = configure

    def invoke(self) -> None:
        try:
            self.command()
        except Exception:  # noqa: BLE001
            pass


class MiniCard(tk.Canvas):
    def __init__(self, parent, title: str, value: str = "—", width: int = 150,
                 height: int = 46, *, bg: str = PANEL, color: str = FG) -> None:
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.w, self.h = width, height
        self._color = color
        rr(self, 1, 1, width - 1, height - 1, 11, fill=CARD, outline=HAIR)
        self.create_text(14, 13, text=title.upper(), anchor="w", fill=FAINT,
                         font=("Microsoft YaHei UI", 7))
        self._value = self.create_text(14, 31, text=value, anchor="w",
                                       fill=color,
                                       font=("Microsoft YaHei UI", 10, "bold"))

    def set(self, value: str, color: str | None = None) -> None:
        self.itemconfigure(self._value, text=str(value))
        if color:
            self.itemconfigure(self._value, fill=color)


class LevelBar(tk.Canvas):
    def __init__(self, parent, width: int = 520, height: int = 18,
                 *, bg: str = PANEL) -> None:
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0)
        self.w, self.h = width, height
        self._level = 0.0
        rr(self, 0, 3, width, height - 3, (height - 6) / 2,
           fill="#0C0E12", outline=HAIR)
        self._segments = []
        n = 44
        for i in range(n):
            x1 = 3 + i * (width - 6) / n
            x2 = x1 + (width - 6) / n - 2
            item = self.create_line(x1, height / 2, x2, height / 2,
                                    width=4, capstyle="round", fill="#1C2028")
            self._segments.append(item)
        self._glow = self.create_oval(0, 0, 0, 0, outline="", fill="")

    def set(self, level: float) -> None:
        self._level = max(0.0, min(1.0, float(level)))
        n = len(self._segments)
        active = int(round(self._level * n))
        for i, item in enumerate(self._segments):
            if i < active:
                t = i / max(1, n - 1)
                self.itemconfigure(item, fill=hexmix(AMBER, CORAL, t))
            else:
                self.itemconfigure(item, fill="#1C2028")
        x = 4 + self._level * (self.w - 8)
        self.coords(self._glow, x - 5, self.h / 2 - 5, x + 5, self.h / 2 + 5)
        self.itemconfigure(self._glow, fill=hexmix(AMBER, "#FFFFFF", 0.35))


class TimelineBar(tk.Canvas):
    """Slim seek bar with a warm playhead; API-compatible with the old Scale."""

    def __init__(self, parent, width: int = 640, height: int = 24,
                 *, bg: str = CARD, command=None) -> None:
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.w, self.h = width, height
        self.command = command
        self._duration = 1.0
        self._value = 0.0
        self._busy = False
        rr(self, 0, 9, width, height - 9, 3, fill="#090B0E", outline="")
        self._fill = self.create_line(0, height / 2, 0, height / 2, width=5,
                                      capstyle="round", fill=AMBER)
        self._dot = self.create_oval(0, 0, 0, 0, outline="", fill="#FFF3DF")
        self.bind("<ButtonPress-1>", self._seek)
        self.bind("<B1-Motion>", self._seek)
        self.bind("<ButtonRelease-1>", self._seek)

    def _x(self) -> float:
        return 3 + (self.w - 6) * (self._value / max(self._duration, 0.01))

    def _seek(self, event) -> None:  # noqa: ANN001
        if self.command is None:
            return
        v = (event.x - 3) / max(1, self.w - 6) * self._duration
        self.set(v)
        self.command(v)

    def set(self, value: float) -> None:
        self._value = max(0.0, min(float(value), self._duration))
        x = self._x()
        self.coords(self._fill, 3, self.h / 2, x, self.h / 2)
        self.coords(self._dot, x - 5, self.h / 2 - 5, x + 5, self.h / 2 + 5)

    def get(self) -> float:
        return self._value

    def configure(self, **kw):  # noqa: D401 - Tk-compatible API
        if "to" in kw:
            self._duration = max(float(kw.pop("to")), 0.01)
            self.set(self._value)
        if kw:
            super().configure(**kw)

    config = configure
