"""OpenRay desktop pet — the little light-spirit that opens the control panel.

The pet is deliberately small (60% of the previous size), with oversized
heterochromatic eyes, several expressions (normal / happy / cheeky / disdain /
wake), an occasional whole-body colour flip, and a bushy fox-like tail.

It is drawn with Tk canvas (no external assets), lives in a transparent
always-on-top window, and is the primary menu: click to open the control panel,
drag to move, double-click for next track, long-press to pause, right-click for
a menu.
"""
from __future__ import annotations

import math
import random
import time
import tkinter as tk

BG = "#010203"          # transparentcolour key
CREAM = "#FFE3B0"
AMBER = "#F2A33A"
CORAL = "#FF6B4A"
INK = "#24160C"
PINK = "#FF9FA1"
CYAN = "#7FE3E7"
LILAC = "#B78CFF"
MINT = "#9BE7A8"
ROSE = "#FF8FB8"

FLIP_COLORS = (LILAC, CYAN, ROSE, MINT, "#FFB35A")
IRIS_COLORS = ("#FFB84D", "#6FE0FF")  # heterochromia: amber left, cyan right
CUTE_TAIL_STYLES = ("plush", "puff", "star")
WEIRD_TAIL_STYLES = ("eye", "spike", "tentacle", "segmented", "mouth", "crystal")
TAIL_ACTIONS = ("fan", "curl", "wave", "shiver")


def _mix(a: str, b: str, t: float) -> str:
    a, b = a.lstrip("#"), b.lstrip("#")
    ca = [int(a[i:i + 2], 16) for i in (0, 2, 4)]
    cb = [int(b[i:i + 2], 16) for i in (0, 2, 4)]
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{int(ca[i] + (cb[i] - ca[i]) * t):02X}" for i in range(3))


class _ScaledCanvas:
    """Tiny proxy that scales design-space canvas calls to screen space."""

    def __init__(self, canvas: tk.Canvas, factor: float) -> None:
        self.c = canvas
        self.f = factor

    def _flat(self, args):
        out = []
        for v in args:
            if isinstance(v, (list, tuple)):
                out.extend(self._flat(v))
            else:
                out.append(float(v) * self.f)
        return out

    def _kw(self, kw):
        out = dict(kw)
        if "width" in out:
            try:
                out["width"] = max(1, int(round(float(out["width"]) * self.f)))
            except (TypeError, ValueError):
                pass
        font = out.get("font")
        if isinstance(font, tuple) and len(font) >= 2:
            f = list(font)
            try:
                f[1] = max(6, int(round(int(f[1]) * self.f)))
            except (TypeError, ValueError):
                pass
            out["font"] = tuple(f)
        return out

    def create_oval(self, *args, **kw):
        return self.c.create_oval(*self._flat(args), **self._kw(kw))

    def create_rectangle(self, *args, **kw):
        return self.c.create_rectangle(*self._flat(args), **self._kw(kw))

    def create_arc(self, *args, **kw):
        return self.c.create_arc(*self._flat(args), **self._kw(kw))

    def create_line(self, *args, **kw):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self.c.create_line(*self._flat(args[0]), **self._kw(kw))
        return self.c.create_line(*self._flat(args), **self._kw(kw))

    def create_polygon(self, *args, **kw):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self.c.create_polygon(*self._flat(args[0]), **self._kw(kw))
        return self.c.create_polygon(*self._flat(args), **self._kw(kw))

    def create_text(self, x, y, **kw):
        return self.c.create_text(float(x) * self.f, float(y) * self.f, **self._kw(kw))

    def delete(self, *args, **kw):
        return self.c.delete(*args, **kw)

class PetWindow:
    # design canvas is 250×270; the on-screen pet is 60% of that
    DESIGN_W, DESIGN_H = 250, 270
    SCALE = 0.6
    # window is larger than the 60% pet so 1–9 exaggerated tails can breathe
    W = 240
    H = 220

    def __init__(self, root: tk.Misc, on_open, on_quit,
                 on_next=None, on_long=None, on_move=None,
                 on_xiaomao=None) -> None:
        self.on_open = on_open
        self.on_quit = on_quit
        self.on_next = on_next
        self.on_long = on_long
        self.on_move = on_move
        self.on_xiaomao = on_xiaomao
        self._press_xy = None
        self._win_xy = None
        self._press_t = 0.0
        self._dragged = False
        self._long_fired = False
        self._after_long = None
        self._after_single = None
        self._last_click = 0.0
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-transparentcolor", BG)
        except tk.TclError:
            self.win.configure(bg=BG)
        self.win.configure(bg=BG)
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"{self.W}x{self.H}+{sw - self.W - 34}+{sh - self.H - 90}")
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self.level = 0.0
        self.bpm = 0.0
        self.accent = 0.0
        self.emotion = "balanced"
        self.valence = 0.5
        self._t = 0.0
        self._blink = 0.0
        self._hover = False
        self._closed = False
        self._wake_until = 0.0
        self._wake_text = "我在～"
        self._wake_flash = 0.0
        # expression / colour state
        self.expression = "normal"
        self._expr_until = 0.0
        self._next_expr = time.monotonic() + random.uniform(4.0, 7.0)
        self._flip_until = 0.0
        self._next_flip = time.monotonic() + random.uniform(9.0, 16.0)
        self._flip_color = random.choice(FLIP_COLORS)
        self.tail_count = 1
        self.tail_theme = "cute"
        self.tail_styles = []
        self.tail_curves = []
        self.tail_phases = []
        self.tail_colors = []
        self._tail_action = "idle"
        self._tail_action_start = 0.0
        self._tail_action_until = 0.0
        self._next_tail_action = time.monotonic() + 4.0
        self._next_tail_reroll = time.monotonic() + 22.0
        self._tail_bloom_until = 0.0
        self._reroll_tails(initial=True)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Enter>", self._enter)
        self.canvas.bind("<Leave>", self._leave)
        self.canvas.bind("<Button-3>", self._menu)
        self._menu = tk.Menu(self.win, tearoff=0, bg="#15181D", fg="#F2F3F5",
                             activebackground=AMBER, activeforeground="#1A1206",
                             font=("Microsoft YaHei UI", 9))
        self._menu.add_command(label="打开 OpenRay 面板", command=self.on_open)
        self._menu.add_command(label="下一首", command=self.on_next)
        self._menu.add_command(label="换一副尾巴", command=self._reroll_tails)
        self._menu.add_command(label="小茂模式（仅宠物区域）", command=self._toggle_xiaomao)
        self._menu.add_separator()
        self._menu.add_command(label="退出 OpenRay", command=self.on_quit)
        self._draw()
        self.win.after(33, self._tick)
        self.win.after(400, self._lift)

    # ------------------------------------------------------------------ input
    def _toggle_xiaomao(self) -> None:
        if self.on_xiaomao is not None:
            self.on_xiaomao()

    def _press(self, event) -> None:
        self._cancel_single()
        self._press_xy = (event.x_root, event.y_root)
        self._win_xy = (self.win.winfo_x(), self.win.winfo_y())
        self._press_t = time.perf_counter()
        self._dragged = False
        self._long_fired = False
        try:
            self._after_long = self.win.after(650, self._fire_long)
        except tk.TclError:
            self._after_long = None

    def _motion(self, event) -> None:
        if self._press_xy is None:
            return
        dx = event.x_root - self._press_xy[0]
        dy = event.y_root - self._press_xy[1]
        if not self._dragged and abs(dx) + abs(dy) < 5:
            return
        self._dragged = True
        self._cancel_long()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = max(0, min(sw - self.W, self._win_xy[0] + dx))
        y = max(0, min(sh - self.H, self._win_xy[1] + dy))
        try:
            self.win.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass
        if self.on_move is not None:
            try:
                self.on_move()
            except Exception:  # noqa: BLE001
                pass

    def _release(self, event) -> None:
        self._cancel_long()
        if self._press_xy is None:
            return
        if self._dragged:
            self._press_xy = None
            return
        if self._long_fired:
            self._press_xy = None
            return
        now = time.perf_counter()
        if now - self._last_click < 0.34:
            self._last_click = 0.0
            self._cancel_single()
            if self.on_next is not None:
                self.on_next()
            self._press_xy = None
            return
        self._last_click = now
        self._cancel_single()
        try:
            self._after_single = self.win.after(300, self._single_click)
        except tk.TclError:
            self._after_single = None
        self._press_xy = None

    def _single_click(self) -> None:
        self._after_single = None
        if not self._dragged and not self._long_fired:
            self.on_open()

    def _fire_long(self) -> None:
        self._after_long = None
        if self._dragged:
            return
        self._long_fired = True
        if self.on_long is not None:
            self.on_long()

    def _cancel_long(self) -> None:
        if self._after_long is not None:
            try:
                self.win.after_cancel(self._after_long)
            except tk.TclError:
                pass
            self._after_long = None

    def _cancel_single(self) -> None:
        if self._after_single is not None:
            try:
                self.win.after_cancel(self._after_single)
            except tk.TclError:
                pass
            self._after_single = None

    def _enter(self, _e=None) -> None:
        self._hover = True

    def _leave(self, _e=None) -> None:
        self._hover = False

    def _menu(self, event) -> None:
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def _lift(self) -> None:
        if self._closed:
            return
        try:
            self.win.lift()
        except tk.TclError:
            return
        self.win.after(450, self._lift)

    # ------------------------------------------------------------------ motion
    def set_motion(self, level: float, bpm: float = 0.0, accent: float = 0.0,
                   emotion: str = "balanced", valence: float = 0.5) -> None:
        self.level = max(0.0, min(1.0, float(level)))
        self.bpm = float(bpm or 0.0)
        self.accent = max(0.0, float(accent))
        self.emotion = emotion or "balanced"
        self.valence = max(0.0, min(1.0, float(valence)))

    def set_expression(self, name: str, duration: float = 2.8) -> None:
        self.expression = name if name in ("normal", "happy", "funny", "disdain") else "normal"
        self._expr_until = time.monotonic() + duration

    def wake_react(self, text: str = "我在～") -> None:
        """Ear-perk + bright eyes + speech bubble for ~2.6 seconds."""
        self._wake_until = time.monotonic() + 2.6
        self._wake_text = text or "我在～"
        self._wake_flash = 1.0
        self.expression = "normal"
        try:
            self.win.lift()
        except tk.TclError:
            pass

    def _tick(self) -> None:
        if self._closed:
            return
        self._t += 0.033
        now = time.monotonic()
        awake = now < self._wake_until
        self._tail_tick(now)
        if not awake:
            if self.expression != "normal" and now > self._expr_until:
                self.expression = "normal"
            if now > self._next_expr:
                self.expression = random.choices(
                    ("funny", "disdain", "happy", "normal"),
                    weights=(34, 27, 27, 12), k=1)[0]
                self._expr_until = now + random.uniform(2.2, 3.4)
                self._next_expr = now + random.uniform(5.5, 10.0)
            if now > self._next_flip:
                self._flip_until = now + random.uniform(0.7, 1.1)
                self._flip_color = random.choice(FLIP_COLORS)
                self._next_flip = now + random.uniform(10.0, 18.0)
        beat = 0.5 + 0.5 * math.sin(self._t * math.pi * 2 * (self.bpm / 60.0 or 1.2))
        self._draw(beat)
        self.win.after(33, self._tick)

    # ------------------------------------------------------------------ tails
    def _reroll_tails(self, initial: bool = False) -> None:
        now = time.monotonic()
        self.tail_count = random.randint(1, 9)
        self.tail_theme = random.choice(("cute", "weird"))
        pool = CUTE_TAIL_STYLES if self.tail_theme == "cute" else WEIRD_TAIL_STYLES
        self.tail_styles = [random.choice(pool) for _ in range(self.tail_count)]
        self.tail_curves = [random.uniform(-1.0, 1.0) for _ in range(self.tail_count)]
        self.tail_phases = [random.uniform(0.0, math.tau) for _ in range(self.tail_count)]
        palette = ("#6D3BFF", "#00E5FF", "#FF2DB3", "#B6FF3B", "#FF7A18")
        self.tail_colors = [random.choice(palette) for _ in range(self.tail_count)]
        self._next_tail_reroll = now + random.uniform(18.0, 32.0)
        self._tail_bloom_until = 0.0 if initial else now + 0.85
        self._tail_action = "idle"
        self._next_tail_action = now + random.uniform(3.0, 6.0)

    def _tail_tick(self, now: float) -> None:
        if now > self._next_tail_reroll:
            self._reroll_tails()
        if now > self._next_tail_action:
            self._tail_action = random.choice(TAIL_ACTIONS)
            self._tail_action_start = now
            self._tail_action_until = now + random.uniform(0.8, 1.5)
            self._next_tail_action = now + random.uniform(6.0, 11.0)
        if self._tail_action != "idle" and now > self._tail_action_until:
            self._tail_action = "idle"

    def _draw_tails(self, c, cx: float, cy: float, warm: str, cream: str,
                    glow: str, now: float) -> None:
        n = self.tail_count
        if n <= 0:
            return
        if now < self._tail_bloom_until:
            p = max(0.0, 1.0 - (self._tail_bloom_until - now) / 0.85)
            bloom = p * p * (3.0 - 2.0 * p)
        else:
            bloom = 1.0
        base = (cx + 54, cy + 50)
        if n == 1:
            angles = [-35.0]
        else:
            angles = [-54.0 + i * 90.0 / (n - 1) for i in range(n)]
        center = -9.0
        length = max(88.0, 124.0 - n * 4.2) * bloom
        width = max(10.5, 27.0 - n * 1.5)
        if self.tail_theme == "cute":
            width *= 1.45
        action = self._tail_action
        if self.tail_theme == "cute":
            dark = _mix(BG, warm, 0.36)
            mid = _mix(warm, cream, 0.42)
            light = _mix(glow, "#FFFFFF", 0.40)
            tips = _mix(cream, "#FFFFFF", 0.28)
        else:
            dark = _mix(BG, "#160B2E", 0.42)
            mid = "#241348"
            light = "#B78CFF"
            tips = "#F3E8FF"
        for i in range(n):
            angle = angles[i]
            style = self.tail_styles[i]
            if action == "fan":
                angle = center + (angle - center) * 1.32
            elif action == "curl":
                angle += 12.0 if angle < center else -12.0
            elif action == "wave":
                angle += math.sin(now * 4.2 + i * 0.95) * 7.0
            elif action == "shiver":
                angle += math.sin(now * 31.0 + i * 1.7) * 4.5
            curve = self.tail_curves[i] * (10.0 if action != "curl" else 16.0)
            if self.tail_theme == "weird":
                col = self.tail_colors[i]
                dark = _mix(BG, col, 0.30)
                mid = _mix(col, "#160B2E", 0.20)
                light = _mix(col, "#FFFFFF", 0.55)
                tips = _mix(col, "#FFFFFF", 0.72)
            self._draw_tail(c, base, angle, length, width, style,
                            self.tail_theme, self.tail_phases[i], curve,
                            now, dark, mid, light, tips)

    def _draw_tail(self, c, base, angle: float, length: float, width: float,
                   style: str, theme: str, phase: float, curve: float,
                   now: float, dark: str, mid: str, light: str, tip: str) -> None:
        rad = math.radians(angle)
        dx, dy = math.cos(rad), math.sin(rad)
        nx, ny = -dy, dx
        bx, by = base
        p0 = (bx, by)
        p1 = (bx + dx * length * 0.30 + nx * curve * 0.45,
              by + dy * length * 0.30 + ny * curve * 0.45)
        p2 = (bx + dx * length * 0.70 + nx * curve,
              by + dy * length * 0.70 + ny * curve)
        p3 = (bx + dx * length, by + dy * length)
        sway = math.sin(now * 2.4 + phase) * 3.2
        samples = []
        m = 15
        for k in range(m):
            t = k / (m - 1)
            u = 1.0 - t
            x = u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0]
            y = u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]
            y += sway * (t ** 1.5)
            samples.append((x, y, t))

        def wid(t: float) -> float:
            base_w = width * (0.58 + 0.62 * math.sin(math.pi * t) ** 0.55) * (1.0 - 0.32 * t) + 1.4
            if style == "ribbon":
                base_w *= 0.55
            return max(1.0, base_w)

        # three overlapping stroke passes: dark rim, colour body, highlight
        for k in range(m - 1):
            x0, y0, t0 = samples[k]
            x1, y1, t1 = samples[k + 1]
            ww = (wid(t0) + wid(t1)) * 0.5
            c.create_line(x0, y0, x1, y1, capstyle="round",
                          fill=dark, width=ww)
        for k in range(m - 1):
            x0, y0, t0 = samples[k]
            x1, y1, t1 = samples[k + 1]
            ww = (wid(t0) + wid(t1)) * 0.5 * 0.64
            c.create_line(x0, y0, x1, y1, capstyle="round",
                          fill=mid, width=ww)
        for k in range(2, m - 1, 2):
            x0, y0, t0 = samples[k]
            x1, y1, t1 = samples[k + 1]
            ww = (wid(t0) + wid(t1)) * 0.5 * 0.20
            c.create_line(x0, y0, x1, y1, capstyle="round",
                          fill=light, width=ww)
        tx, ty, _ = samples[-1]
        if theme == "cute":
            if style == "star":
                self._star(c, tx, ty, 9, tip)
            elif style == "ribbon":
                self._star(c, tx, ty, 7, light)
            elif style == "puff":
                for k in range(2, m - 1, 2):
                    x, y, t = samples[k]
                    r = wid(t) * 0.62
                    c.create_oval(x - r, y - r, x + r, y + r, fill=mid, outline="")
                for ox, oy, r in ((0, 0, 6), (-5, -4, 4), (5, -3, 4)):
                    c.create_oval(tx + ox - r, ty + oy - r, tx + ox + r, ty + oy + r,
                                  fill=tip, outline="")
            else:  # plush
                for ox, oy, r in ((0, 0, 6), (-5, -3, 4), (5, -2, 4)):
                    c.create_oval(tx + ox - r, ty + oy - r, tx + ox + r, ty + oy + r,
                                  fill=tip, outline="")
                c.create_line(tx - 6, ty + 3, tx + 6, ty + 3, fill=light, width=1)
        # weird decorations
        if style == "eye":
            for frac, er in ((0.72, 7.0), (1.0, 12.0)):
                idx = min(int(frac * (m - 1)), m - 1)
                ex, ey, _ = samples[idx]
                c.create_oval(ex - er, ey - er, ex + er, ey + er,
                              fill="#F4EEFF", outline="#1A0C2A", width=1)
                c.create_oval(ex - er * 0.66, ey - er * 0.66,
                              ex + er * 0.66, ey + er * 0.66,
                              fill=light, outline="")
                c.create_oval(ex - er * 0.24, ey - er * 0.24,
                              ex + er * 0.24, ey + er * 0.24,
                              fill="#140A20", outline="")
                c.create_line(ex - er * 0.55, ey - er * 0.55,
                              ex + er * 0.15, ey - er * 0.15,
                              fill="#FFFFFF", width=1)
        elif style == "spike":
            for k in (0.45, 0.62, 0.78):
                idx = int(k * (m - 1))
                x, y, _ = samples[idx]
                c.create_polygon([(x - 3, y), (x + 2, y - 9), (x + 6, y - 1)],
                                 fill=light, outline="")
            c.create_polygon([(tx - 5, ty - 2), (tx + 1, ty - 14), (tx + 6, ty - 2),
                              (tx + 1, ty + 3)], fill=tip, outline="#140A20")
        elif style == "tentacle":
            for k in (0.55, 0.72, 0.87):
                idx = int(k * (m - 1))
                x, y, _ = samples[idx]
                c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5,
                              fill=light, outline="#2B1240")
            c.create_oval(tx - 4, ty - 4, tx + 4, ty + 4, fill=tip, outline="#2B1240")
        elif style == "segmented":
            for k in range(2, m - 2, 3):
                x, y, t = samples[k]
                r = wid(t) * 0.7
                c.create_oval(x - r, y - r, x + r, y + r, fill=dark, outline="")
            c.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, fill=light, outline="")
            c.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill="#160B2E", outline="")
        elif style == "mouth":
            c.create_oval(tx - 14, ty - 8, tx + 14, ty + 11, fill="#2A0C18", outline="")
            for ox in (-10, -5, 0, 5, 10):
                c.create_polygon([(tx + ox - 3, ty - 7), (tx + ox, ty - 2),
                                  (tx + ox + 3, ty - 7)], fill="#F8F0FF", outline="")
            c.create_oval(tx - 4, ty + 3, tx + 4, ty + 10, fill="#FF6B8A", outline="")
        elif style == "crystal":
            c.create_polygon([(tx, ty - 16), (tx + 11, ty - 3), (tx + 5, ty + 11),
                              (tx - 7, ty + 11), (tx - 12, ty - 2)],
                             fill=tip, outline="#2A1050")
            c.create_line(tx, ty - 12, tx - 2, ty + 8, fill=light, width=1)
        if theme == "weird":
            # extra little eyes gazing out from the tail itself
            for frac in (0.34, 0.54):
                idx = min(int(frac * (m - 1)), m - 1)
                ex, ey, _ = samples[idx]
                c.create_oval(ex - 5, ey - 5, ex + 5, ey + 5, fill="#F4EEFF",
                              outline="#140A20")
                c.create_oval(ex - 3, ey - 3, ex + 3, ey + 3, fill=light, outline="")
                c.create_oval(ex - 1.2, ey - 1.2, ex + 1.2, ey + 1.2,
                              fill="#140A20", outline="")
        if theme == "cute" and style in ("plush", "puff"):
            self._bow(c, tx, ty - 8, 7)
        # a soft dot of light at the base for all tails
        c.create_oval(bx - 4, by - 4, bx + 4, by + 4, fill=light, outline="")

    @staticmethod
    def _star(c, x: float, y: float, r: float, color: str) -> None:
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.42
            pts.extend((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
        c.create_polygon(pts, fill=color, outline="")

    @staticmethod
    def _heart(c, x: float, y: float, r: float, color: str) -> None:
        c.create_oval(x - r, y - r * 0.80, x, y + r * 0.30, fill=color, outline="")
        c.create_oval(x, y - r * 0.80, x + r, y + r * 0.30, fill=color, outline="")
        c.create_polygon(x - r * 0.92, y - r * 0.10, x + r * 0.92, y - r * 0.10,
                         x, y + r * 1.20, fill=color, outline="")

    def _bow(self, c, x: float, y: float, r: float) -> None:
        pink = "#FF8FB8" if self.tail_theme == "cute" else "#FF2DB3"
        c.create_polygon(x - r, y - r, x - r * 0.15, y, x - r, y + r,
                         fill=pink, outline="#8A2450")
        c.create_polygon(x + r, y - r, x + r * 0.15, y, x + r, y + r,
                         fill=pink, outline="#8A2450")
        c.create_oval(x - r * 0.28, y - r * 0.28, x + r * 0.28, y + r * 0.28,
                      fill="#FFE3B0", outline="#8A2450")

    def _theme_aura(self, c, cx: float, cy: float, now: float) -> None:
        if self.tail_theme == "cute":
            for i in range(10):
                ang = now * 0.35 + i * math.tau / 10
                r = 124 + 14 * math.sin(now * 0.8 + i)
                x = cx + math.cos(ang) * r
                y = cy - 10 + math.sin(ang) * r * 0.78
                if i % 2:
                    self._heart(c, x, y, 5.0, "#FF8FB8")
                else:
                    self._star(c, x, y, 5.5, "#FFE08A")
        else:
            for i in range(6):
                ang = -now * 0.22 + i * math.tau / 6
                r = 126 + 10 * math.sin(now + i)
                x = cx + math.cos(ang) * r
                y = cy - 8 + math.sin(ang) * r * 0.75
                c.create_oval(x - 7, y - 7, x + 7, y + 7,
                              fill="#F4EEFF", outline="#1A0C2A")
                c.create_oval(x - 4.5, y - 4.5, x + 4.5, y + 4.5,
                              fill=["#FF2DB3", "#00E5FF", "#B6FF3B"][i % 3], outline="")
                c.create_oval(x - 1.6, y - 1.6, x + 1.6, y + 1.6,
                              fill="#140A20", outline="")
            # little floating teeth / orbs
            for i in range(4):
                x = cx + 118 * math.cos(now * 0.3 + i * 1.6)
                y = cy + 74 + 22 * math.sin(now * 0.5 + i)
                c.create_polygon([(x - 5, y), (x, y - 6), (x + 5, y)],
                                 fill="#F8F0FF", outline="")

    # ------------------------------------------------------------------ drawing
    def _draw(self, beat: float = 0.0) -> None:
        c = _ScaledCanvas(self.canvas, self.SCALE)
        c.delete("all")
        cx, cy = 125, 150
        now = time.monotonic()
        wake = max(0.0, self._wake_until - now)
        awake = 1.0 if wake > 0 else 0.0
        wake_phase = (2.6 - wake) if wake > 0 else 0.0
        flip = now < self._flip_until
        color = self._flip_color if flip else None
        bounce = -3.0 * self.level - 2.0 * beat * self.level - 5.0 * awake
        warm = _mix(AMBER, CORAL, max(0.0, 1.0 - self.valence))
        cool = _mix(AMBER, CYAN, max(0.0, self.valence - 0.55) * 1.2)
        if color:
            warm = _mix(color, "#FFFFFF", 0.20)
            cool = _mix(color, "#FFFFFF", 0.42)
            cream = _mix(color, "#FFFFFF", 0.70)
            ink = _mix(INK, color, 0.30)
            glow = _mix(color, "#FFFFFF", 0.55)
        else:
            cream = CREAM
            ink = INK
            glow = _mix(warm, "#FFFFFF", 0.25 + 0.25 * self.level + 0.25 * awake)
        if awake:
            glow = _mix(glow, "#FFFFFF", 0.35)

        # halo / light spill
        for i, (rx, ry, t) in enumerate(((104, 98, 0.10), (86, 82, 0.15),
                                         (68, 66, 0.20))):
            c.create_oval(cx - rx, cy - ry + 12 + bounce, cx + rx, cy + ry + 12 + bounce,
                          outline=_mix(BG, glow, t), width=2 + i)

        # soft layered shadow
        for rx, ry, shade in ((66, 15, 0.24), (56, 12, 0.38), (44, 9, 0.55)):
            c.create_oval(cx - rx, cy + 92 - ry, cx + rx, cy + 92 + ry,
                          fill=_mix(BG, "#000000", shade), outline="")

        # random 1–9 fox / eldritch tails behind the body
        try:
            self._draw_tails(c, cx, cy, warm, cream, glow, now)
        except Exception:  # noqa: BLE001 - tails must never kill the pet
            pass

        # ---------------------------------------------------------- body
        body = [cx - 48, cy + 18, cx - 58, cy + 60, cx - 38, cy + 90,
                cx + 38, cy + 90, cx + 58, cy + 60, cx + 48, cy + 18,
                cx + 28, cy - 2, cx - 28, cy - 2]
        c.create_polygon(body, smooth=True, splinesteps=24,
                         fill=_mix(BG, warm, 0.50), outline="")
        c.create_oval(cx - 41, cy + 16, cx + 41, cy + 78,
                      fill=_mix(warm, cream, 0.42), outline="")
        c.create_oval(cx - 26, cy + 31, cx + 26, cy + 73,
                      fill=_mix(cream, "#FFFFFF", 0.30), outline="")
        c.create_arc(cx - 44, cy + 18, cx + 44, cy + 82, start=210, extent=100,
                     style="arc", outline=_mix("#FFFFFF", glow, 0.62), width=2)
        c.create_oval(cx - 50, cy + 74, cx + 50, cy + 96,
                      fill=_mix(BG, warm, 0.14), outline="")

        # paws
        for dx in (-32, 32):
            c.create_oval(cx + dx - 15, cy + 72, cx + dx + 15, cy + 91,
                          fill=_mix(warm, cream, 0.30), outline="")
            for toe in (-6, 0, 6):
                c.create_line(cx + dx + toe, cy + 79, cx + dx + toe, cy + 87,
                              fill=_mix(ink, warm, 0.5), width=1)

        # ears
        ear_y = cy - 66 + bounce - 6.0 * awake
        for side in (-1, 1):
            bx = cx + side * 33
            ear = [bx - side * 17, ear_y + 20, bx + side * 4, ear_y - 25,
                   bx + side * 22, ear_y + 18]
            c.create_polygon(ear, fill=_mix(BG, warm, 0.46), outline="")
            c.create_polygon([bx - side * 10, ear_y + 13,
                              bx + side * 4, ear_y - 16,
                              bx + side * 13, ear_y + 12],
                             fill=_mix(PINK, cream, 0.4), outline="")

        # head
        hy = cy - 28 + bounce
        c.create_oval(cx - 57, hy - 45, cx + 57, hy + 47,
                      fill=_mix(warm, cream, 0.38), outline="")
        c.create_oval(cx - 51, hy - 39, cx + 51, hy + 41,
                      fill=_mix(cream, "#FFF7E4", 0.55), outline="")
        c.create_arc(cx - 52, hy - 40, cx + 52, hy + 42, start=25, extent=115,
                     style="arc", outline=_mix("#FFFFFF", glow, 0.25), width=2)

        # forehead star
        sx, sy = cx, hy - 32
        star = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = 8 if i % 2 == 0 else 3.5
            star.extend((sx + math.cos(ang) * r, sy + math.sin(ang) * r))
        c.create_polygon(star, fill=_mix(glow, "#FFFFFF", 0.5), outline="")

        # ---------------------------------------------------------- face
        expression = "wake" if awake else self.expression
        eye_y = hy + 2
        eye_dx = 25
        if expression == "disdain":
            c.create_line(cx - 37, hy - 15, cx - 17, hy - 21, fill=ink, width=2)
            c.create_line(cx + 17, hy - 19, cx + 37, hy - 10, fill=ink, width=2)
        elif expression == "funny":
            c.create_line(cx - 37, hy - 17, cx - 18, hy - 11, fill=ink, width=2)

        self._blink += 0.033
        blinking = 2.9 < (self._blink % 3.4) < 3.08 and expression in ("normal",)
        for side in (-1, 1):
            ex = cx + side * eye_dx
            if expression == "happy":
                c.create_arc(ex - 13, eye_y - 8, ex + 13, eye_y + 10,
                             start=200, extent=140, style="arc", outline=ink, width=3)
                continue
            if expression == "funny" and side == -1:
                c.create_line(ex - 12, eye_y + 2, ex + 12, eye_y + 2,
                              fill=ink, width=3, capstyle="round")
                continue
            if blinking:
                c.create_line(ex - 10, eye_y + 2, ex + 10, eye_y + 2,
                              fill=ink, width=3, capstyle="round")
                continue
            rr = 16 + 2 * awake
            c.create_oval(ex - rr, eye_y - rr, ex + rr, eye_y + rr,
                          fill=ink, outline="")
            c.create_oval(ex - rr + 3, eye_y - rr + 3, ex + rr - 3, eye_y + rr - 3,
                          fill=_mix(IRIS_COLORS[0 if side < 0 else 1], "#FFFFFF", 0.10),
                          outline="")
            c.create_oval(ex - 6, eye_y - 6, ex + 6, eye_y + 6, fill=ink, outline="")
            c.create_oval(ex - 2, eye_y - 5, ex + 3, eye_y, fill="#FFFFFF", outline="")
            if awake:
                c.create_line(ex - 7, eye_y, ex + 7, eye_y, fill="#FFF6D2", width=1)
                c.create_line(ex, eye_y - 7, ex, eye_y + 7, fill="#FFF6D2", width=1)
            if expression == "funny" and side == 1:
                c.create_oval(ex - 5, eye_y + 6, ex + 5, eye_y + 13,
                              fill=_mix(PINK, "#FFFFFF", 0.2), outline="")
        if expression == "disdain":
            for side in (-1, 1):
                ex = cx + side * eye_dx
                c.create_rectangle(ex - 20, eye_y - 19, ex + 20, eye_y - 3,
                                   fill=_mix(cream, "#FFFFFF", 0.55), outline="")
                c.create_line(ex - 18, eye_y - 3, ex + 18, eye_y - 3,
                              fill=ink, width=2)
            c.create_line(cx - 33, hy - 15, cx - 16, hy - 9, fill=ink, width=2)
            c.create_line(cx + 16, hy - 12, cx + 33, hy - 17, fill=ink, width=2)

        # blush
        for side in (-1, 1):
            c.create_oval(cx + side * 34 - 10, hy + 19, cx + side * 34 + 10, hy + 27,
                          fill=_mix(PINK, cream, 0.25), outline="")
        if expression == "happy":
            c.create_arc(cx - 11, hy + 14, cx + 11, hy + 30, start=200, extent=140,
                         style="arc", outline=ink, width=3)
        elif expression == "funny":
            c.create_oval(cx - 7, hy + 19, cx + 7, hy + 32, fill=ink, outline="")
            c.create_polygon(cx - 4, hy + 30, cx + 2, hy + 30, cx + 3, hy + 41,
                             cx - 1, hy + 39, fill="#FF7E8E", outline="")
        elif expression == "disdain":
            c.create_line(cx - 8, hy + 25, cx + 8, hy + 21, fill=ink, width=2)
        elif expression == "wake":
            c.create_oval(cx - 5, hy + 22, cx + 5, hy + 32,
                          fill=_mix(ink, "#FFFFFF", 0.1), outline="")
        else:
            c.create_arc(cx - 6, hy + 18, cx + 6, hy + 29, start=200, extent=140,
                         style="arc", outline=ink, width=2)
        for side in (-1, 1):
            for dy in (-2, 4):
                c.create_line(cx + side * 38, hy + 22 + dy,
                              cx + side * 58, hy + 18 + dy,
                              fill=_mix(ink, warm, 0.45), width=1)

        # sparkles
        for i in range(5):
            ang = self._t * 0.5 + i * math.tau / 5
            r = 92 + 8 * math.sin(self._t * 1.2 + i)
            px = cx + math.cos(ang) * r
            py = cy - 24 + math.sin(ang) * (r * 0.72)
            s = 3 + 2 * (0.5 + 0.5 * math.sin(self._t * 2 + i))
            c.create_line(px - s, py, px + s, py, fill=_mix(glow, "#FFFFFF", 0.45), width=1)
            c.create_line(px, py - s, px, py + s, fill=_mix(glow, "#FFFFFF", 0.45), width=1)

        self._theme_aura(c, cx, cy, now)

        # wake-word speech bubble
        if awake:
            bx0, by0, bx1, by1 = cx - 68, 7, cx + 68, 50
            c.create_oval(bx0, by0, bx1, by1, fill=cream,
                          outline=_mix(AMBER, ink, 0.35), width=1)
            c.create_polygon(cx - 9, by1 - 1, cx + 1, by1 + 11, cx + 9, by1 - 1,
                             fill=cream, outline="")
            c.create_text(cx, (by0 + by1) / 2, text=self._wake_text,
                          fill=ink, font=("Microsoft YaHei UI", 15, "bold"))
            for i in range(6):
                ax = cx + 78 * math.cos(i * math.tau / 6 + wake_phase * 2)
                ay = 28 + 18 * math.sin(i * math.tau / 6 + wake_phase * 2)
                c.create_line(ax - 3, ay, ax + 3, ay,
                              fill=_mix(AMBER, "#FFFFFF", 0.5), width=1)

        # hover hint
        if self._hover:
            c.create_text(cx, 258,
                          text=f"尾巴 {self.tail_count} · 单击面板 · 双击切歌 · 长按暂停",
                          fill="#D8CBB4", font=("Microsoft YaHei UI", 11))

    def close(self) -> None:
        self._closed = True
        try:
            self.win.destroy()
        except tk.TclError:
            pass
