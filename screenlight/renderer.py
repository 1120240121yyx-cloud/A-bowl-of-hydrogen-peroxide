"""Tk full-screen overlay renderer (dim / canvas / dj / wake).

Canvas mode draws through :class:`screenlight.render.CanvasScene` (gradient
backdrop, glow, capsule spectrum bars, VU peak caps and expanding beat rings).
DJ mode draws through :class:`screenlight.render.DJScene` (two premium canvases,
keyboard/touch instrument with chain reactions, swipe to morph the canvas) and
makes the overlay interactive (mouse / touch) while active.
If a scene fails to import or raises while drawing, the overlay degrades to the
original flat-rectangle drawing so the demo never crashes.
"""
from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont

import numpy as np

from .mapping import mix, palette_color, scale_color

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000


class Overlay:
    def __init__(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        if width <= 0 or height <= 0:
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
        self.width = width
        self.height = height

        self.win = tk.Toplevel(self.root)
        self.win.title("OpenRay Overlay")
        self.win.overrideredirect(True)
        self.win.geometry(f"{width}x{height}+{x}+{y}")
        self.win.configure(bg="black")
        self.win.attributes("-topmost", True)
        self._suppressed = False

        self.canvas = tk.Canvas(
            self.win, bg="black", highlightthickness=0, bd=0, width=width, height=height
        )
        self.canvas.pack(fill="both", expand=True)

        # full-screen background: its luminance follows the beat so the
        # canvas / wake modes pulse just like the dim overlay does
        self.bg_item = self.canvas.create_rectangle(
            0, 0, width, height, fill="#000000", outline=""
        )

        self.n_bars = 28
        self.bars: list[int] = []
        self.bar_w = max(6, int(width / (self.n_bars * 1.9)))
        for i in range(self.n_bars):
            x0 = int((i + 0.5) * width / self.n_bars)
            self.bars.append(
                self.canvas.create_rectangle(
                    x0 - self.bar_w // 2, height, x0 + self.bar_w // 2, height,
                    fill="#000000", outline="",
                )
            )

        self.text_font = tkfont.Font(family="Microsoft YaHei", size=-int(height * 0.075), weight="bold")
        self.sub_font = tkfont.Font(family="Microsoft YaHei", size=-int(height * 0.022))
        self.text_shadow = self.canvas.create_text(
            width // 2 + 4, height // 2 + 4, text="", fill="#000000", font=self.text_font
        )
        self.sub_shadow = self.canvas.create_text(
            width // 2 + 2, int(height * 0.68) + 2, text="", fill="#000000", font=self.sub_font
        )
        self.text_item = self.canvas.create_text(
            width // 2, height // 2, text="", fill="#FFFFFF", font=self.text_font
        )
        self.sub_item = self.canvas.create_text(
            width // 2, int(height * 0.68), text="", fill="#9E9E9E", font=self.sub_font
        )
        self.badge = self.canvas.create_text(
            int(width * 0.5), int(height * 0.08), text="", fill="#666666",
            font=tkfont.Font(family="Microsoft YaHei", size=-int(height * 0.016)),
        )

        # fancy canvas-mode scene; any failure at all is non-fatal
        self.scene = None
        try:
            from .render import CanvasScene

            self.scene = CanvasScene(self.canvas, width, height, n_bars=self.n_bars)
            self.scene.set_visible(False)
        except Exception:  # pragma: no cover - legacy fallback path
            self.scene = None
            try:  # remove any half-built scene items (they are all tagged)
                self.canvas.delete("sl_scene")
            except tk.TclError:
                pass

        # DJ scene: two premium full-screen canvases + input instrument
        self.dj = None
        try:
            from .render import DJScene

            self.dj = DJScene(self.canvas, width, height)
            self.dj.set_visible(False)
        except Exception:  # pragma: no cover - legacy fallback path
            self.dj = None
        self.dj_action_callback = None
        self._dj_down = None
        self._interactive = False
        self.canvas.bind("<ButtonPress-1>", self._dj_press)
        self.canvas.bind("<B1-Motion>", self._dj_drag)
        self.canvas.bind("<ButtonRelease-1>", self._dj_release)
        # top-most flash surface used by the optional burst module
        self.burst_item = self.canvas.create_rectangle(
            0, 0, self.width, self.height, fill="#000000", outline="",
            state="hidden")
        self.canvas.bind("<MouseWheel>", self._dj_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._dj_wheel_h)

        self.win.update_idletasks()
        self.win.deiconify()
        self.win.attributes("-alpha", 0.0)
        self._click_through()

        self.mode = "dim"
        self._last_alpha = -1.0
        self._layered = True

    # ------------------------------------------------------------ windows
    def _click_through(self) -> None:
        if sys.platform != "win32":
            return
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        )
        self.hwnd = hwnd

    def _set_layered(self, on: bool) -> None:
        """Opaque modes (canvas / dj) drop WS_EX_LAYERED: per-pixel alpha
        composition of a full-screen window is a large part of the frame cost."""
        if sys.platform != "win32" or getattr(self, "_layered", True) == on:
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = getattr(self, "hwnd", None)
            if not hwnd:
                return
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if on:
                style |= WS_EX_LAYERED
            else:
                style &= ~WS_EX_LAYERED
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            self._layered = on
            self._last_alpha = -1.0
            if on:
                self._set_alpha(1.0)
        except Exception:  # noqa: BLE001
            pass

    def set_interactive(self, enabled: bool) -> None:
        """DJ mode needs mouse/touch *and* keyboard focus on the overlay.

        Click-through + WS_EX_NOACTIVATE are both removed so the window can be
        hit-tested and take keyboard focus; other modes restore both.
        """
        self._interactive = bool(enabled)
        if sys.platform != "win32":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = getattr(self, "hwnd", None)
            if not hwnd:
                return
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if enabled:
                style &= ~WS_EX_TRANSPARENT
                style &= ~WS_EX_NOACTIVATE
            else:
                style |= WS_EX_TRANSPARENT
                style |= WS_EX_NOACTIVATE
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            if enabled:
                self.focus_dj()
            else:
                user32.SetForegroundWindow(hwnd)
        except Exception:  # noqa: BLE001 - input is best effort
            pass

    def focus_dj(self) -> None:
        """Give the overlay keyboard focus (DJ mode only)."""
        if self.mode != "dj":
            return
        try:
            self.win.lift()
            self.win.focus_force()
        except tk.TclError:
            pass

    def _dj_wheel(self, event) -> None:  # noqa: ANN001
        if self.mode != "dj":
            return
        self.dj_swipe(0, 200 if event.delta > 0 else -200)

    def _dj_wheel_h(self, event) -> None:  # noqa: ANN001
        if self.mode != "dj":
            return
        self.dj_swipe(200 if event.delta > 0 else -200, 0)

    def render_burst(self, white: bool) -> None:
        """Full-screen pure white / pure black flash (optional module)."""
        self._set_alpha(1.0)
        self.canvas.itemconfig(self.burst_item, state="normal",
                               fill="#FFFFFF" if white else "#000000")

    def hide_burst(self) -> None:
        try:
            self.canvas.itemconfig(self.burst_item, state="hidden")
        except tk.TclError:
            pass

    # ------------------------------------------------------- dj input
    def _dj_press(self, event) -> None:  # noqa: ANN001
        if self.mode != "dj" or self.dj is None:
            return
        try:
            self._dj_down = (event.x, event.y)
            self.focus_dj()          # keyboard follows the click/touch
            self.dj.pointer_press(event.x, event.y)
        except Exception:  # noqa: BLE001
            pass

    def _dj_drag(self, event) -> None:  # noqa: ANN001
        if self.mode != "dj" or self.dj is None or self._dj_down is None:
            return
        try:
            self.dj.pointer_move(event.x, event.y)
        except Exception:  # noqa: BLE001
            pass

    def _dj_release(self, event) -> None:  # noqa: ANN001
        if self.mode != "dj" or self.dj is None or self._dj_down is None:
            return
        self._dj_down = None
        try:
            action = self.dj.pointer_release(event.x, event.y)
            if self.dj_action_callback is not None:
                self.dj_action_callback(action, self.dj.status())
        except Exception:  # noqa: BLE001
            pass

    def dj_key(self, index: int) -> None:
        if self.mode != "dj" or self.dj is None:
            return
        try:
            self.dj.press_index(index)
        except Exception:  # noqa: BLE001
            pass

    def dj_swipe(self, dx: float, dy: float) -> None:
        if self.mode != "dj" or self.dj is None:
            return
        try:
            action = self.dj.swipe(dx, dy)
            if self.dj_action_callback is not None:
                self.dj_action_callback(action, self.dj.status())
        except Exception:  # noqa: BLE001
            pass

    def dj_status(self) -> str:
        if self.dj is None:
            return "DJ 不可用"
        try:
            return self.dj.status()
        except Exception:  # noqa: BLE001
            return "DJ"

    def _set_alpha(self, value: float) -> None:
        value = min(max(value, 0.0), 1.0)
        if not getattr(self, "_layered", True):
            self._last_alpha = 1.0
            return
        if abs(value - self._last_alpha) < 0.002:
            return
        self._last_alpha = value
        try:
            self.win.attributes("-alpha", value)
        except tk.TclError:
            pass

    # ------------------------------------------------------- mode / scenes
    def _apply_mode(self, mode: str) -> None:
        """Switch item visibility for a new mode, in one place."""
        c = self.canvas
        hidden, normal = "hidden", "normal"
        for item in (self.bg_item, *self.bars, self.badge):
            c.itemconfig(item, state=hidden)
        for item in (self.text_item, self.sub_item, self.text_shadow, self.sub_shadow):
            c.itemconfig(item, state=hidden)
        if self.scene is not None:
            try:
                self.scene.set_visible(False)
            except Exception:  # pragma: no cover - defensive
                self.scene = None
        if self.dj is not None:
            try:
                self.dj.set_visible(False)
            except Exception:  # pragma: no cover - defensive
                self.dj = None
        if mode == "dim":
            self._set_layered(True)
            return
        if mode in ("canvas", "dj"):
            self._set_layered(False)
        if mode == "dj":
            if self.dj is not None:
                try:
                    self.dj.set_visible(True)
                    return
                except Exception:  # pragma: no cover - defensive
                    self.dj = None
            self._legacy_canvas_visible()
            return
        if mode == "canvas":
            if self.scene is not None:
                try:
                    self.scene.set_visible(True)
                    return
                except Exception:  # pragma: no cover - defensive
                    self.scene = None
            self._legacy_canvas_visible()
            return
        # wake: legacy background + big text
        self._set_layered(True)
        c.itemconfig(self.bg_item, state=normal)
        for item in (self.text_item, self.sub_item, self.text_shadow, self.sub_shadow):
            c.itemconfig(item, state=normal)

    def _drop_scene(self) -> None:
        scene = self.scene
        self.scene = None
        if scene is not None:
            try:
                scene.set_visible(False)
            except Exception:  # pragma: no cover - defensive
                pass
        try:  # scenes tag all their items; remove them so legacy drawing is clean
            self.canvas.delete("sl_scene")
        except tk.TclError:
            pass

    def _legacy_canvas_visible(self) -> None:
        normal = "normal"
        self.canvas.itemconfig(self.bg_item, state=normal)
        for item in self.bars:
            self.canvas.itemconfig(item, state=normal)

    # ------------------------------------------------------------ render
    def render(self, level: float, bands: np.ndarray | None, mode: str,
               palette: list[str], text: str, max_dim: float,
               hint: str = "", accent: float = 0.0,
               beat_phase: float = 0.0) -> None:
        level = min(max(level, 0.0), 1.0)
        if mode != self.mode:
            self.mode = mode
            self._apply_mode(mode)

        if mode == "dim":
            # keep the dim alpha law exactly as before: (1 - level) * max_dim
            self._set_alpha((1.0 - level) * max_dim)
            return

        self._set_alpha(1.0)
        if mode == "canvas":
            self._render_canvas(level, bands, palette, hint, accent, beat_phase)
        elif mode == "dj":
            self._render_dj(level, bands, palette, hint, accent, beat_phase)
        else:  # wake
            self._render_wake(level, palette, text, hint)

    def _render_canvas(self, level: float, bands: np.ndarray | None,
                       palette: list[str], hint: str, accent: float,
                       beat_phase: float) -> None:
        scene = self.scene
        if scene is not None:
            try:
                scene.update(level, bands, palette, hint,
                             accent=accent, beat_phase=beat_phase)
                return
            except Exception:  # pragma: no cover - never crash the demo
                self._drop_scene()
                self._legacy_canvas_visible()
        self._legacy_canvas(level, bands, palette, hint)

    def _render_dj(self, level: float, bands: np.ndarray | None,
                   palette: list[str], hint: str, accent: float,
                   beat_phase: float) -> None:
        scene = self.dj
        if scene is not None:
            try:
                scene.update(level, bands, palette, hint,
                             accent=accent, beat_phase=beat_phase)
                return
            except Exception:  # pragma: no cover - never crash the demo
                self.dj = None
                try:
                    self.canvas.delete("sl_dj")
                except tk.TclError:
                    pass
        self._legacy_canvas_visible()
        self._legacy_canvas(level, bands, palette, hint)

    def _legacy_canvas(self, level: float, bands: np.ndarray | None,
                       palette: list[str], hint: str) -> None:
        """Original flat-rectangle canvas mode, kept as a safe fallback."""
        base_color = palette[0] if palette else "#FF7A18"
        wash = mix(base_color, "#FFFFFF", 0.30)
        self.canvas.itemconfig(
            self.bg_item, fill=mix("#000000", wash, 0.02 + 0.60 * level)
        )
        if bands is None:
            bands = np.zeros(self.n_bars, dtype=np.float32)
        for i in range(self.n_bars):
            v = float(bands[min(i * len(bands) // self.n_bars, len(bands) - 1)]) if len(bands) else 0.0
            h = max(3.0, v * self.height * 0.52 * (0.35 + 0.65 * level))
            x0 = int((i + 0.5) * self.width / self.n_bars) - self.bar_w // 2
            x1 = x0 + self.bar_w
            y1 = self.height - h
            color = palette_color(palette, i / max(self.n_bars - 1, 1))
            color = scale_color(color, 0.45 + 0.55 * level)
            self.canvas.coords(self.bars[i], x0, y1, x1, self.height)
            self.canvas.itemconfig(self.bars[i], fill=color)
        self.canvas.itemconfig(
            self.badge, text=hint,
            fill="#3A2A18" if level > 0.45 else "#8A8A90",
            state="normal" if hint else "hidden",
        )

    def _render_wake(self, level: float, palette: list[str], text: str,
                     hint: str) -> None:
        base_color = palette[0] if palette else "#FF7A18"
        wash = mix(base_color, "#FFFFFF", 0.30)
        self.canvas.itemconfig(
            self.bg_item, fill=mix("#000000", wash, 0.02 + 0.60 * level)
        )
        tx, ty = self.width // 2, int(self.height * 0.48)
        self.canvas.itemconfig(self.text_shadow, text=text)
        self.canvas.coords(self.text_shadow, tx + 4, ty + 4)
        self.canvas.itemconfig(
            self.text_item, text=text, fill=scale_color("#FFFFFF", max(level, 0.55))
        )
        self.canvas.coords(self.text_item, tx, ty)
        self.canvas.itemconfig(self.sub_shadow, text=hint)
        self.canvas.coords(self.sub_shadow, tx + 2, int(self.height * 0.68) + 2)
        self.canvas.itemconfig(
            self.sub_item, text=hint, fill=scale_color("#EDEDED", max(level, 0.5))
        )
        self.canvas.itemconfig(self.badge, text="", state="hidden")

    def ensure_topmost(self) -> None:
        """Keep the base dim mask visible + topmost unless explicitly hidden.

        ``attributes(-topmost)`` alone does not re-raise above another topmost
        window, so issue one SetWindowPos(TOPMOST) as well.  The control panel
        (and the pet) are raised immediately after this in the app tick.
        """
        if getattr(self, "_suppressed", False):
            return
        try:
            if not self.win.winfo_viewable():
                self.win.deiconify()
            if not bool(self.win.attributes("-topmost")):
                self.win.attributes("-topmost", True)
            self.win.lift()
            if sys.platform == "win32":
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(self.win.winfo_id())
                ctypes.windll.user32.SetWindowPos(
                    hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:  # noqa: BLE001
            pass

    def set_suppressed(self, on: bool) -> None:
        """Hide the normal overlay while the spatial mask owns the screen."""
        if on == getattr(self, "_suppressed", False):
            return
        self._suppressed = on
        try:
            if on:
                self.win.withdraw()
            else:
                self.win.deiconify()
                self.win.attributes("-topmost", True)
                self.win.lift()
        except tk.TclError:
            pass

    def lift_control(self, control: tk.Toplevel) -> None:
        try:
            control.lift()
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
