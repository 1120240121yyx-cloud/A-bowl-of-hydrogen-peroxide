"""Local-only light mask for 小茂模式 (Xiaomao mode).

When enabled, the full-screen dim overlay is switched off and this small
transparent disc around the pet becomes the only mask.  Its darkness follows
the same music level, so the rest of the desktop stays untouched.
"""
from __future__ import annotations

import ctypes
import tkinter as tk

BG = "#010203"
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class PetAura:
    def __init__(self, root: tk.Misc, pet) -> None:
        self.pet = pet
        self.root = root
        self.w = max(230, int(pet.W * 1.35))
        self.h = max(210, int(pet.H * 1.25))
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=BG)
        try:
            self.win.attributes("-transparentcolor", BG)
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(self.win, width=self.w, height=self.h, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self._clickthrough()
        self.win.withdraw()
        self.enabled = False
        self._palette = ("#F2A33A",)
        self._valence = 0.5
        self._last = (-1, -1)

    def _clickthrough(self) -> None:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.win.winfo_id())
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:  # noqa: BLE001
            pass

    def follow(self) -> None:
        try:
            px = self.pet.win.winfo_x()
            py = self.pet.win.winfo_y()
        except tk.TclError:
            return
        x = px + self.pet.W // 2 - self.w // 2
        y = py + self.pet.H // 2 - self.h // 2
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = max(0, min(sw - self.w, x))
        y = max(0, min(sh - self.h, y))
        if (x, y) != self._last:
            self._last = (x, y)
            try:
                self.win.geometry(f"{self.w}x{self.h}+{x}+{y}")
            except tk.TclError:
                pass

    def show(self, level: float, palette, valence: float = 0.5) -> None:
        self.enabled = True
        self.follow()
        self._palette = palette or ("#F2A33A",)
        self._valence = valence
        # darker when quiet, almost clear when loud — same direction as the
        # normal dim mask, but only inside this disc.
        alpha = _clamp(0.06 + 0.56 * (1.0 - level), 0.03, 0.72)
        self.canvas.delete("all")
        edge = self._palette[0] if self._palette else "#F2A33A"
        self.canvas.create_oval(2, 2, self.w - 2, self.h - 2,
                                fill="#000000", outline="")
        # soft warm/cool rim
        self.canvas.create_oval(3, 3, self.w - 3, self.h - 3,
                                outline=edge, width=2)
        self.canvas.create_oval(12, 12, self.w - 12, self.h - 12,
                                outline="#FFFFFF" if self._valence >= 0.5 else "#DFF4FF",
                                width=1, dash=(2, 6))
        try:
            self.win.attributes("-alpha", alpha)
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.win.lift()
        except tk.TclError:
            pass

    def hide(self) -> None:
        self.enabled = False
        try:
            self.win.withdraw()
        except tk.TclError:
            pass

    def close(self) -> None:
        self.hide()
        try:
            self.win.destroy()
        except tk.TclError:
            pass
