"""Global hotkeys (Win32 RegisterHotKey) — work even when the app is not focused."""
from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

VK = {
    "Q": 0x51, "D": 0x44, "H": 0x48, "S": 0x53,
    "1": 0x31, "2": 0x32, "3": 0x33, "SPACE": 0x20,
    "F5": 0x74, "F6": 0x75, "F7": 0x76, "F9": 0x78, "F10": 0x79, "F11": 0x7A,
}


def combo_name(mods: int, vk: int) -> str:
    parts = []
    if mods & MOD_CONTROL:
        parts.append("Ctrl")
    if mods & MOD_ALT:
        parts.append("Alt")
    if mods & MOD_SHIFT:
        parts.append("Shift")
    inv = {v: k for k, v in VK.items()}
    parts.append(inv.get(vk, f"VK{vk:02X}"))
    return "+".join(parts)


class Hotkeys:
    """Register Ctrl+Alt+* hotkeys on a dedicated Win32 message thread."""

    def __init__(self, specs: list[tuple[int, int, str]]) -> None:
        self._specs = specs
        self._queue: queue.Queue[str] = queue.Queue()
        self._tid: int | None = None
        self.registered: list[tuple[str, str]] = []
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(0.5)

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._tid = int(kernel32.GetCurrentThreadId())
        registered: dict[int, tuple[str, str]] = {}
        for i, (mods, vk, name) in enumerate(self._specs, start=1):
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                registered[i] = (name, combo_name(mods, vk))
        self.registered = list(registered.values())
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                item = registered.get(int(msg.wParam))
                if item:
                    self._queue.put(item[0])
        for i in registered:
            user32.UnregisterHotKey(None, i)

    def poll(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
