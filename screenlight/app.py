"""OpenRay v2.4 controller / UI."""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np

from . import __version__
from .agent import apply_text, describe
from .audio import InAppSource, build_sources
from .config import ASSETS, PRESETS, Policy
from .core.engine import build_live, build_offline
from .core.frame import AnalysisFrame
from .features import load_wav
from .hotkeys import MOD_ALT, MOD_CONTROL, MOD_SHIFT, VK, Hotkeys
from .core import dsp as core_dsp
from .mapping import (SafetyLimiter, apply_light_controls,
                      burst_is_white, emotion_palette)
from .render.light_slider import LightSlider
from .render.pet import PetWindow
from .render.pet_aura import PetAura
from .render.spatial import (SpatialOverlay, choose_pattern, PATTERN_LABELS,
                             PATTERNS)
from .render.ui_kit import (AMBER, CARD, CARD2, FAINT, HAIR, MINT,
                            MUTED, MiniCard, LevelBar, PillButton,
                            TimelineBar, rr)
from .renderer import Overlay
from .sources.decode import DecodeError, ensure_wav, list_sources
from .ai import capabilities as ai_capabilities
from .ai import describe as ai_describe
from .ai.emotion import EmotionProfile, curve_for, intent_for, profile_for
from .ai.emotion import estimate as emotion_estimate
from .ai.emotion import from_track_curve
from .ai.kws import WakeWordListener, action_for, recognize_wav
from .ai.scene_yamnet import (SCENE_ACCENT, SCENE_LABELS_CN, SceneClassifier,
                              SceneTimeline)

# physical keyboard rows / columns used to recognise a quick "swipe" gesture
_KB_ROWS = ("`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./")
_KB_COLS = ("1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm",
            "8ik,", "9ol.", "0p;/")
SCENE_LABELS = {"music": "音乐", "speech": "人声", "applause": "掌声",
                "laughter": "笑声", "silence": "安静", "other": "环境",
                "live": "实时"}
EMOTION_CN = {"bright": "明亮", "warm": "温暖", "intense": "激烈",
               "sad": "忧伤", "balanced": "平衡", "gentle": "轻柔",
               "explosive": "爆发"}
DJ_MODE_LABELS = {"dim": "跟随", "canvas": "画布", "dj": "DJ"}

BG = "#0B0C0F"
FG = "#F2F3F5"
DIM = "#8B909A"
ACCENT = "#F2A33A"
ACCENT2 = "#FF6B4A"


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.policy = Policy.load(args.preset)
        if args.mode:
            self.policy.mode = args.mode
        if args.safety:
            self.policy.safety = args.safety
        if getattr(args, "engine", None):
            self.policy.engine = args.engine
        if getattr(args, "sensitivity", None) is not None:
            self.policy.sensitivity = float(args.sensitivity)
        if getattr(args, "base_brightness", None) is not None:
            self.policy.base_brightness = float(args.base_brightness)
        if self.policy.sensitivity < 0.0:
            self.policy.sensitivity = {"strict": 0.10, "comfort": 0.40,
                                       "off": 0.60}.get(self.policy.safety, 0.40)
        self.policy.clamp()

        # playlist: user sources (demo音源集, decoded on demand) + bundled WAVs
        self.playlist = self._build_playlist(args)
        self.play_index = self._initial_index(args)
        track = self.playlist[self.play_index]
        self.wav, self.track_note, dec_err = self._decode(track["path"])
        self.track_title = track["title"] if dec_err is None else Path(self.wav).stem
        samples, sr = load_wav(str(self.wav))
        # v3: analysis goes through the pluggable engine chain (perceptual ->
        # classic fallback).  Analysis metadata (duration/bpm/scenes) is read
        # from the engine that actually won, so the classic pre-analysis is not
        # repeated when the perceptual engine is active.
        self.off_engine = build_offline(samples, sr, self.policy)
        self.duration = float(getattr(self.off_engine, "duration", 0.0) or (len(samples) / sr))
        self.sources = build_sources(self.wav, self.duration)

        self.source_key = args.source
        if self.source_key not in self.sources or not self.sources[self.source_key].available:
            self.source_key = "inapp"

        self.live_engine = None
        self.frame = AnalysisFrame(engine=self.off_engine.active_name)
        self.limiter = SafetyLimiter("off")
        self.wake_until = 0.0
        self.wake_cooldown = 0.0
        self.t_start = time.perf_counter()
        self.last = time.perf_counter()
        self.frames = 0
        self.fps = 0.0
        self.fps_t0 = self.last
        self.status_note = (f"引擎 {self.off_engine.describe()}"
                            if self.off_engine.healthy else "引擎不可用")
        self.level = 0.0
        self.env = 0.0
        self.scene = "music"
        self.bands = np.zeros(24, dtype=np.float32)
        self.use_llm = not args.no_llm and bool(os.getenv("OPENDEV_API_KEY"))
        # playlist switching: analysis runs off the Tk thread, a calm loading
        # breath is shown until the new engine is ready
        self._pending: dict | None = None
        self._pending_result: tuple | None = None
        self._last_switch = 0.0
        self._scale_busy = False
        self._space_last = 0.0
        self._dj_held: set[str] = set()
        self._dj_key_times: dict[str, float] = {}
        self._dj_gesture_hist: list = []
        self._dj_gesture_last = 0.0
        # ---- optional AI layer (all of it degrades to heuristics) ----
        self.ai_mode = getattr(args, "ai", "auto") or "auto"
        self.kws_mode = "off" if getattr(args, "no_kws", False) else             (getattr(args, "kws", "auto") or "auto")
        self.emotion_mode = getattr(args, "emotion", "auto") or "auto"
        self.scene_classifier: SceneClassifier | None = None
        self.scene_timeline: SceneTimeline | None = None
        self.scene_conf = 0.0
        self.emotion: EmotionProfile | None = None
        self._emo_level = 0.0
        self._last_tick = 0.0
        self.ai_note = "AI: " + ai_describe()
        self.llm_emotion: dict | None = None
        self.emotion_override: dict | None = None
        self.llm_emotion_mode = getattr(args, "llm_emotion", "auto") or "auto"
        if self.emotion_mode != "off":
            try:
                self.emotion = from_track_curve(
                    getattr(self.off_engine, "curve", None))
            except Exception:  # noqa: BLE001
                self.emotion = None
        self._scene_token = 0
        self._scene_result = None
        self._wake_events: list = []
        self._kws: WakeWordListener | None = None
        self.pet_enabled = not getattr(args, "no_pet", False)
        self.burst = bool(getattr(args, "burst", False))
        self.spatial_mode = getattr(args, "spatial", "off") or "off"
        self.xiaomao_mode = bool(getattr(args, "xiaomao", False))
        self.pet_aura: PetAura | None = None
        self.spatial: SpatialOverlay | None = None
        self._spatial_last = 0.0
        self.pet: PetWindow | None = None

        self.overlay = Overlay(args.width, args.height, args.x, args.y)
        self.overlay.dj_action_callback = self._on_dj_action
        if self.spatial_mode != "off":
            self.spatial = SpatialOverlay(self.overlay.root, self.overlay.width,
                                          self.overlay.height, 0, 0)
        self._build_control()
        self._bind_keys()
        if self.pet_enabled:
            try:
                self.pet = PetWindow(self.overlay.root, self.show_panel, self.quit,
                                     on_next=self.playlist_next, on_long=self.pet_long_press,
                                     on_move=self.pet_moved,
                                     on_xiaomao=self.toggle_xiaomao)
                self.pet_aura = PetAura(self.overlay.root, self.pet)
                self.control.withdraw()
                self.control.protocol("WM_DELETE_WINDOW", self.hide_panel)
            except Exception:  # noqa: BLE001
                self.pet = None
        if self.xiaomao_mode and self.pet is None:
            self.xiaomao_mode = False
        self.hotkeys = Hotkeys([
            (MOD_CONTROL | MOD_ALT, VK["Q"], "quit"),
            (MOD_CONTROL | MOD_ALT, VK["F9"], "quit"),
            (MOD_CONTROL | MOD_SHIFT, VK["Q"], "quit"),
            (MOD_CONTROL | MOD_ALT, VK["D"], "dim"),
            (MOD_CONTROL | MOD_ALT, VK["F10"], "dim"),
            (MOD_CONTROL | MOD_SHIFT, VK["D"], "dim"),
            (MOD_CONTROL | MOD_ALT, VK["H"], "panel"),
            (MOD_CONTROL | MOD_ALT, VK["F11"], "panel"),
            (MOD_CONTROL | MOD_SHIFT, VK["H"], "panel"),
            (MOD_CONTROL | MOD_ALT, VK["1"], "inapp"),
            (MOD_CONTROL | MOD_ALT, VK["2"], "mic"),
            (MOD_CONTROL | MOD_ALT, VK["3"], "loop"),
        ])
        names = sorted({f"{n}={c}" for n, c in self.hotkeys.registered})
        print("Global hotkeys:", ", ".join(names) if names else "none (use the control panel)")

        if not args.no_autoplay:
            self.start_source(self.source_key)
        self._sync_mode_input()
        if getattr(args, "kws_wav", ""):
            self.overlay.root.after(350, self._kws_wav_demo)
        elif self.kws_mode != "off":
            self.overlay.root.after(250, self._start_kws)
        if self.ai_mode != "off":
            self.overlay.root.after(120, self._start_scene_ai)
        if self.llm_emotion_mode != "off":
            self.overlay.root.after(800, lambda: self._start_llm_emotion(self.off_engine))
        self.overlay.root.after(16, self.tick)

    # ------------------------------------------------------------ control
    def _build_control(self) -> None:
        self.control = tk.Toplevel(self.overlay.root)
        self.control.title(f"OpenRay  {__version__}")
        self.control.configure(bg=BG)
        self.control.attributes("-topmost", True)
        self.control.resizable(False, False)
        h = self.overlay.height
        panel_w, panel_h = 720, 600
        self.control.geometry(f"{panel_w}x{panel_h}+18+{max(18, h - panel_h - 24)}")
        self.control.protocol("WM_DELETE_WINDOW", self.quit)

        # ---- brand header -------------------------------------------------
        head = tk.Canvas(self.control, width=688, height=58, bg=BG,
                         highlightthickness=0, bd=0)
        head.place(x=16, y=8)
        head.create_oval(9, 17, 31, 39, outline="", fill="#241C10")
        head.create_oval(13, 21, 27, 35, outline="", fill=AMBER)
        head.create_oval(18, 26, 22, 30, outline="", fill="#FFF4DE")
        head.create_arc(6, 14, 34, 42, start=205, extent=130, style="arc",
                        outline=ACCENT2, width=1)
        head.create_text(44, 17, text="OPENRAY", anchor="w", fill=FG,
                         font=("Microsoft YaHei UI", 13, "bold"))
        head.create_text(44, 37, text="让屏幕看见音乐  ·  AUDIO-REACTIVE LIGHT",
                         anchor="w", fill=FAINT,
                         font=("Microsoft YaHei UI", 7))
        head.create_text(568, 20, text="●  LIVE", anchor="w", fill=MINT,
                         font=("Microsoft YaHei UI", 8, "bold"))
        self._button("退出", self.quit, 620, 14, 84, height=30)
        tk.Frame(self.control, bg=HAIR, height=1).place(x=16, y=66, width=688)

        # ---- status cards -------------------------------------------------
        self.card_mode = MiniCard(self.control, "模式", "跟随", 150, 46)
        self.card_mode.place(x=16, y=74)
        self.card_scene = MiniCard(self.control, "场景", "音乐", 150, 46)
        self.card_scene.place(x=176, y=74)
        self.card_emotion = MiniCard(self.control, "情绪", "平衡", 150, 46)
        self.card_emotion.place(x=336, y=74)
        self.card_beat = MiniCard(self.control, "节拍 / 性能", "—", 192, 46)
        self.card_beat.place(x=496, y=74)

        # ---- brightness ---------------------------------------------------
        self.lbl_level = tk.Label(self.control, text="0%", bg=BG, fg=AMBER,
                                  font=("Microsoft YaHei UI", 15, "bold"),
                                  anchor="w")
        self.lbl_level.place(x=16, y=130, width=96)
        self.level_bar = LevelBar(self.control, width=474)
        self.level_bar.place(x=112, y=142, width=474, height=18)
        self.lbl_status = tk.Label(self.control, text="", bg=BG, fg=MUTED,
                                   font=("Microsoft YaHei UI", 7), anchor="w")
        self.lbl_meter = tk.Label(self.control, text="", bg=BG, fg=ACCENT,
                                  font=("Microsoft YaHei UI", 7), anchor="w")
        self.lbl_meter.place(x=16, y=174, width=688)

        # ---- quick controls ----------------------------------------------
        self.btn_source = self._button("音源 · 内置", self.cycle_source,
                                       16, 194, 152)
        self.btn_mode = self._button("模式 · 跟随", self.cycle_mode,
                                     176, 194, 132)
        self.btn_wake = self._button("F2 唤醒", self.trigger_wake,
                                     316, 194, 120)
        self.btn_light_reset = self._button("光感复位", self.reset_light_controls,
                                            444, 194, 112)
        self.btn_spatial = self._button("光影", self.cycle_spatial, 564, 194, 64)
        self.btn_burst = self._button("爆闪", self.toggle_burst, 636, 194, 68)

        # ---- agent --------------------------------------------------------
        tk.Label(self.control, text="指令", bg=BG, fg=FAINT,
                 font=("Microsoft YaHei UI", 7)).place(x=16, y=247)
        entry_bg = tk.Canvas(self.control, width=510, height=34, bg=BG,
                             highlightthickness=0, bd=0)
        entry_bg.place(x=58, y=236)
        rr(entry_bg, 1, 1, 509, 33, 10, fill=CARD, outline=HAIR)
        self.entry = tk.Entry(self.control, bg=CARD, fg=FG, insertbackground=AMBER,
                              relief="flat", bd=0,
                              font=("Microsoft YaHei UI", 9))
        self.entry.place(x=72, y=242, width=482, height=22)
        self.entry.bind("<Return>", self.on_agent)
        self.entry.bind("<FocusIn>", self._entry_focus_in)
        self.entry.bind("<FocusOut>", self._entry_focus_out)
        self._entry_placeholder = "说句话，比如：温柔一点 / 跟着鼓点 / 亮起来"
        self._show_entry_placeholder()
        self._button("执行", self.on_agent, 582, 236, 106, accent=True, height=34)

        # ---- light sliders ------------------------------------------------
        tk.Label(self.control, text="光感 / LIGHT", bg=BG, fg=FAINT,
                 font=("Microsoft YaHei UI", 7)).place(x=16, y=280)
        self.sens_slider = LightSlider(
            self.control, "灵敏度", value=self.policy.sensitivity,
            accents=((86, 225, 255), (197, 107, 255)), bg=BG,
            on_change=self._on_sensitivity, width=332)
        self.sens_slider.place(x=16, y=294, width=332, height=44)
        self.base_slider = LightSlider(
            self.control, "基础亮度", value=self.policy.base_brightness,
            accents=((255, 196, 107), (255, 122, 92)), bg=BG,
            on_change=self._on_base_brightness, width=332)
        self.base_slider.place(x=356, y=294, width=332, height=44)
        self.lbl_note = tk.Label(self.control, text="", bg=BG, fg=MINT,
                                 font=("Microsoft YaHei UI", 8), anchor="w")
        self.lbl_note.place(x=16, y=584, width=688)

        # ---- player card --------------------------------------------------
        card = tk.Canvas(self.control, width=688, height=232, bg=BG,
                         highlightthickness=0, bd=0)
        card.place(x=16, y=350)
        rr(card, 1, 1, 687, 231, 16, fill=CARD, outline=HAIR)
        self.lbl_kicker = tk.Label(self.control, text="NOW PLAYING", bg=CARD,
                                   fg=FAINT, font=("Microsoft YaHei UI", 7))
        self.lbl_kicker.place(x=34, y=356)
        self.lbl_time = tk.Label(self.control, text="--:-- / --:--", bg=CARD,
                                 fg=MUTED, font=("Consolas", 8), anchor="e")
        self.lbl_time.place(x=558, y=358, width=112)
        self.lbl_track = tk.Label(self.control, text="", bg=CARD, fg=FG,
                                  font=("Microsoft YaHei UI", 12, "bold"),
                                  anchor="w")
        self.lbl_track.place(x=34, y=370, width=600)
        self.scale = TimelineBar(self.control, width=652, height=24, bg=CARD,
                                 command=self._on_scale_move)
        self.scale.place(x=34, y=396, width=652, height=24)
        self.btn_prev = self._button("上一首", self.playlist_prev, 34, 424, 82,
                                     height=30)
        self.btn_play = self._button("暂停", self.player_toggle, 124, 424, 100,
                                     height=30)
        self.btn_next = self._button("下一首", self.playlist_next, 232, 424, 82,
                                     height=30)
        tk.Label(self.control, text="曲目 / PLAYLIST", bg=CARD, fg=FAINT,
                 font=("Microsoft YaHei UI", 7)).place(x=34, y=462)
        self.listbox = tk.Listbox(self.control, bg=CARD, fg=MUTED,
                                  selectbackground=AMBER, selectforeground="#191106",
                                  relief="flat", bd=0, highlightthickness=0,
                                  font=("Microsoft YaHei UI", 9),
                                  activestyle="none")
        self.listbox.place(x=34, y=478, width=632, height=96)
        self._sb = tk.Scrollbar(self.control, command=self.listbox.yview,
                                bg=CARD, troughcolor=CARD2, relief="flat",
                                bd=0, width=10)
        self._sb.place(x=672, y=478, width=10, height=96)
        self.listbox.configure(yscrollcommand=self._sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_playlist_select)
        self._fill_playlist()
        self._style_window()

    def _style_window(self) -> None:
        """Dark OS titlebar + app icon, so the window feels finished."""
        try:
            icon = ASSETS / "icon.ico"
            if icon.exists():
                self.control.iconbitmap(str(icon))
        except Exception:  # noqa: BLE001
            pass
        try:
            import ctypes
            from ctypes import byref, c_int
            self.control.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.control.winfo_id())
            for attr, val in ((20, 1), (19, 1), (33, 2)):
                try:
                    v = c_int(val)
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attr, byref(v), ctypes.sizeof(v))
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    def _button(self, text: str, command, x: int, y: int, w: int,
                accent: bool = False, height: int = 32) -> PillButton:
        b = PillButton(self.control, text, command, width=w, height=height,
                       bg=BG, accent=accent)
        b.place(x=x, y=y, width=w, height=height)
        return b

    def _bind_keys(self) -> None:
        """bind_all: shortcuts and the DJ instrument keep working whichever
        window (control panel or full-screen overlay) currently has focus."""
        c = self.control
        c.bind_all("<Control-Key-1>", lambda e: self.set_source("inapp"))
        c.bind_all("<Control-Key-2>", lambda e: self.set_source("mic"))
        c.bind_all("<Control-Key-3>", lambda e: self.set_source("loop"))
        c.bind_all("<Control-Key-m>", lambda e: self.cycle_mode())
        c.bind_all("<Control-Alt-Key-b>", lambda e: self.toggle_burst())
        c.bind_all("<Control-Alt-Key-l>", lambda e: self.cycle_spatial())
        c.bind_all("<Control-Alt-Key-x>", lambda e: self.toggle_xiaomao())
        c.bind_all("<Control-Key-s>", lambda e: self.reset_light_controls())
        c.bind_all("<Control-Key-r>", lambda e: self.restart())
        c.bind_all("<F2>", lambda e: self.trigger_wake())
        c.bind_all("<space>", self._on_space_key)
        c.bind_all("<Control-Left>", lambda e: self.playlist_prev())
        c.bind_all("<Control-Right>", lambda e: self.playlist_next())
        c.bind_all("<Control-Key-p>", lambda e: self.listbox.focus_set())
        c.bind_all("<Escape>", self.on_escape)
        c.bind_all("<KeyPress>", self.on_any_key)
        c.bind_all("<KeyRelease>", self.on_any_key_release)
        c.focus_force()

    def _focus_is_entry(self) -> bool:
        try:
            return self.control.focus_get() is self.entry
        except Exception:  # noqa: BLE001
            return False

    def on_any_key(self, event) -> None:  # noqa: ANN001
        """DJ instrument: *any* key triggers a cell (full keyboard)."""
        if self.policy.mode != "dj" or self._focus_is_entry():
            return
        if event.state & 0x4 or event.state & 0x20000:   # Ctrl / Alt combos
            return
        if event.keysym.lower() in ("space", "escape", "f2", "control_l",
                                    "control_r", "alt_l", "alt_r", "shift_l",
                                    "shift_r"):
            return
        from .render.scene_dj import key_to_cell

        try:
            cell = key_to_cell(event.keysym, getattr(event, "char", ""))
        except Exception:  # noqa: BLE001
            return
        if cell is None:
            return
        now = time.perf_counter()
        if now - self._dj_key_times.get(event.keysym, 0.0) < 0.09:
            return
        self._dj_key_times[event.keysym] = now
        self._dj_held.add(event.keysym)
        self.on_dj_key(cell[0] * 4 + cell[1])
        self._track_dj_gesture(now, event.keysym, getattr(event, "char", ""))

    def _track_dj_gesture(self, now: float, keysym: str,
                          char: str = "") -> None:
        """A fast glissando across neighbouring keys = a canvas/palette swipe.

        Physical positions are used (not the mapped play cells): pressing
        ``1 2 3 4`` or ``Q W E R`` in a row becomes a horizontal swipe;
        ``Q A Z`` / ``W S X`` in a column becomes a vertical swipe.
        This is the keyboard ``划一道`` gesture; mouse/touchpad swipes also work.
        """
        if self.policy.mode != "dj":
            return
        ch = (char or "").lower()
        if not ch and len(keysym or "") == 1:
            ch = (keysym or "").lower()
        if not ch:
            return
        row = pos = col = -1
        for ri, line in enumerate(_KB_ROWS):
            if ch in line:
                row, pos = ri, line.index(ch)
                break
        for ci, line in enumerate(_KB_COLS):
            if ch in line:
                col = ci * 10 + line.index(ch)
                break
        if row < 0:
            return
        hist = self._dj_gesture_hist
        hist.append((now, row, pos, col))
        hist[:] = [h for h in hist if now - h[0] < 0.7]
        if len(hist) < 3 or now - self._dj_gesture_last < 0.9:
            return
        a, b, c = hist[-3:]
        if a[1] == b[1] == c[1] and (b[2] - a[2]) == (c[2] - b[2]) \
                and abs(c[2] - a[2]) == 2:
            hist.clear()
            self._dj_gesture_last = now
            self.overlay.dj_swipe(240 if c[2] > a[2] else -240, 0)
            self.status_note = "键盘滑动 · 切换画布"
            return
        if a[3] >= 0 and b[3] >= 0 and c[3] >= 0:
            da, db = b[3] - a[3], c[3] - b[3]
            if abs(da) == 1 and abs(db) == 1 and (da > 0) == (db > 0):
                hist.clear()
                self._dj_gesture_last = now
                self.overlay.dj_swipe(0, 240 if db > 0 else -240)
                self.status_note = "键盘滑动 · 切换配色"

    def on_any_key_release(self, event) -> None:  # noqa: ANN001
        self._dj_held.discard(event.keysym)

    def _on_space_key(self, event=None) -> None:  # noqa: ANN001
        if self._focus_is_entry():
            return
        self._on_space(event)

    # ------------------------------------------------------------ playlist
    def _build_playlist(self, args) -> list[dict]:
        items: list[dict] = []
        try:
            infos = list_sources()
        except Exception:  # noqa: BLE001 - playlist is optional
            infos = []
        for info in infos:
            path = Path(info["path"])
            title = path.stem if info.get("kind") == "demo音源集" else f"内置 · {path.stem}"
            items.append({"path": path, "title": title})
        cur = Path(args.wav)
        if cur.exists() and all(Path(it["path"]).resolve() != cur.resolve() for it in items):
            items.insert(0, {"path": cur, "title": cur.stem})
        if not items:
            items.append({"path": ASSETS / "demo_beat.wav", "title": "内置 · demo_beat"})
        return items

    def _initial_index(self, args) -> int:
        if getattr(args, "play", 0):
            return max(0, min(int(args.play) - 1, len(self.playlist) - 1))
        cur = Path(args.wav)
        for i, item in enumerate(self.playlist):
            try:
                if Path(item["path"]).resolve() == cur.resolve():
                    return i
            except OSError:
                continue
        return 0

    def _decode(self, path) -> tuple[Path, str, str | None]:
        """Decode any source to PCM WAV; fall back to the bundled demo on error."""
        try:
            wav, note = ensure_wav(path)
            return Path(wav), note, None
        except DecodeError as exc:
            fb = ASSETS / "demo_beat.wav"
            print(f"[screenlight] 无法解码 {path}: {exc}; 回退到 {fb}")
            return fb, "回退默认曲目", str(exc)
        except Exception as exc:  # noqa: BLE001
            fb = ASSETS / "demo_beat.wav"
            print(f"[screenlight] 解码异常 {path}: {exc}; 回退到 {fb}")
            return fb, "回退默认曲目", str(exc)

    def _fill_playlist(self) -> None:
        self.listbox.delete(0, "end")
        for i, item in enumerate(self.playlist):
            mark = "●" if i == self.play_index else " "
            self.listbox.insert("end", f"{mark}  {i + 1:02d}    {item['title']}")
        for i in range(len(self.playlist)):
            active = i == self.play_index
            self.listbox.itemconfig(
                i, background="#1A1D23" if active else CARD,
                foreground=FG if active else MUTED)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.play_index)
        self.listbox.see(self.play_index)
        self.lbl_track.configure(text=self.track_title)
        if hasattr(self, "lbl_kicker"):
            self.lbl_kicker.configure(
                text=f"NOW PLAYING   ·   {self.play_index + 1:02d} / {len(self.playlist):02d}")

    def _on_playlist_select(self, event=None) -> None:  # noqa: ANN001
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx == self.play_index and self.sources.get("inapp") is not None:
            return
        self.play_index = idx
        self._load_track(self.playlist[idx]["path"])

    def _on_scale_move(self, value) -> None:  # noqa: ANN001
        if self._scale_busy or self.source_key != "inapp":
            return
        src = self.sources.get("inapp")
        if src is None or not src.supports_seek():
            return
        target = float(value)
        if abs(target - src.position()) > 0.8:
            src.seek(target)
            self.status_note = f"跳转到 {self._fmt_time(target)}"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _sync_scale_range(self) -> None:
        self._scale_busy = True
        try:
            self.scale.configure(to=max(self.duration, 1.0))
        finally:
            self._scale_busy = False

    def _sync_player_ui(self) -> None:
        src = self.sources.get("inapp")
        if self.source_key == "inapp" and src is not None and src.available:
            pos = src.position()
            dur = max(self.duration, src.duration, 1.0)
            self._scale_busy = True
            try:
                self.scale.set(min(max(pos, 0.0), dur))
            finally:
                self._scale_busy = False
            self.lbl_time.configure(text=f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")
            if src.supports_pause():
                self.btn_play.configure(text="继续" if src.is_paused() else "暂停")
            else:
                self.btn_play.configure(text="暂停")
            self.lbl_track.configure(text=self.track_title)
            if hasattr(self, "lbl_kicker"):
                self.lbl_kicker.configure(
                    text=f"NOW PLAYING   ·   {self.play_index + 1:02d} / "
                         f"{len(self.playlist):02d}   ·   {self.track_note}")
        else:
            self.lbl_time.configure(text="--:-- / --:--")
            self.lbl_track.configure(text="外部音源输入中（Ctrl+1 回到播放列表）")

    def _on_space(self, event=None) -> None:  # noqa: ANN001
        if self.control.focus_get() is self.entry:
            return
        now = time.perf_counter()
        if now - self._space_last < 0.25:      # key auto-repeat guard
            return
        self._space_last = now
        self.player_toggle()

    # ------------------------------------------------------------ transport
    def playlist_prev(self, event=None) -> None:  # noqa: ANN001
        if not self.playlist:
            return
        self.play_index = (self.play_index - 1) % len(self.playlist)
        self._load_track(self.playlist[self.play_index]["path"])

    def playlist_next(self, event=None) -> None:  # noqa: ANN001
        if not self.playlist:
            return
        self.play_index = (self.play_index + 1) % len(self.playlist)
        self._load_track(self.playlist[self.play_index]["path"])

    def player_toggle(self, event=None) -> None:  # noqa: ANN001
        if self.source_key != "inapp":
            return
        src = self.sources.get("inapp")
        if src is None or not src.available:
            return
        if not src.supports_pause():
            self.status_note = "当前播放器不支持暂停（MCI 不可用）"
            return
        src.toggle_pause()
        self.status_note = "已暂停" if src.is_paused() else "▶ 继续播放"
        self._sync_player_ui()

    def _load_track(self, path, autoplay: bool = True) -> None:
        """Switch track: decode -> swap audio -> analyse in a background thread."""
        try:
            wav, note, err = self._decode(path)
            samples, sr = load_wav(str(wav))
        except Exception as exc:  # noqa: BLE001
            self.status_note = f"切换失败：{exc}"
            return
        old = self.sources.get("inapp")
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        src = InAppSource(wav, len(samples) / sr)
        self.sources["inapp"] = src
        self.duration = src.duration if src.duration > 0.2 else len(samples) / sr
        self.wav = str(wav)
        self.track_note = note
        self.track_title = Path(path).stem
        for key, other in self.sources.items():
            if key != "inapp":
                try:
                    other.stop()
                except Exception:  # noqa: BLE001
                    pass
        self.source_key = "inapp"
        if autoplay:
            src.start()
        self._sync_scale_range()
        self._fill_playlist()
        self._refresh_buttons()
        # background analysis (2-4 s for long tracks); a loading breath is shown
        token = time.perf_counter()
        self._pending = {"token": token}
        self._pending_result = None
        self._emo_level = 0.0
        self.emotion_override = None
        self.status_note = f"分析中… {self.track_title}"
        if self.ai_mode != "off":
            self._start_scene_ai()

        def work() -> None:
            try:
                engine = build_offline(samples, sr, self.policy)
                self._pending_result = (token, engine, None)
            except Exception as exc:  # noqa: BLE001
                self._pending_result = (token, None, exc)

        threading.Thread(target=work, daemon=True).start()
        self._last_switch = time.perf_counter()

    def _drain_analysis(self) -> None:
        result = self._pending_result
        if result is None or self._pending is None:
            return
        token, engine, err = result
        if token != self._pending.get("token"):
            return
        self._pending = None
        self._pending_result = None
        if err is not None or engine is None:
            self.status_note = f"分析失败：{err}"
            return
        self.off_engine = engine
        self.duration = float(getattr(engine, "duration", self.duration) or self.duration)
        self._sync_scale_range()
        if self.emotion_mode != "off":
            try:
                self.emotion = from_track_curve(getattr(engine, "curve", None))
            except Exception:  # noqa: BLE001
                self.emotion = None
            if self.emotion_mode in ("gentle", "balanced", "explosive"):
                from .ai.emotion import CURVES
                self.emotion = EmotionProfile(
                    label=self.emotion_mode, valence=0.5, arousal=0.5,
                    source="manual", **CURVES[self.emotion_mode])
        self.status_note = f"已载入 {self.track_title} · {self.track_note}"

    # ------------------------------------------------------------ actions
    def start_source(self, key: str) -> None:
        src = self.sources.get(key)
        if src is None:
            return
        if not src.available:
            self.status_note = f"{src.label}不可用：{src.reason}"
            return
        try:
            src.start()
        except Exception as exc:
            self.status_note = f"{src.label}启动失败：{exc}"
            return
        if not src.available:
            self.status_note = f"{src.label}不可用：{src.reason}"
            return
        self.source_key = key
        for k, other in self.sources.items():
            if k != key:
                try:
                    other.stop()
                except Exception:
                    pass
        self.status_note = f"音源切换为 {src.label}"
        if key == "loop" and self.policy.mode == "dim":
            self.status_note = "系统内录已开启（Ctrl+M 可切画布/DJ）"
        if key != "inapp":
            # v3: live engine (perceptual-live -> classic-live fallback)
            self.live_engine = build_live(getattr(src, "sr", 44100), self.policy)
            self.status_note += f" · 引擎 {self.live_engine.describe()}"

    def set_source(self, key: str) -> None:
        self.start_source(key)
        self._refresh_buttons()

    def cycle_source(self) -> None:
        keys = ["inapp", "mic", "loop"]
        i = keys.index(self.source_key) if self.source_key in keys else 0
        for step in range(1, len(keys) + 1):
            key = keys[(i + step) % len(keys)]
            if self.sources[key].available:
                self.set_source(key)
                return
        self.status_note = "没有可用的音源"

    def toggle_burst(self) -> None:
        self.burst = not self.burst
        if self.burst:
            self.xiaomao_mode = False
            if self.pet_aura is not None:
                self.pet_aura.hide()
        if self.burst:
            self.status_note = "爆闪已开启 · 亮度≥90% 纯白 / 其余纯黑（光敏提示）"
        else:
            self.status_note = "爆闪已关闭"
        self._refresh_buttons()

    def _spatial_pattern(self) -> str:
        if self.spatial_mode in PATTERNS:
            return self.spatial_mode
        lbl = self.emotion.label if self.emotion is not None else "balanced"
        return choose_pattern(self.scene, lbl, self.frame.valence, self.frame.arousal)

    def cycle_spatial(self) -> None:
        order = ["off", "auto", "columns", "wave", "window", "spotlight"]
        try:
            idx = order.index(self.spatial_mode)
        except ValueError:
            idx = 0
        self.spatial_mode = order[(idx + 1) % len(order)]
        if self.spatial_mode != "off":
            self.xiaomao_mode = False
            if self.pet_aura is not None:
                self.pet_aura.hide()
            self.policy.mode = "dim"
            if self.spatial is None:
                self.spatial = SpatialOverlay(self.overlay.root, self.overlay.width,
                                              self.overlay.height, 0, 0)
            label = "AI 自动" if self.spatial_mode == "auto" else PATTERN_LABELS.get(self.spatial_mode, "")
            self.status_note = f"光影: {label}（空间蒙版）"
        else:
            self.status_note = "光影已关闭"
            if self.spatial is not None:
                self.spatial.hide()
        self._refresh_buttons()


    def _raise_window(self, win) -> None:
        """Force a topmost Tk window above the full-screen overlay."""
        try:
            win.update_idletasks()
            win.lift()
            win.attributes("-topmost", True)
            if sys.platform == "win32":
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
                ctypes.windll.user32.SetWindowPos(
                    hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        except Exception:  # noqa: BLE001
            pass

    def pet_long_press(self) -> None:
        if self.source_key == "inapp":
            self.player_toggle()
            self.status_note = "桌宠长按：暂停 / 继续"
        else:
            self.status_note = "桌宠长按：当前音源不支持暂停"

    def pet_moved(self) -> None:
        if self.pet_aura is not None:
            self.pet_aura.follow()

    def toggle_xiaomao(self) -> None:
        self.set_xiaomao(not self.xiaomao_mode)

    def set_xiaomao(self, on: bool) -> None:
        if on and self.pet is None:
            self.status_note = "小茂模式需要桌面宠物（去掉 --no-pet）"
            return
        self.xiaomao_mode = bool(on)
        if self.xiaomao_mode:
            self.burst = False
            self.spatial_mode = "off"
            if self.spatial is not None:
                self.spatial.hide()
            try:
                self.overlay.set_suppressed(True)
            except Exception:  # noqa: BLE001
                pass
            if self.pet_aura is not None:
                try:
                    palette = emotion_palette(self.policy.palette, self.frame.valence,
                                            self.frame.arousal) if self.frame is not None else self.policy.palette
                except Exception:  # noqa: BLE001
                    palette = self.policy.palette
                self.pet_aura.show(self.level, palette,
                                   self.frame.valence if self.frame else 0.5)
            self.status_note = "小茂模式 · 只有宠物周围有亮度蒙版"
        else:
            if self.pet_aura is not None:
                self.pet_aura.hide()
            try:
                self.overlay.set_suppressed(False)
            except Exception:  # noqa: BLE001
                pass
            self.status_note = "小茂模式已关闭"
        self._refresh_buttons()

    def show_panel(self) -> None:
        try:
            self.control.deiconify()
            self._raise_window(self.control)
            self.control.focus_force()
        except tk.TclError:
            pass

    def hide_panel(self) -> None:
        try:
            self.control.withdraw()
            self.overlay.ensure_topmost()
            self._raise_window(self.control)
        except tk.TclError:
            pass
        if self.pet is not None:
            try:
                self.pet.win.lift()
            except tk.TclError:
                pass

    def cycle_mode(self) -> None:
        order = ["dim", "canvas", "dj"]
        self.policy.mode = order[(order.index(self.policy.mode) + 1) % len(order)]
        self._sync_mode_input()
        self.status_note = f"模式切换为 {DJ_MODE_LABELS.get(self.policy.mode, self.policy.mode)}"
        self._refresh_buttons()

    def _sync_mode_input(self) -> None:
        """DJ needs the overlay to receive mouse/touch/keyboard; other modes are
        click-through and keyboard focus returns to the control panel."""
        dj = self.policy.mode == "dj"
        try:
            self.overlay.set_interactive(dj)
        except Exception:  # noqa: BLE001 - input is best effort
            pass
        try:
            if dj:
                self.overlay.focus_dj()
            else:
                self.control.focus_force()
        except Exception:  # noqa: BLE001
            pass

    def on_dj_key(self, index: int) -> None:
        if self.policy.mode != "dj":
            return
        self.overlay.dj_key(index)
        self.status_note = f"DJ · {self.overlay.dj_status()}"

    def _on_dj_action(self, action: str, status: str) -> None:
        labels = {"style": "画布", "palette": "配色", "tap": "触发", "none": ""}
        self.status_note = f"{labels.get(action, '')} {status}".strip()

    def _on_sensitivity(self, value: float) -> None:
        self.policy.sensitivity = min(max(float(value), 0.0), 1.0)
        self.status_note = f"灵敏度 {self.policy.sensitivity * 100:.0f}%"

    def _on_base_brightness(self, value: float) -> None:
        self.policy.base_brightness = min(max(float(value), 0.0), 1.0)
        self.status_note = f"基础亮度 {self.policy.base_brightness * 100:.0f}%"

    def reset_light_controls(self) -> None:
        self.policy.sensitivity = 0.40
        self.policy.base_brightness = 0.08
        try:
            self.sens_slider.set_value(self.policy.sensitivity)
            self.base_slider.set_value(self.policy.base_brightness)
        except Exception:  # noqa: BLE001
            pass
        self.status_note = "光感复位：灵敏度 40% · 基础亮度 8%"

    def restart(self) -> None:
        if self.source_key == "inapp":
            self.sources["inapp"].restart()
            self.status_note = "重新播放内置曲目"

    def trigger_wake(self) -> None:
        # v2.0.1: no screen overlay; v2.4: the pet reacts instead
        self.wake_until = 0.0
        self.status_note = f"唤醒触发：{self.policy.text}"
        if self.pet is not None:
            try:
                self.pet.wake_react("我在～")
            except Exception:  # noqa: BLE001
                pass

    def on_escape(self, event=None) -> None:  # noqa: ANN001
        if self.control.focus_get() is self.entry:
            self.entry.delete(0, "end")
            return
        self.quit()

    def _show_entry_placeholder(self) -> None:
        if self.entry.get().strip():
            return
        self.entry.configure(fg=FAINT)
        self.entry.insert(0, self._entry_placeholder)

    def _entry_focus_in(self, event=None) -> None:  # noqa: ANN001
        if self.entry.get() == self._entry_placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg=FG)

    def _entry_focus_out(self, event=None) -> None:  # noqa: ANN001
        self._show_entry_placeholder()

    def on_agent(self, event=None) -> None:  # noqa: ANN001
        text = self.entry.get().strip()
        if not text or text == self._entry_placeholder:
            return
        if "换尾巴" in text or "尾巴" in text and any(k in text for k in ("换", "随机")):
            if self.pet is not None:
                self.pet._reroll_tails()
                self.status_note = "尾巴已重掷"
            self.entry.delete(0, "end")
            return
        if "小茂模式" in text or "小茂" in text and any(k in text for k in ("模式", "打开", "关闭", "关掉")):
            self.set_xiaomao(not any(k in text for k in ("关", "停", "取消", "别")))
            self.entry.delete(0, "end")
            return
        if any(k in text for k in ("爆闪", "频闪", "纯白", "闪白")):
            self.burst = not any(k in text for k in ("关", "停", "别", "取消"))
            self.entry.delete(0, "end")
            self.status_note = "爆闪已开启 · 亮度≥90% 纯白 / 其余纯黑（光敏提示）" if self.burst else "爆闪已关闭"
            self._refresh_buttons()
            return
        for keys, mode, label in (
            (("四列", "四块", "分屏", "四格"), "columns", "四列"),
            (("海浪", "波浪"), "wave", "海浪"),
            (("窗光", "斜光", "窗户", "光柱"), "window", "窗光"),
            (("追光", "舞台光", "spotlight"), "spotlight", "追光"),
        ):
            if any(k in text for k in keys):
                self.spatial_mode = mode
                self.policy.mode = "dim"
                self.entry.delete(0, "end")
                self.status_note = f"光影: {label}（空间蒙版）"
                self._refresh_buttons()
                return
        if "光影" in text and any(k in text for k in ("关", "停", "别", "取消")):
            self.spatial_mode = "off"
            if self.spatial is not None:
                self.spatial.hide()
            self.entry.delete(0, "end")
            self.status_note = "光影已关闭"
            self._refresh_buttons()
            return
        policy, note, backend = apply_text(text, self.policy, use_llm=self.use_llm)
        if any(w in text for w in ("亮起来", "唤醒", "看见声音")):
            self.trigger_wake()
        emotion = intent_for(text)
        if emotion is not None:
            self.emotion_override = emotion
            tag = f"情绪:{emotion['label']}"
            note = (note + " · " + tag) if note else tag
        self.policy = policy
        self.limiter.set_level("off")
        self.entry.delete(0, "end")
        self.status_note = f"{note} · {backend}"
        self._refresh_buttons()

    # ------------------------------------------------------------ AI layer
    def _start_scene_ai(self) -> None:
        """Background YAMNet/heuristic scene timeline for the current track."""
        if self.ai_mode == "off":
            return
        wav = str(self.wav)
        token = time.perf_counter()
        self._scene_token = token
        self._scene_result = None
        self.ai_note = "场景识别加载中…"

        def work() -> None:
            try:
                clf = SceneClassifier(self.ai_mode)
                samples, sr = load_wav(wav)
                timeline = SceneTimeline.build(samples, sr, clf, max_windows=60)
                self._scene_result = (token, timeline, clf, None)
            except Exception as exc:  # noqa: BLE001
                self._scene_result = (token, None, None, exc)

        threading.Thread(target=work, name="screenlight-scene", daemon=True).start()

    def _drain_scene_ai(self) -> None:
        result = self._scene_result
        if result is None:
            return
        token, timeline, clf, err = result
        if token != self._scene_token:
            return
        self._scene_result = None
        if err is not None or timeline is None:
            self.ai_note = f"场景识别失败: {err}"
            return
        self.scene_timeline = timeline
        self.scene_classifier = clf
        summary = timeline.summary()
        top = "、".join(f"{SCENE_LABELS_CN.get(k, k)}{v:.0f}s"
                        for k, v in list(summary.items())[:3])
        self.ai_note = f"场景 {clf.name} · {top}"

    def _start_llm_emotion(self, engine) -> None:
        """Ask the LLM once per track to interpret the emotion (optional)."""
        if self.llm_emotion_mode == "off":
            return
        try:
            from .ai import llm as ai_llm
            if not ai_llm.available():
                return
        except Exception:  # noqa: BLE001
            return
        curve = getattr(engine, "curve", None)
        if curve is None:
            return
        try:
            import numpy as _np
            features = {
                "bpm": round(float(curve.bpm), 1),
                "local_valence": round(float(_np.mean(curve.valence)), 3),
                "local_arousal": round(float(_np.mean(curve.arousal)), 3),
                "duration_s": round(float(curve.duration), 1),
            }
        except Exception:  # noqa: BLE001
            return
        title = self.track_title

        def work() -> None:
            try:
                from .ai import llm as ai_llm
                hint = ai_llm.emotion_hint(title, features)
                if hint is not None:
                    self._pending_llm_emotion = hint
                    self.ai_note = (f"LLM 情绪 {hint['label']} "
                                    f"({hint['reason']}) · " + self.ai_note)
            except Exception:  # noqa: BLE001
                pass

        self._pending_llm_emotion = None
        threading.Thread(target=work, name="screenlight-llm-emotion",
                         daemon=True).start()

    def _start_kws(self) -> None:
        if self._kws is not None or self.kws_mode == "off":
            return
        try:
            listener = WakeWordListener(self._on_wake_event, self.kws_mode)
            listener.start()
            self._kws = listener
            self.ai_note = f"语音唤醒 {listener.name}"
        except Exception as exc:  # noqa: BLE001
            self.ai_note = f"语音唤醒不可用: {exc}"

    def _on_wake_event(self, event) -> None:  # called from the listener thread
        self._wake_events.append(event)

    def _consume_wake_events(self) -> None:
        while self._wake_events:
            event = self._wake_events.pop(0)
            action = action_for(event.text)
            self.status_note = (f"语音「{event.text}」 "
                                f"({event.confidence:.2f} · {event.backend})")
            self._apply_voice_action(action, event.text)

    def _apply_voice_action(self, action, text: str) -> None:
        if action == "wake":
            self.policy.text = "看见声音"
            self.wake_until = 0.0
            return
        if action == "canvas":
            self.policy.mode = "canvas"
            self._refresh_buttons()
            return
        phrase = {"gentle": "温柔一点", "brighter": "亮一点",
                  "cyber": "赛博朋克", "follow_beat": "跟着鼓点",
                  "unsafe": "关掉护栏"}.get(action, text)
        emotion = intent_for(phrase)
        if emotion is not None:
            self.emotion_override = emotion
        policy, note, backend = apply_text(phrase, self.policy,
                                           use_llm=self.use_llm)
        self.policy = policy
        self.limiter.set_level("off")
        if action == "cyber":
            self.policy.mode = "canvas"
        self.status_note = f"语音: {note} · {backend}"
        self._sync_mode_input()
        self._refresh_buttons()

    def _kws_wav_demo(self) -> None:
        path = getattr(self.args, "kws_wav", "")
        self.status_note = f"识别唤醒词: {Path(path).name}"
        try:
            backend = "auto" if self.kws_mode == "off" else self.kws_mode
            events = recognize_wav(path, backend)
        except Exception as exc:  # noqa: BLE001
            self.status_note = f"唤醒词识别失败: {exc}"
            return
        if not events:
            self.status_note = "唤醒词识别：没有匹配"
            return
        for event in events:
            self._wake_events.append(event)
        self._consume_wake_events()

    def _refresh_buttons(self) -> None:
        src = self.sources.get(self.source_key)
        self.btn_source.configure(text=f"音源: {src.label if src else '—'}")
        mode_cn = DJ_MODE_LABELS[self.policy.mode]
        self.btn_mode.configure(text=f"模式: {mode_cn}")

    # ------------------------------------------------------------ loop
    def tick(self) -> None:
        try:
            for action in self.hotkeys.poll():
                if action == "quit":
                    self.quit()
                    return
                if action == "dim":
                    self.policy.mode = "dim"
                    self.status_note = "已切到后台跟随模式"
                    self._refresh_buttons()
                elif action == "panel":
                    self.show_panel()
                elif action in ("inapp", "mic", "loop"):
                    self.set_source(action)

            self._consume_wake_events()
            now = time.perf_counter()
            self.frames += 1
            if now - self.fps_t0 >= 1.0:
                self.fps = self.frames / (now - self.fps_t0)
                self.frames = 0
                self.fps_t0 = now

            self._sample(now)
            self._drain_analysis()
            self._drain_scene_ai()
            # auto-advance to the next playlist entry when a track ends
            if (self.source_key == "inapp" and self._pending is None and self.playlist
                    and now - self._last_switch > 1.0):
                src_inapp = self.sources.get("inapp")
                if src_inapp is not None and src_inapp.available and src_inapp.finished():
                    self.playlist_next()

            # the engine already applied base/gain/gamma/contrast for the
            # current policy, so Agent changes take effect immediately
            target = self.frame.level

            # ---- emotion response: same BPM can feel different -------------
            # A sad/gentle song gets a slower, lower-gain brightness contour; an
            # intense one gets a punchier curve; valence also shifts the palette
            # warm/cool.  User light controls scale the result afterwards.
            emo_curve = {"attack_ms": 40.0, "decay_ms": 300.0,
                         "gain": 1.0, "accent_gain": 1.0,
                         "label": "balanced"}
            if getattr(self, "_pending_llm_emotion", None):
                self.llm_emotion = self._pending_llm_emotion
                self._pending_llm_emotion = None
            v_eff, a_eff = self.frame.valence, self.frame.arousal
            if self.emotion_override is not None:
                v_eff = float(self.emotion_override["valence"])
                a_eff = float(self.emotion_override["arousal"])
            elif self.llm_emotion is not None and self.emotion_mode != "off":
                v_eff = min(max(0.78 * v_eff + 0.22 * self.llm_emotion["valence"],
                                0.0), 1.0)
                a_eff = min(max(0.78 * a_eff + 0.22 * self.llm_emotion["arousal"],
                                0.0), 1.0)
            if self.emotion_mode != "off":
                emo_curve = curve_for(v_eff, a_eff)
                dt_tick = max(now - self._last_tick, 1e-4) if self._last_tick else 1 / 60.0
                self._last_tick = now
                emo_target = min(max(target * emo_curve["gain"], 0.0), 1.0)
                self._emo_level = core_dsp.ar_step(
                    self._emo_level, emo_target, dt_tick,
                    emo_curve["attack_ms"], emo_curve["decay_ms"])
                target = self._emo_level
            elif self._last_tick:
                self._emo_level = target
                self._last_tick = now
            palette = (emotion_palette(self.policy.palette, v_eff, a_eff)
                       if self.emotion_mode != "off" else self.policy.palette)
            # ---- user light controls (v2.1) ------------------------------
            # Do not change the response curve/timing: take the existing
            # response signal as-is, set the silent floor with base brightness,
            # then scale the *amount* by sensitivity.  Timing is untouched.
            try:
                from .features import shape_level_smooth
                silence = shape_level_smooth(0.0, self.policy)
            except Exception:  # noqa: BLE001
                silence = 0.08
            target = apply_light_controls(
                target, self.policy.base_brightness, self.policy.sensitivity,
                silence=silence)
            self.level = self.limiter.step(target, now)
            bpm_text = f"{self.frame.bpm:.0f} BPM" if self.frame.bpm > 0 else "-- BPM"
            if self.policy.mode == "canvas":
                hint = (f"{self.scene_label()}  ·  {bpm_text}  ·  "
                        f"{self.ai_short()}  ·  {self.fps:.0f} fps")
            elif self.policy.mode == "dj":
                hint = (f"{self.overlay.dj_status()}   ·   {bpm_text}   ·   "
                        f"按键演奏 · 横向滑动换画布 · 纵向滑动换配色")
            else:
                hint = ""
            display_text = ""
            accent = self.frame.accent * SCENE_ACCENT.get(self.scene, 1.0)
            accent *= emo_curve.get("accent_gain", 1.0)
            accent = min(max(accent, 0.0), 1.8)
            use_spatial = (not self.burst and not self.xiaomao_mode
                           and self.spatial_mode != "off"
                           and self.policy.mode == "dim")
            if self.xiaomao_mode and self.pet is not None:
                if self.spatial is not None:
                    self.spatial.hide()
                try:
                    self.overlay.set_suppressed(True)
                    self.overlay.hide_burst()
                except Exception:  # noqa: BLE001
                    pass
                if self.pet_aura is not None:
                    try:
                        self.pet_aura.follow()
                        self.pet_aura.show(self.level, palette, self.frame.valence)
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    self.pet.win.lift()
                    self.control.lift()
                except tk.TclError:
                    pass
            elif self.burst:
                if self.spatial is not None:
                    self.spatial.hide()
                try:
                    self.overlay.set_suppressed(False)
                    self.overlay.render_burst(burst_is_white(self.level))
                except Exception:  # noqa: BLE001
                    pass
            elif use_spatial:
                try:
                    self.overlay.hide_burst()
                    self.overlay.set_suppressed(True)
                    if self.spatial is None:
                        self.spatial = SpatialOverlay(self.overlay.root, self.overlay.width,
                                                      self.overlay.height, 0, 0)
                    pattern = self._spatial_pattern()
                    emo = self.emotion.label if self.emotion is not None else "balanced"
                    self.spatial.show(pattern, self.level, self.bands, palette,
                                      self.frame.valence, self.frame.arousal,
                                      self.frame.beat_phase, emo)
                    try:
                        self.control.lift()
                    except tk.TclError:
                        pass
                    if self.pet is not None and self.frames % 6 == 0:
                        try:
                            self.pet.win.lift()
                        except tk.TclError:
                            pass
                except Exception as exc:  # noqa: BLE001
                    self.status_note = f"光影异常: {exc}"
                    self.spatial_mode = "off"
                    self.overlay.set_suppressed(False)
            else:
                if self.spatial is not None:
                    self.spatial.hide()
                try:
                    self.overlay.set_suppressed(False)
                    self.overlay.hide_burst()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.overlay.render(
                        self.level, self.bands, self.policy.mode, palette,
                        display_text, self.policy.max_dim, hint=hint,
                        accent=accent, beat_phase=self.frame.beat_phase,
                    )
                except TypeError:
                    # old renderer without the v3 accent/phase keywords
                    self.overlay.render(
                        self.level, self.bands, self.policy.mode, self.policy.palette,
                        display_text, self.policy.max_dim, hint=hint,
                    )

            if self.frames % 20 == 0:
                self._update_labels(now)
                if not self.xiaomao_mode and not self.burst and self.spatial_mode == "off":
                    self.overlay.ensure_topmost()
                self._raise_window(self.control)
                if self.pet is not None:
                    try:
                        self.pet.win.lift()
                    except tk.TclError:
                        pass
            if self.pet is not None and self.frames % 3 == 0:
                try:
                    lbl = self.emotion.label if self.emotion is not None else "balanced"
                    self.pet.set_motion(self.level, self.frame.bpm,
                                        self.frame.accent, lbl, self.frame.valence)
                except Exception:  # noqa: BLE001
                    pass

            if self.args.duration and now - self.t_start >= self.args.duration:
                self.quit()
                return
        except Exception as exc:  # keep the overlay alive
            self.status_note = f"渲染异常：{exc}"
        finally:
            try:
                self.overlay.root.after(16, self.tick)
            except tk.TclError:
                pass

    def _sample(self, now: float) -> None:
        if self.source_key == "inapp":
            src = self.sources.get("inapp")
            t = src.position() if src is not None else 0.0
            if self._pending is not None or not self.off_engine.healthy:
                # calm loading breath while the new track is being analysed
                lvl = 0.16 + 0.06 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.7 * now))
                self.frame = AnalysisFrame(t=t, level=lvl, energy=0.5,
                                           bands=np.zeros(24, dtype=np.float32),
                                           engine="loading")
            else:
                self.frame = self.off_engine.frame_at(t, self.policy, now)
            scene, conf = "other", 0.0
            if self.scene_timeline is not None:
                scene, conf = self.scene_timeline.label_at(t)
            if scene in ("other", "silence") and conf < 0.3:
                scene = self.frame.scene
            self.scene = scene
            self.scene_conf = conf
        else:
            src = self.sources[self.source_key]
            block = src.latest(2048)
            if block is not None and len(block) and self.live_engine is not None:
                self.live_engine.push(block)
            if self.live_engine is not None:
                self.frame = self.live_engine.frame(now, self.policy)
            else:
                self.frame = AnalysisFrame(engine="none")
            self.scene = "live"
        self.env = float(self.frame.energy)
        self.bands = self.frame.bands

    def auto_wake(self, now: float) -> bool:
        if self.source_key == "inapp":
            return False
        if self.args.no_auto_wake:
            return False
        if now < self.wake_until:
            return True
        loud = self.frame.accent > 0.70 or self.frame.energy > 0.85
        if loud and now > self.wake_cooldown:
            self.wake_until = now + 2.0
            self.wake_cooldown = now + 3.0
            return True
        return False

    def scene_label(self) -> str:
        return SCENE_LABELS.get(self.scene, self.scene)

    def ai_short(self) -> str:
        bits = []
        if self.scene_timeline is not None:
            bits.append(f"场景:{self.scene_label()}")
        if self.emotion is not None:
            bits.append(f"情绪:{self.emotion.label}")
        if self._kws is not None:
            bits.append(f"唤醒:{self._kws.name}")
        return " · ".join(bits) if bits else "AI:off"

    def _update_labels(self, now: float) -> None:
        src = self.sources.get(self.source_key)
        avail = " · ".join(k for k, s in self.sources.items() if s.available)
        engine = self.live_engine if self.source_key != "inapp" else self.off_engine
        engine_txt = engine.describe() if engine is not None else "none"
        self.lbl_status.configure(
            text=f"音源 {src.label if src else '—'}   ·   引擎 {engine_txt}   ·   可用 {avail}"
        )
        bpm_text = f"{self.frame.bpm:.0f} BPM" if self.frame.bpm > 0 else "-- BPM"
        if self.emotion_mode != "off":
            v_now = (self.emotion_override["valence"]
                     if self.emotion_override is not None
                     else self.frame.valence)
            a_now = (self.emotion_override["arousal"]
                     if self.emotion_override is not None
                     else self.frame.arousal)
            self.emotion = profile_for(v_now, a_now)
            tag = " 指令" if self.emotion_override is not None else ""
            emo = f"{EMOTION_CN.get(self.emotion.label, self.emotion.label)}{tag}"
        else:
            emo = "—"
        self.lbl_level.configure(text=f"{self.level * 100:3.0f}%")
        self.level_bar.set(self.level)
        self.card_mode.set(DJ_MODE_LABELS.get(self.policy.mode, self.policy.mode))
        self.card_scene.set(self.scene_label())
        self.card_emotion.set(emo)
        self.card_beat.set(f"{bpm_text}   ·   {self.fps:.0f} fps")
        self.lbl_meter.configure(
            text=f"{bpm_text}   ·   节拍置信 {self.frame.beat_confidence:.2f}   ·   "
                 f"{self.scene_label()}   ·   感度 {self.policy.sensitivity * 100:.0f}%"
        )
        note = self.status_note
        if self.source_key != "inapp":
            s = self.sources[self.source_key]
            if not s.available:
                note = f"{s.label}不可用：{s.reason}"
        if self.xiaomao_mode:
            note = "小茂模式 · 仅宠物周围亮度蒙版"
        elif self.burst:
            note = "爆闪 · 亮度≥90% 纯白 / 其余纯黑"
        elif self.spatial_mode != "off" and self.policy.mode == "dim":
            pat = self._spatial_pattern()
            ai = " · AI" if self.spatial_mode == "auto" else ""
            note = f"光影 · {PATTERN_LABELS.get(pat, pat)}{ai}"
        ai_extra = f"   ·   {self.ai_note}" if self.ai_note else ""
        shown = (note + ai_extra).strip()
        if len(shown) > 96:
            shown = shown[:94] + "…"
        self.lbl_note.configure(text=shown)
        self._sync_player_ui()
        self._refresh_buttons()

    # ------------------------------------------------------------ lifecycle
    def run(self) -> None:
        self._refresh_buttons()
        self.overlay.root.mainloop()

    def quit(self) -> None:
        if self.spatial is not None:
            try:
                self.spatial.close()
            except Exception:  # noqa: BLE001
                pass
        if self.pet is not None:
            try:
                self.pet.close()
            except Exception:  # noqa: BLE001
                pass
        if self._kws is not None:
            try:
                self._kws.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.hotkeys.close()
        except Exception:
            pass
        for src in self.sources.values():
            try:
                src.stop()
            except Exception:
                pass
            try:
                if hasattr(src, "close"):
                    src.close()
            except Exception:
                pass
        try:
            self.overlay.set_interactive(False)
        except Exception:  # noqa: BLE001
            pass
        self.overlay.close()


def _screen_size() -> tuple[int, int]:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    probe = tk.Tk()
    probe.withdraw()
    w, h = probe.winfo_screenwidth(), probe.winfo_screenheight()
    probe.destroy()
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="screenlight", description="OpenRay v2.4")
    p.add_argument("--preset", default=str(PRESETS / "default.json"))
    p.add_argument("--wav", default=str(ASSETS / "demo_beat.wav"))
    p.add_argument("--source", default="inapp", choices=["inapp", "mic", "loop"])
    p.add_argument("--mode", default=None, choices=["dim", "canvas", "dj"])
    p.add_argument("--safety", default=None, choices=["off", "comfort", "strict"],
                   help="legacy preset: maps to sensitivity 60/40/10%%")
    p.add_argument("--sensitivity", type=float, default=None,
                   help="0..1 dynamic amplification; timing stays unchanged")
    p.add_argument("--base-brightness", type=float, default=None,
                   help="0..1 screen brightness when the music is silent")
    p.add_argument("--engine", default=None, choices=["auto", "perceptual", "classic"],
                   help="light engine: auto (perceptual with classic fallback) / "
                        "perceptual / classic")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--x", type=int, default=0)
    p.add_argument("--y", type=int, default=0)
    p.add_argument("--no-autoplay", action="store_true", help="不自动播放内置曲目")
    p.add_argument("--no-llm", action="store_true", help="只用规则解析，不调用 OpenDev")
    p.add_argument("--no-auto-wake", action="store_true", help="麦克风模式下不自动触发唤醒")
    p.add_argument("--duration", type=float, default=0.0, help="N 秒后自动退出（0=不退出）")
    p.add_argument("--list-audio", action="store_true", help="列出所有音频设备后退出")
    p.add_argument("--list-sources", action="store_true",
                   help="列出播放列表音源（demo音源集 + assets）后退出")
    p.add_argument("--play", type=int, default=0,
                   help="启动时播放播放列表第 N 首（1 基）")
    p.add_argument("--test-mic", action="store_true", help="尝试打开麦克风后退出")
    p.add_argument("--ai", default="auto", choices=["auto", "off"],
                   help="场景识别: auto = YAMNet(有模型时) / 启发式")
    p.add_argument("--kws", default="off", choices=["auto", "off", "sherpa", "sapi"],
                   help="语音唤醒后端（默认 off：开麦会触发 Windows 通信音频压低，影响音质）")
    p.add_argument("--no-kws", action="store_true", help="关闭语音唤醒")
    p.add_argument("--no-pet", action="store_true", help="不显示桌面宠物，直接打开控制面板")
    p.add_argument("--burst", action="store_true", help="爆闪模块：亮度 >=90%% 时纯白，否则纯黑")
    p.add_argument("--xiaomao", action="store_true", help="小茂模式：只给宠物周围加亮度蒙版，其余屏幕不动")
    p.add_argument("--spatial", default="off",
                   choices=["off", "auto", "columns", "wave", "window", "spotlight"],
                   help="空间光模块：四列 / 海浪 / 窗光 / 追光 / AI 自动")
    p.add_argument("--kws-wav", default="",
                   help="用 WAV 文件演示唤醒词（无需麦克风），例如 --kws-wav wake.wav")
    p.add_argument("--emotion", default="auto",
                   choices=["auto", "off", "gentle", "balanced", "explosive"],
                   help="情绪响应曲线（启发式；off 关闭）")
    p.add_argument("--llm-emotion", default="off", choices=["auto", "off"],
                   help="每首歌用 LLM 读特征纠正情绪（需要 OPENDEV key；off 关闭）")
    p.add_argument("--list-ai", action="store_true",
                   help="列出 AI 后端/模型/唤醒词后退出")
    p.add_argument("--llm-test", action="store_true",
                   help="测试 OpenDev/OpenAI 兼容接口连通性后退出")
    p.add_argument("--llm-base", default="", help="临时覆盖 OPENDEV_BASE_URL")
    p.add_argument("--llm-model", default="", help="临时覆盖 OPENDEV_MODEL")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_sources:
        from .sources.decode import list_sources
        for i, info in enumerate(list_sources(), 1):
            path = Path(info["path"])
            mark = "已缓存" if info.get("cached") else "首次播放时解码"
            print(f"{i:2d}  [{info.get('kind', ''):8s}] {path.name}   ({mark})")
        return 0
    if args.list_audio:
        from .audio import list_devices
        for d in list_devices():
            print(f"{d['index']:>2}  in={d['in']} out={d['out']} rate={d['rate']:>5} "
                  f"loop={int(d['loop'])}  {d['host']:<18} {d['name']}")
        return 0
    if args.test_mic:
        from .audio import MicSource
        mic = MicSource()
        mic.start()
        print(f"available={mic.available}")
        print(f"device={mic.device_name}")
        print(f"reason={mic.reason}")
        mic.stop()
        return 0
    if args.no_kws:
        args.kws = "off"
    if args.llm_base:
        os.environ["OPENDEV_BASE_URL"] = args.llm_base
    if args.llm_model:
        os.environ["OPENDEV_MODEL"] = args.llm_model
    if args.llm_test:
        from .ai.llm import probe
        info = probe()
        print("LLM endpoint:", info["base"])
        print("model       :", info["model"])
        print("key         :", info["key"])
        print("ok          :", info["ok"])
        if info.get("reply"):
            print("reply       :", info["reply"])
        if info.get("error"):
            print("error       :", info["error"])
        return 0 if info["ok"] else 1
    if args.list_ai:
        caps = ai_capabilities()
        print("AI capabilities")
        for key, value in caps.items():
            print(f"  {key:18s} {value}")
        from .ai.kws import SHERPA_PHRASES
        from .ai.palette import list_moods
        print("  wake keywords      ", ", ".join(SHERPA_PHRASES))
        print("  emotion curves     gentle / balanced / explosive (heuristic)")
        print("  palette moods      ", ", ".join(m["label"] for m in list_moods()))
        print("  llm env            OPENDEV_BASE_URL / OPENDEV_API_KEY / OPENDEV_MODEL")
        print("  llm configured     ", bool(os.getenv("OPENDEV_BASE_URL")
                                           and os.getenv("OPENDEV_API_KEY")))
        print("  llm endpoint       ", os.getenv("OPENDEV_BASE_URL", "(unset)"))
        print("  llm model          ", os.getenv("OPENDEV_MODEL", "gpt-4o-mini"))
        print("  emotion commands   ", "悲伤/温柔/明亮/激烈 …（本地即时；LLM 可选）")
        return 0
    if args.width <= 0 or args.height <= 0:
        args.width, args.height = _screen_size()
    app = App(args)
    app.run()
    return 0
