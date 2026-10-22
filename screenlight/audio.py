"""Audio sources: in-app playback (always available), mic, system loopback.

pyaudiowpatch is bundled under vendor/ so users do not install anything.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .vendor import ensure_pyaudiowpatch

try:  # Windows only
    import winsound
except Exception:  # pragma: no cover
    winsound = None  # type: ignore[assignment]


def _load_pyaudio():
    try:
        import pyaudiowpatch as pa  # type: ignore
        return pa
    except Exception:
        return ensure_pyaudiowpatch()


class BaseSource:
    key = "base"
    label = "音源"
    available = True
    reason = ""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def position(self) -> float:
        return 0.0
    def latest(self, n: int = 4096) -> np.ndarray | None:
        return None


# ---------------------------------------------------------------------------
# playback engines (for the built-in / playlist tracks)
# ---------------------------------------------------------------------------

class _Player:
    """Minimal player contract: open / play / pause / resume / seek / close."""

    supports_seek = False
    supports_pause = False
    ok = False
    mode = "none"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.length = 0.0

    def play(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def toggle(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def position(self) -> float:
        return 0.0
    def paused(self) -> bool:
        return False
    def finished(self) -> bool:
        return False

    def close(self) -> None:
        self.stop()


def _mci(cmd: str) -> str:
    """Run one mciSendStringW command; return the response text."""
    import ctypes

    buf = ctypes.create_unicode_buffer(512)
    err = ctypes.windll.winmm.mciSendStringW(cmd, buf, len(buf) - 1, None)
    if err:
        err_buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciGetErrorStringW(err, err_buf, len(err_buf) - 1)
        raise OSError(f"MCI {err}: {err_buf.value} ({cmd})")
    return buf.value


def _short_ascii_path(path: str | Path) -> Path:
    """MCI rejects long / non-ASCII filenames (error 304); give it a short link.

    A hard link keeps the decoded WAV in place (zero copy); a copy is the
    fallback when the temp dir is on another volume.
    """
    import hashlib
    import tempfile

    src = Path(path)
    try:
        st = src.stat()
        key = f"{src}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        key = str(src)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    d = Path(tempfile.gettempdir()) / "screenlight_mci"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"trk_{digest}.wav"
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            import shutil

            shutil.copyfile(src, dst)
    return dst


class MciPlayer(_Player):
    """WAV playback through winmm MCI: play / pause / resume / seek / position.

    This is what makes the playlist UI (pause, progress, seek) possible without
    any third-party package.  Falls back to :class:`WinsoundPlayer` on failure.
    """

    supports_seek = True
    supports_pause = True

    _counter = 0

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        MciPlayer._counter += 1
        self.alias = f"sl{MciPlayer._counter}"
        self._length_ms = 0
        self._paused = False
        self._link: Path | None = None
        try:
            self._link = _short_ascii_path(self.path)
            _mci(f'open "{self._link}" type waveaudio alias {self.alias}')
            _mci(f"set {self.alias} time format milliseconds")
            self._length_ms = int(float(_mci(f"status {self.alias} length")))
            self.length = self._length_ms / 1000.0
            self.mode = "mci"
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"MCI 打开失败：{exc}"
            self.ok = False

    # -- state ------------------------------------------------------------
    def paused(self) -> bool:
        if not self.ok:
            return False
        try:
            return _mci(f"status {self.alias} mode") != "playing"
        except Exception:  # noqa: BLE001
            return self._paused

    def finished(self) -> bool:
        if not self.ok:
            return False
        try:
            mode = _mci(f"status {self.alias} mode")
        except Exception:  # noqa: BLE001
            return False
        return mode in ("stopped", "paused") and self.position() >= self.length - 0.08

    def position(self) -> float:
        if not self.ok:
            return 0.0
        try:
            return int(float(_mci(f"status {self.alias} position"))) / 1000.0
        except Exception:  # noqa: BLE001
            return 0.0

    # -- transport --------------------------------------------------------
    def play(self) -> None:
        if not self.ok:
            return
        try:
            _mci(f"seek {self.alias} to 0")
            _mci(f"play {self.alias}")
            self._paused = False
        except Exception:  # noqa: BLE001
            self.ok = False

    def stop(self) -> None:
        if self.ok:
            try:
                _mci(f"stop {self.alias}")
            except Exception:  # noqa: BLE001
                pass

    def pause(self) -> None:
        if self.ok:
            try:
                _mci(f"pause {self.alias}")
                self._paused = True
            except Exception:  # noqa: BLE001
                pass

    def resume(self) -> None:
        if self.ok:
            try:
                _mci(f"resume {self.alias}")
                self._paused = False
            except Exception:  # noqa: BLE001
                pass

    def toggle(self) -> None:
        if not self.ok:
            return
        try:
            mode = _mci(f"status {self.alias} mode").strip()
        except Exception:  # noqa: BLE001
            mode = ""
        if mode == "playing":
            self.pause()
        elif mode == "paused":
            if self.position() >= self.length - 0.08:
                self.play()          # at the very end: restart the track
            else:
                self.resume()
        else:                         # stopped / not started yet
            self.play()

    def seek(self, seconds: float) -> None:
        if not self.ok:
            return
        was_playing = not self.paused()
        ms = int(max(0.0, min(float(seconds), max(self.length - 0.05, 0.0))) * 1000)
        try:
            _mci(f"seek {self.alias} to {ms}")
            if was_playing:
                _mci(f"play {self.alias}")
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self.ok:
            try:
                _mci(f"close {self.alias}")
            except Exception:  # noqa: BLE001
                pass
            self.ok = False
        if self._link is not None:
            try:
                self._link.unlink()
            except OSError:
                pass
            self._link = None


class WinsoundPlayer(_Player):
    """Last-resort player (play/stop only, no pause/seek) using winsound."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._t0 = 0.0
        self._playing = False
        if winsound is None:
            self.reason = "当前系统不支持 winsound"
            return
        try:
            from .features import load_wav

            samples, sr = load_wav(self.path)
            self.length = len(samples) / float(sr)
        except Exception:  # noqa: BLE001
            self.length = 0.0
        self.mode = "winsound"
        self.ok = self.length > 0.2

    def play(self) -> None:
        if not self.ok:
            return
        self.stop()
        self._t0 = time.perf_counter()
        self._playing = True
        try:
            winsound.PlaySound(str(self.path), winsound.SND_FILENAME
                               | winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception:  # noqa: BLE001
            self._playing = False

    def stop(self) -> None:
        if winsound is not None:
            try:
                winsound.PlaySound(None, 0)
            except Exception:  # noqa: BLE001
                pass
        self._playing = False

    def position(self) -> float:
        if not self._playing or self.length <= 0:
            return 0.0
        return (time.perf_counter() - self._t0) % self.length

    def paused(self) -> bool:
        return not self._playing


def make_player(path: str | Path) -> _Player:
    """Best available player for a local WAV (MCI -> winsound)."""
    try:
        player = MciPlayer(path)
        if player.ok:
            return player
    except Exception:  # noqa: BLE001
        pass
    return WinsoundPlayer(path)


class InAppSource(BaseSource):
    """Plays a bundled/decoded wav and reports its own clock position.

    Zero dependencies, zero permissions, always works -> the demo safety net.
    Supports pause / resume / seek through :class:`MciPlayer`; if MCI is not
    available it degrades to winsound (play/stop only).
    """

    key = "inapp"
    label = "内置曲目"

    def __init__(self, wav_path: str | Path, duration: float) -> None:
        self.path = Path(wav_path)
        self.duration = max(duration, 0.1)
        self.player = make_player(self.path)
        self.available = self.player.ok
        if not self.player.ok:
            self.reason = getattr(self.player, "reason", "播放器不可用")
        if self.player.length > 0.2:
            self.duration = self.player.length

    # -- transport --------------------------------------------------------
    def start(self) -> None:
        if not self.available:
            return
        self.player.play()

    def stop(self) -> None:
        try:
            self.player.stop()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self.player.close()
        except Exception:  # noqa: BLE001
            pass

    def position(self) -> float:
        if not self.available:
            return 0.0
        return float(self.player.position())

    def restart(self) -> None:
        self.player.play()

    # -- extra controls used by the playlist UI ---------------------------
    def toggle_pause(self) -> None:
        if self.available and self.player.supports_pause:
            self.player.toggle()

    def pause(self) -> None:
        if self.available and self.player.supports_pause:
            self.player.pause()

    def resume(self) -> None:
        if self.available and self.player.supports_pause:
            self.player.resume()

    def is_paused(self) -> bool:
        return bool(self.available and self.player.paused())

    def seek(self, seconds: float) -> None:
        if self.available and self.player.supports_seek:
            self.player.seek(seconds)

    def supports_seek(self) -> bool:
        return bool(self.available and self.player.supports_seek)

    def supports_pause(self) -> bool:
        return bool(self.available and self.player.supports_pause)

    def finished(self) -> bool:
        return bool(self.available and self.player.finished())


class _Ring:
    def __init__(self, seconds: float = 1.0, sr: int = 44100) -> None:
        self.lock = threading.Lock()
        self.blocks: deque[np.ndarray] = deque(maxlen=max(2, int(seconds * sr / 512)))

    def push(self, block: np.ndarray) -> None:
        with self.lock:
            self.blocks.append(np.ascontiguousarray(block, dtype=np.float32))

    def latest(self, n: int = 4096) -> np.ndarray | None:
        with self.lock:
            if not self.blocks:
                return None
            data = np.concatenate(list(self.blocks))
        return data[-n:]


class _PASource(BaseSource):
    """Shared pyaudiowpatch plumbing."""

    def __init__(self) -> None:
        self.pa = None
        self.stream = None
        self.sr = 44100
        self.device_name = ""
        self.ring = _Ring(1.0, 44100)
        self._mod = _load_pyaudio()
        self.available = self._mod is not None
        if not self.available:
            self.reason = "内置音频引擎缺失（vendor/pyaudiowpatch 或 Python 版本不匹配）"

    def _open(self, index: int, channels: int, rate: int) -> None:
        pa = self._mod
        assert pa is not None
        self.pa = pa.PyAudio()
        self.sr = int(rate)
        ch = max(1, int(channels))

        def cb(in_data, frame_count, time_info, status):  # noqa: ANN001
            pcm = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1)
            self.ring.push(pcm)
            return (None, pa.paContinue)

        self.stream = self.pa.open(
            format=pa.paInt16,
            channels=ch,
            rate=self.sr,
            input=True,
            input_device_index=int(index),
            frames_per_buffer=512,
            stream_callback=cb,
        )
        self.stream.start_stream()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception:
                pass
            finally:
                self.pa = None

    def latest(self, n: int = 4096) -> np.ndarray | None:
        return self.ring.latest(n)


def _input_candidates(pa) -> list[tuple[int, int, dict, str]]:
    """All input devices, preferring WASAPI > DirectSound > MME."""
    p = pa.PyAudio()
    out: list[tuple[int, int, dict, str]] = []
    try:
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if int(d.get("maxInputChannels") or 0) <= 0:
                continue
            name = str(d.get("name", ""))
            low = name.lower()
            if "loopback" in low or "stereo mix" in low or "立体声混音" in name:
                continue
            try:
                host = str(p.get_host_api_info_by_index(d["hostApi"])["name"])
            except Exception:
                host = ""
            hu = host.upper()
            pri = 0 if "WASAPI" in hu else (1 if "DIRECTSOUND" in hu else (2 if "MME" in hu else 3))
            out.append((pri, i, d, host))
    except Exception:
        pass
    finally:
        p.terminate()
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def list_devices() -> list[dict]:
    """Diagnostics for `python -m screenlight --list-audio`."""
    pa = _load_pyaudio()
    if pa is None:
        return []
    p = pa.PyAudio()
    out: list[dict] = []
    try:
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            try:
                host = str(p.get_host_api_info_by_index(d["hostApi"])["name"])
            except Exception:
                host = ""
            out.append({
                "index": i,
                "name": d.get("name"),
                "host": host,
                "in": int(d.get("maxInputChannels") or 0),
                "out": int(d.get("maxOutputChannels") or 0),
                "rate": int(d.get("defaultSampleRate") or 0),
                "loop": bool(d.get("isLoopbackDevice")),
            })
        return out
    finally:
        p.terminate()


class MicSource(_PASource):
    key = "mic"
    label = "麦克风"

    def start(self) -> None:
        if not self.available or self.stream is not None:
            return
        pa = self._mod
        entries = _input_candidates(pa)
        if not entries:
            self.available = False
            self.reason = "没有找到可用的输入设备"
            return

        last: Exception | None = None
        tried = 0
        for _, idx, device, host in entries[:6]:
            maxch = int(device.get("maxInputChannels") or 1)
            rates = [int(device.get("defaultSampleRate") or 44100)]
            for r in (48000, 44100, 16000):
                if r not in rates:
                    rates.append(r)
            for ch in ([1] if maxch < 2 else [1, maxch]):
                for rate in rates[:3]:
                    tried += 1
                    try:
                        self._open(idx, ch, rate)
                        self.available = True
                        self.reason = ""
                        self.device_name = f"{device.get('name')} [{host}] {ch}ch {rate}Hz"
                        return
                    except Exception as exc:
                        last = exc
                        self.stop()
        self.available = False
        self.reason = f"麦克风打开失败（尝试 {tried} 种组合）：{last}"


class LoopbackSource(_PASource):
    key = "loop"
    label = "系统内录"

    def start(self) -> None:
        if not self.available or self.stream is not None:
            return
        pa = self._mod
        try:
            probe = pa.PyAudio()
            wasapi = probe.get_host_api_info_by_type(pa.paWASAPI)
            speakers = probe.get_device_info_by_index(wasapi["defaultOutputDevice"])
            device = speakers
            if not speakers.get("isLoopbackDevice", False):
                for loop in probe.get_loopback_device_info_generator():
                    if speakers["name"] in loop["name"]:
                        device = loop
                        break
                else:
                    raise RuntimeError("找不到默认扬声器的 loopback 设备")
            index = int(device["index"])
            channels = int(device.get("maxInputChannels") or 2)
            rate = int(device.get("defaultSampleRate") or 48000)
            probe.terminate()
            self._open(index, channels, rate)
        except Exception as exc:  # pragma: no cover
            self.available = False
            self.reason = f"内录打开失败: {exc}"
            self.stop()


def build_sources(wav_path: str | Path, duration: float) -> dict[str, BaseSource]:
    return {
        "inapp": InAppSource(wav_path, duration),
        "mic": MicSource(),
        "loop": LoopbackSource(),
    }
