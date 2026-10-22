"""Voice wake / keyword spotting (P0 in the guide: ``喊「亮起来」→点亮``).

Two offline backends, both optional at runtime:

* **sherpa-onnx KWS** — 3M zipformer zh/en, custom keyword file.  The keyword
  ``亮起来`` is written as pinyin tokens (``l iàng q ǐ l ái``); English
  ``LIGHT UP`` also works.  Verified here by synthesising the phrase with the
  Windows Chinese TTS voice and feeding it back into the spotter.
* **Windows SAPI (System.Speech)** — built into Windows, Chinese recogniser
  ``MS-2052-80-DESK`` with a tiny grammar.  No download at all; also verified
  with a TTS-generated wav.  Used for live mic when sherpa's model is missing.

Live mode: sherpa reads blocks from :class:`screenlight.audio.MicSource`
(pyaudiowpatch is vendored), SAPI uses the default audio device itself.
There is no physical mic on the dev machine, so live mode is exercised only by
the WAV path in tests; the wiring is identical.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models

# phrase -> intent used by app.py
KEYWORD_ACTIONS = {
    "小茂": "wake",
    "亮起来": "wake",
    "light up": "wake",
    "lightup": "wake",
    "画布": "canvas",
    "打开画布": "canvas",
    "温柔一点": "gentle",
    "亮一点": "brighter",
    "赛博朋克": "cyber",
    "跟着鼓点": "follow_beat",
    "关掉护栏": "unsafe",
}

# pinyin token spelling for the sherpa keyword file (phones are in tokens.txt)
SHERPA_PHRASES = {
    "小茂": "x iǎo m ào",
    "亮起来": "l iàng q ǐ l ái",
    "LIGHT UP": "L AY1 T AH1 P",
}


def phrase_tokens() -> str:
    # sherpa keyword labels cannot contain spaces: LIGHT UP -> @LIGHT_UP
    lines = [f"{tokens} @{label.replace(' ', '_')}"
             for label, tokens in SHERPA_PHRASES.items()]
    return "\n".join(lines) + "\n"


def keywords_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(phrase_tokens(), encoding="utf-8")
    return path


@dataclass
class WakeEvent:
    text: str
    confidence: float
    backend: str
    at: float


# ------------------------------------------------------------------ sherpa

class SherpaKws:
    name = "sherpa"

    def __init__(self) -> None:
        import sherpa_onnx  # noqa: F401
        d = models.sherpa_kws_dir()
        if d is None:
            raise FileNotFoundError("sherpa KWS model missing")
        import sherpa_onnx

        self.model_dir = d
        kw = keywords_file(models.models_dir() / "kws_keywords.txt")
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(d / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
            joiner=str(d / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=str(kw),
            num_threads=1,
            sample_rate=16000,
            keywords_threshold=0.12,
        )

    def recognize_wav(self, path: str | Path) -> list[WakeEvent]:
        from .scene_yamnet import load_wav_mono16k

        x, sr = load_wav_mono16k(path)
        stream = self.kws.create_stream()
        stream.accept_waveform(16000, x)
        stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
        stream.input_finished()
        out: list[WakeEvent] = []
        while self.kws.is_ready(stream):
            self.kws.decode_stream(stream)
            res = self.kws.get_result(stream)
            if res:
                for word in str(res).split("/"):
                    if word:
                        out.append(WakeEvent(word, 1.0, self.name, time.time()))
                self.kws.reset_stream(stream)
        return out

    def start_live(self, on_event, stop_event: threading.Event) -> None:
        from ..audio import MicSource

        mic = MicSource()
        mic.start()
        if not mic.available:
            raise RuntimeError(f"microphone unavailable: {mic.reason}")
        stream = self.kws.create_stream()
        pending = np.zeros(0, dtype=np.float32)
        try:
            while not stop_event.is_set():
                block = mic.latest(4096)
                if block is None:
                    time.sleep(0.02)
                    continue
                x = np.asarray(block, dtype=np.float32)
                if mic.sr and mic.sr != 16000:
                    n = max(1, int(len(x) * 16000 / mic.sr))
                    x = np.interp(np.linspace(0, len(x) - 1, n),
                                  np.arange(len(x)), x).astype(np.float32)
                pending = np.concatenate([pending, x])
                while len(pending) >= 1600:
                    stream.accept_waveform(16000, pending[:1600])
                    pending = pending[1600:]
                    while self.kws.is_ready(stream):
                        self.kws.decode_stream(stream)
                        res = self.kws.get_result(stream)
                        if res:
                            for word in str(res).split("/"):
                                if word:
                                    on_event(WakeEvent(word, 1.0, self.name,
                                                       time.time()))
                            self.kws.reset_stream(stream)
        finally:
            try:
                mic.stop()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ SAPI

_SAPI_WORDS = ("小茂", "亮起来", "跟着鼓点", "温柔一点", "亮一点",
               "赛博朋克", "打开画布", "关掉护栏")

_SAPI_TEMPLATE = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
Add-Type -AssemblyName System.Globalization
$ci = [System.Globalization.CultureInfo]::GetCultureInfo("zh-CN")
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine($ci)
$choices = New-Object System.Speech.Recognition.Choices
foreach ($w in @(__WORDS__)) { $choices.Add($w) }
$gb = New-Object System.Speech.Recognition.GrammarBuilder
$gb.Culture = $ci
$gb.Append($choices)
$rec.LoadGrammar((New-Object System.Speech.Recognition.Grammar($gb)))
$rec.add_SpeechRecognized({
  param($sender, $e)
  if ($e.Result) {
    [Console]::Out.WriteLine("RESULT|" + $e.Result.Text + "|" + $e.Result.Confidence)
    [Console]::Out.Flush()
  }
})
__INPUT__
if (__LIVE__) {
  $rec.SetInputToDefaultAudioDevice()
  $rec.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
  $parentPid = __PARENT_PID__
  while ($true) {
    # if the Python parent disappears (crash/force quit), release the mic
    if (-not (Get-Process -Id $parentPid -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 150
  }
} else {
  $rec.SetInputToWaveFile("__WAV__")
  while ($true) {
    $r = $rec.Recognize()
    if ($r -eq $null) { break }
    [Console]::Out.WriteLine("RESULT|" + $r.Text + "|" + $r.Confidence)
    [Console]::Out.Flush()
  }
}
[Console]::Out.WriteLine("DONE")
[Console]::Out.Flush()
"""


class SapiKws:
    name = "sapi"

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("SAPI is Windows-only")
        if not models.sapi_available():
            raise RuntimeError("no zh-CN SAPI recognizer")

    def _script(self, wav: str | None) -> Path:
        words = ",".join(f'"{w}"' for w in _SAPI_WORDS)
        script = _SAPI_TEMPLATE.replace("__WORDS__", words)
        script = script.replace("__LIVE__", "0" if wav else "1")
        script = script.replace("__WAV__", wav or "")
        script = script.replace("__INPUT__", "")
        script = script.replace("__PARENT_PID__", str(os.getpid()))
        fd, path = tempfile.mkstemp(prefix="screenlight_sapi_", suffix=".ps1")
        os.close(fd)
        Path(path).write_text(script, encoding="utf-8-sig")
        return Path(path)

    def recognize_wav(self, path: str | Path, timeout: float = 25.0
                      ) -> list[WakeEvent]:
        script = self._script(str(Path(path).resolve()))
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(script)],
                capture_output=True, timeout=timeout, encoding="utf-8",
                errors="replace")
        finally:
            script.unlink(missing_ok=True)
        events: list[WakeEvent] = []
        for line in (out.stdout or "").splitlines():
            if line.startswith("RESULT|"):
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        conf = float(parts[2])
                    except ValueError:
                        conf = 0.0
                    events.append(WakeEvent(parts[1].strip(), conf, self.name,
                                            time.time()))
        return events

    def start_live(self, on_event, stop_event: threading.Event) -> None:
        script = self._script(None)
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self._proc = proc
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if stop_event.is_set():
                    break
                if line.startswith("RESULT|"):
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        try:
                            conf = float(parts[2])
                        except ValueError:
                            conf = 0.0
                        on_event(WakeEvent(parts[1].strip(), conf, self.name,
                                           time.time()))
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
            script.unlink(missing_ok=True)

    def stop(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ facade

class WakeWordListener:
    """Background wake-word listener with a callback for each recognised phrase."""

    def __init__(self, on_event, requested: str = "auto") -> None:
        self.on_event = on_event
        self.requested = requested
        self.backend = None
        self.failures: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: "queue.Queue[WakeEvent]" = queue.Queue()

        order = ["sherpa", "sapi"] if requested == "auto" else [requested]
        for name in order:
            try:
                self.backend = SherpaKws() if name == "sherpa" else SapiKws()
                break
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if self.backend is None:
            raise RuntimeError("; ".join(self.failures) or "no KWS backend")

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", "none")

    def start(self) -> None:
        def run() -> None:
            try:
                self.backend.start_live(self._deliver, self._stop)
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"live: {type(exc).__name__}: {exc}")
                # sherpa needs an input device; if it is missing, try the
                # Windows recognizer before giving up (it can share the mic)
                if self.backend.name == "sherpa" and not self._stop.is_set():
                    try:
                        self.backend = SapiKws()
                        self.backend.start_live(self._deliver, self._stop)
                    except Exception as exc2:  # noqa: BLE001
                        self.failures.append(
                            f"sapi live: {type(exc2).__name__}: {exc2}")

        self._thread = threading.Thread(target=run, name="screenlight-kws",
                                        daemon=True)
        self._thread.start()

    def _deliver(self, event: WakeEvent) -> None:
        self.events.put(event)
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._stop.set()
        try:
            stop = getattr(self.backend, "stop", None)
            if callable(stop):
                stop()
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def poll(self) -> WakeEvent | None:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None


def action_for(text: str) -> str | None:
    low = text.lower().strip()
    for keyword, action in KEYWORD_ACTIONS.items():
        if keyword in low:
            return action
    return None


def recognize_wav(path: str | Path, requested: str = "auto") -> list[WakeEvent]:
    """One-shot WAV recognition (used by ``--kws-wav`` and the selftest)."""
    errors: list[str] = []
    order = ["sherpa", "sapi"] if requested == "auto" else [requested]
    for name in order:
        try:
            impl = SherpaKws() if name == "sherpa" else SapiKws()
            return impl.recognize_wav(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def status() -> dict:
    st = models.status()
    return {
        "sherpa": st["sherpa_onnx"] and st["sherpa_kws_model"],
        "sapi": st["sapi_zh"],
        "keywords": list(SHERPA_PHRASES),
    }


