"""Model discovery for the optional AI backends.

Models live in ``<repo>/models`` (override with ``OPENRAY_MODELS``).  They
are *optional*: every AI feature has a dependency-free fallback, so the demo
always runs.  ``tools/fetch_models.py`` downloads the open models used here.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    env = os.getenv("OPENRAY_MODELS", "").strip()
    return Path(env) if env else ROOT / "models"


@dataclass
class YamnetPaths:
    onnx: Path
    class_map: Path

    @property
    def ready(self) -> bool:
        return self.onnx.is_file() and self.onnx.stat().st_size > 1_000_000 \
            and self.class_map.is_file()


def yamnet() -> YamnetPaths:
    base = models_dir()
    return YamnetPaths(base / "yamnet.onnx", base / "yamnet_class_map.csv")


def sherpa_kws_dir() -> Path | None:
    base = models_dir()
    for candidate in sorted(base.glob("sherpa-onnx-kws-*")):
        if (candidate / "tokens.txt").is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def sapi_available() -> bool:
    """Windows built-in Chinese speech recognizer (no download needed)."""
    if os.name != "nt":
        return False
    import subprocess
    script = ("Add-Type -AssemblyName System.Speech;"
              "[System.Speech.Recognition.SpeechRecognitionEngine]::"
              "InstalledRecognizers() | ForEach-Object { $_.Culture.Name }")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            capture_output=True, text=True, timeout=20, encoding="utf-8",
            errors="replace")
        return "zh-CN" in (out.stdout or "")
    except Exception:  # noqa: BLE001 - capability probe only
        return False


def status() -> dict:
    y = yamnet()
    kws = sherpa_kws_dir()
    try:
        import onnxruntime  # noqa: F401
        ort_ok = True
    except Exception:  # noqa: BLE001
        ort_ok = False
    try:
        import sherpa_onnx  # noqa: F401
        sherpa_ok = True
    except Exception:  # noqa: BLE001
        sherpa_ok = False
    return {
        "yamnet_model": y.ready,
        "yamnet_path": str(y.onnx),
        "onnxruntime": ort_ok,
        "sherpa_kws_model": kws is not None,
        "sherpa_onnx": sherpa_ok,
        "sapi_zh": sapi_available(),
        "models_dir": str(models_dir()),
    }
