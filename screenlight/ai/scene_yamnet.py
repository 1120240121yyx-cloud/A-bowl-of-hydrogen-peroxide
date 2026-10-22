"""Scene classification: YAMNet (ONNX) with a dependency-free heuristic fallback.

Guide mapping
-------------
P0 场景识别: ''放音乐 → 跟拍；突然鼓掌 → 爆发；有人讲话 → 轻柔呼吸''

The classifier returns a coarse scene plus a confidence:

  music / speech / applause / laughter / silence / other

``YamnetClassifier`` needs ``onnxruntime`` + ``models/yamnet.onnx`` (see
``tools/fetch_models.py``).  If either is missing it raises at construction and
``SceneClassifier`` silently falls back to ``HeuristicSceneClassifier``, which
works on plain numpy from the spectral features the light engine already has.
"""
from __future__ import annotations

import csv
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models

SCENES = ("music", "speech", "applause", "laughter", "silence", "other")
SCENE_LABELS_CN = {
    "music": "音乐", "speech": "人声", "applause": "掌声",
    "laughter": "笑声", "silence": "安静", "other": "环境",
}

# AudioSet display-name keywords -> coarse scene (checked in this order)
_GROUP_KEYWORDS = (
    ("applause", ("applause", "clapping", "clap")),
    ("laughter", ("laughter", "giggle", "laugh")),
    ("speech", ("speech", "conversation", "narration", "monologue",
                "child speech", "singing")),
    ("music", ("music", "musical instrument", "piano", "guitar", "drum",
               "electronic music", "singing", "choir", "orchestra", "violin",
               "synthesizer", "pop", "rock", "hip hop", "jazz", "classical")),
    ("silence", ("silence", "quiet")),
)


def _resample(x: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    if sr == to or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    n = max(1, int(len(x) * to / sr))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def load_wav_mono16k(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
        width = w.getsampwidth()
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return _resample(data, sr), 16000


# ------------------------------------------------------------------ yamnet

class YamnetClassifier:
    name = "yamnet"
    needs_model = True

    def __init__(self, window: float = 0.975) -> None:
        paths = models.yamnet()
        if not paths.ready:
            raise FileNotFoundError(f"YAMNet model missing: {paths.onnx}")
        import onnxruntime as ort

        self.window = window
        self.session = ort.InferenceSession(
            str(paths.onnx), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes: dict[int, str] = {}
        with open(paths.class_map, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.classes[int(row["index"])] = row["display_name"]
        self._group_of = {i: self._group(name) for i, name in self.classes.items()}

    @staticmethod
    def _group(name: str) -> str | None:
        low = name.lower()
        for group, keys in _GROUP_KEYWORDS:
            if any(k in low for k in keys):
                return group
        return None

    def scores(self, samples: np.ndarray, sr: int = 16000) -> np.ndarray:
        x = _resample(np.asarray(samples, dtype=np.float32), sr)
        if len(x) < 400:
            raise ValueError("audio too short")
        out = self.session.run(None, {self.input_name: x.reshape(-1)})[0]
        return np.asarray(out, dtype=np.float32)          # [frames, 521]

    def classify(self, samples: np.ndarray, sr: int = 16000) -> dict:
        frame_scores = self.scores(samples, sr)
        mean = frame_scores.mean(axis=0)
        top = int(np.argmax(mean))
        group_scores = {g: 0.0 for g in SCENES}
        for idx, group in self._group_of.items():
            if group:
                group_scores[group] += float(mean[idx])
        # normalise group scores so they read as probabilities over the scenes
        total = sum(group_scores.values()) or 1.0
        group_scores = {g: v / total for g, v in group_scores.items()}
        best = max(group_scores, key=group_scores.get)
        if group_scores[best] < 0.18:
            best = "other"
        return {
            "scene": best,
            "confidence": float(group_scores.get(best, 0.0)),
            "top_class": self.classes.get(top, str(top)),
            "top_score": float(mean[top]),
            "scores": group_scores,
            "backend": self.name,
        }

    def classify_file(self, path: str | Path, seconds: float = 3.0) -> dict:
        samples, sr = load_wav_mono16k(path)
        return self.classify(samples[: int(sr * seconds)], sr)


# --------------------------------------------------------------- heuristic

class HeuristicSceneClassifier:
    """Cheap causal classifier used when the ONNX model is unavailable.

    Features: level, spectral flatness, zero-crossing rate, low/high band ratio
    and transient density.  Tuned to separate music / speech / applause /
    silence well enough to demo the scene-driven lighting behaviour.
    """

    name = "heuristic"
    needs_model = False

    def __init__(self, hop: float = 0.5) -> None:
        self.hop = hop

    def classify(self, samples: np.ndarray, sr: int = 16000) -> dict:
        x = _resample(np.asarray(samples, dtype=np.float32), sr)
        if len(x) < int(sr * 0.2):
            return {"scene": "silence", "confidence": 0.9,
                    "top_class": "silence", "top_score": 1.0,
                    "scores": {g: 0.0 for g in SCENES}, "backend": self.name}
        win = int(sr * 0.04)
        n = len(x) // win
        if n < 2:
            return {"scene": "other", "confidence": 0.3,
                    "top_class": "other", "top_score": 0.0,
                    "scores": {g: 0.0 for g in SCENES}, "backend": self.name}
        frames = x[: n * win].reshape(n, win)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        level = float(np.clip(np.sqrt((x ** 2).mean()) * 4.0, 0.0, 1.0))
        if level < 0.012:
            scores = {"silence": 0.92, "music": 0.02, "speech": 0.02,
                      "applause": 0.02, "laughter": 0.01, "other": 0.01}
            return {"scene": "silence", "confidence": 0.92,
                    "top_class": "silence", "top_score": 0.0,
                    "scores": scores, "backend": self.name}

        # spectral features on a 1024 window
        spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-9
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        flat = float(np.exp(np.mean(np.log(spec), axis=1)).mean()
                     / (np.mean(spec) + 1e-9))
        zcr = float((np.diff(np.sign(frames), axis=1) != 0).mean())
        low = float(spec[:, freqs < 300].sum() / (spec.sum() + 1e-9))
        high = float(spec[:, freqs > 3000].sum() / (spec.sum() + 1e-9))
        # transients: positive flux peaks over the mean
        flux = np.maximum(0.0, np.diff(rms))
        if len(flux) > 2:
            thresh = flux.mean() + 1.2 * flux.std()
            density = float((flux > thresh).mean())
        else:
            density = 0.0
        low_rms = float(rms.std() / (rms.mean() + 1e-9))

        scores = {g: 0.0 for g in SCENES}
        # applause: broadband, dense transients, little tonal centre
        scores["applause"] = np.clip(0.35 + 2.2 * density + 0.8 * high
                                     - 1.2 * low, 0.0, 1.0)
        # speech: formant-ish high flatness, mid zcr, low low-band ratio
        scores["speech"] = np.clip(0.45 + 1.6 * flat + 1.2 * zcr
                                   - 2.4 * low - 0.4 * density, 0.0, 1.0)
        # music: strong low band (kick/bass), moderate flux variance
        scores["music"] = np.clip(0.40 + 2.0 * low + 0.8 * low_rms
                                  - 1.2 * flat - 0.5 * density, 0.0, 1.0)
        scores["laughter"] = np.clip(0.25 + density * 1.2 + flat * 0.8, 0.0, 1.0)
        scores["silence"] = np.clip(0.5 - level * 2.0, 0.0, 1.0)
        scores["other"] = 0.15
        total = sum(scores.values()) or 1.0
        scores = {g: v / total for g, v in scores.items()}
        best = max(scores, key=scores.get)
        return {"scene": best, "confidence": float(scores[best]),
                "top_class": best, "top_score": float(scores[best]),
                "scores": scores, "backend": self.name}


class SceneClassifier:
    """``auto`` -> YAMNet when its model + onnxruntime exist, else heuristic."""

    def __init__(self, requested: str = "auto") -> None:
        self.requested = requested
        self.impl = None
        self.failures: list[str] = []
        if requested in ("auto", "yamnet"):
            try:
                self.impl = YamnetClassifier()
            except Exception as exc:  # noqa: BLE001 - fallback is the feature
                self.failures.append(f"yamnet: {type(exc).__name__}: {exc}")
        if self.impl is None:
            self.impl = HeuristicSceneClassifier()

    @property
    def name(self) -> str:
        return self.impl.name

    @property
    def available(self) -> bool:
        return self.impl is not None

    def classify(self, samples: np.ndarray, sr: int = 16000) -> dict:
        return self.impl.classify(samples, sr)

    def classify_file(self, path, seconds: float = 3.0) -> dict:
        samples, sr = load_wav_mono16k(path)
        return self.impl.classify(samples[: int(sr * seconds)], sr)


# ------------------------------------------------------------------ timeline

@dataclass
class SceneSegment:
    start: float
    end: float
    scene: str
    confidence: float


@dataclass
class SceneTimeline:
    segments: list[SceneSegment] = field(default_factory=list)
    backend: str = "heuristic"
    duration: float = 0.0

    def label_at(self, t: float) -> tuple[str, float]:
        for seg in self.segments:
            if seg.start <= t < seg.end:
                return seg.scene, seg.confidence
        return "other", 0.0

    def summary(self) -> dict:
        out: dict[str, float] = {}
        for seg in self.segments:
            out[seg.scene] = out.get(seg.scene, 0.0) + (seg.end - seg.start)
        return {k: round(v, 1) for k, v in sorted(out.items(),
                                                  key=lambda kv: -kv[1])}

    @classmethod
    def build(cls, samples: np.ndarray, sr: int, classifier: SceneClassifier,
              hop: float = 1.0, window: float = 1.8,
              max_windows: int = 90) -> "SceneTimeline":
        x = _resample(np.asarray(samples, dtype=np.float32), sr)
        total = len(x) / 16000.0
        if total <= 0:
            return cls(backend=classifier.name)
        step = max(hop, total / max_windows)
        segments: list[SceneSegment] = []
        t = 0.0
        while t < total:
            a = int(t * 16000)
            b = min(len(x), a + int(window * 16000))
            if b - a < 4000:
                break
            try:
                res = classifier.classify(x[a:b], 16000)
                scene = res["scene"]
                conf = res["confidence"]
            except Exception:  # noqa: BLE001
                scene, conf = "other", 0.0
            segments.append(SceneSegment(t, min(total, t + step), scene, conf))
            t += step
        # merge consecutive identical scenes
        merged: list[SceneSegment] = []
        for seg in segments:
            if merged and merged[-1].scene == seg.scene:
                merged[-1] = SceneSegment(merged[-1].start, seg.end,
                                          seg.scene,
                                          max(merged[-1].confidence, seg.confidence))
            else:
                merged.append(seg)
        return cls(segments=merged, backend=classifier.name, duration=total)


# scene -> response multiplier used by app.py (accent / beat feel)
SCENE_ACCENT = {
    "music": 1.0,
    "applause": 1.8,
    "laughter": 1.25,
    "speech": 0.35,
    "silence": 0.0,
    "other": 0.8,
}
