"""Musical emotion features: valence (mood colour) and arousal (intensity).

This is the part that makes two songs with the *same BPM* look different: the
brightness contour and colour temperature are driven by continuous valence /
arousal estimates, not just by loudness.

Features are classic MIR descriptors, all vectorised over the STFT:

  valence : major/minor mode (Krumhansl profiles), spectral brightness,
            harmonicity / tonalness, low-mid warmth
  arousal : tempo, spectral centroid, high-band ratio, dynamics, transient
            density, loudness

Everything is numpy-only and causal enough for the live path (<2 s window).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66,
                  2.29, 2.88], dtype=np.float32)
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69,
                  3.34, 3.17], dtype=np.float32)
_MINOR_K = np.array([0, 2, 3, 5, 7, 8, 10], dtype=int)

EMOTION_LABELS = {
    "bright": "明亮",
    "warm": "温暖",
    "intense": "激烈",
    "sad": "忧伤",
    "balanced": "平衡",
}


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (v.sum(axis=-1, keepdims=True) + 1e-9)


@lru_cache(maxsize=4)
def _chroma_weight(sr: int, frame: int, lo_hz: float = 55.0,
                   hi_hz: float = 2400.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    n = freqs.size
    w = np.zeros((n, 12), dtype=np.float32)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return w
    f = freqs[mask]
    pc = np.round(12.0 * np.log2(f / 440.0 + 1e-9)).astype(int) % 12
    # soft weighting around A440 reference (log-frequency window)
    w[np.where(mask)[0], pc] = 1.0
    return w


@lru_cache(maxsize=2)
def _profiles() -> tuple[np.ndarray, np.ndarray]:
    out_m = np.zeros((12, 12), dtype=np.float32)
    out_n = np.zeros((12, 12), dtype=np.float32)
    for k in range(12):
        out_m[k] = _norm(np.roll(MAJOR, k))
        out_n[k] = _norm(np.roll(MINOR, k))
    return out_m, out_n


def chroma(spec: np.ndarray, sr: int, frame: int) -> np.ndarray:
    """spec (T, bins) magnitude -> (T, 12) normalised chroma."""
    w = _chroma_weight(int(sr), int(frame))
    c = np.asarray(spec, dtype=np.float32) @ w
    return _norm(c)


def _mode_valence(chroma_n: np.ndarray) -> np.ndarray:
    pm, pn = _profiles()
    ch = chroma_n / (np.linalg.norm(chroma_n, axis=1, keepdims=True) + 1e-9)
    pm = pm / (np.linalg.norm(pm, axis=1, keepdims=True) + 1e-9)
    pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-9)
    major = np.max(ch @ pm.T, axis=1)
    minor = np.max(ch @ pn.T, axis=1)
    diff = (major - minor) / (major + minor + 1e-9)
    # the absolute major/minor cosine gap is small (~0.03-0.10), so amplify it
    return np.tanh(9.0 * diff).astype(np.float32)      # -1..1


def _smooth(x: np.ndarray, fps: float, ms: float) -> np.ndarray:
    sigma = max(1.0, fps * ms / 1000.0)
    radius = int(min(3 * sigma, len(x) / 3))
    if radius < 1 or len(x) < 3:
        return x
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    # normalise by the convolved window: plain 'same' convolution zero-pads and
    # drags short active segments towards zero (this flattened happy vs sad)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float32), k, mode="same")
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


def _rolling_std(x: np.ndarray, fps: float, sec: float = 2.0) -> np.ndarray:
    n = max(3, int(fps * sec))
    c1 = np.convolve(x, np.ones(n, dtype=np.float32) / n, mode="same")
    c2 = np.convolve(x * x, np.ones(n, dtype=np.float32) / n, mode="same")
    return np.sqrt(np.maximum(c2 - c1 * c1, 0.0)).astype(np.float32)


def emotion_from_spec(spec: np.ndarray, rms: np.ndarray, fps: float, sr: int,
                      frame: int, bpm: float | np.ndarray | None = None,
                      ) -> dict[str, np.ndarray]:
    """Return valence / arousal arrays in 0..1 (same rate as ``spec``).

    Features are pooled to ~4 Hz before analysis: emotion changes slowly, and
    the pooled chroma/spectrum is much more stable than per-STFT-frame values
    (per-frame mode estimates flipped between major and minor on single notes).
    """
    spec = np.asarray(spec, dtype=np.float32)
    rms = np.asarray(rms, dtype=np.float32)
    t = spec.shape[0]
    if t == 0:
        return {"valence": np.zeros(0, np.float32),
                "arousal": np.zeros(0, np.float32)}
    fps = max(float(fps), 1.0)
    step = max(1, int(round(fps / 4.0)))
    td = t // step
    if td < 2:
        return {"valence": np.full(t, 0.5, np.float32),
                "arousal": np.full(t, 0.5, np.float32)}
    binned = spec[: td * step].reshape(td, step, -1).mean(axis=1)
    rms_d = rms[: td * step].reshape(td, step).mean(axis=1)
    fps_d = fps / step
    freqs = np.fft.rfftfreq(frame, 1.0 / sr).astype(np.float32)
    total = binned.sum(axis=1) + 1e-9
    cent = (binned * freqs).sum(axis=1) / total
    brightness = np.clip(np.log2(np.maximum(cent, 80.0) / 150.0) / 4.0, 0.0, 1.0)
    high = binned[:, freqs > 3500.0].sum(axis=1) / total
    low = binned[:, (freqs >= 60.0) & (freqs < 250.0)].sum(axis=1) / total
    mid = binned[:, (freqs >= 250.0) & (freqs < 2500.0)].sum(axis=1) / total
    ch = chroma(binned, sr, frame)
    harmonicity = np.clip(np.max(ch, axis=1) * 2.0, 0.0, 1.0)
    tonal = _mode_valence(ch)

    active = rms_d > max(1e-4, 0.06 * float(np.percentile(rms_d, 90)))
    if not np.any(active):
        return {"valence": np.full(t, 0.5, np.float32),
                "arousal": np.full(t, 0.5, np.float32)}
    idx = np.flatnonzero(active)

    rms_n = np.clip(rms_d / (np.percentile(rms_d, 90) + 1e-9), 0.0, 1.0)
    dyn = np.clip(_rolling_std(rms_n, fps_d, 2.0) / 0.25, 0.0, 1.0)
    flux = np.maximum(0.0, np.diff(rms_n, prepend=rms_n[0]))
    trans = (flux > (np.percentile(flux, 85) + 1e-6)).astype(np.float32)
    trans = _smooth(trans, fps_d, 700.0)

    if bpm is None:
        tempo = np.full(td, 0.45, dtype=np.float32)
    elif np.isscalar(bpm):
        tempo = np.full(td, np.clip(float(bpm) / 170.0, 0.0, 1.0), dtype=np.float32)
    else:
        b = np.asarray(bpm, dtype=np.float32)
        b = np.interp(np.linspace(0, b.size - 1, t), np.arange(b.size), b)
        tempo = np.clip(np.resize(b[: td * step].reshape(td, step).mean(1), td)
                        / 170.0, 0.0, 1.0)

    valence = np.clip(
        0.55 * (0.5 + 0.5 * tonal) + 0.15 * brightness
        + 0.15 * harmonicity + 0.15 * (0.55 * low + 0.45 * mid), 0.0, 1.0)
    arousal = np.clip(
        0.30 * tempo + 0.20 * brightness + 0.15 * high
        + 0.15 * dyn + 0.12 * trans + 0.08 * rms_n, 0.0, 1.0)

    # mild contrast expansion: the raw descriptors are conservative, and the
    # demo should make major/minor + bright/dark clearly visible
    valence = np.clip(0.5 + (valence - 0.5) * 1.35, 0.0, 1.0)
    arousal = np.clip(0.5 + (arousal - 0.5) * 1.25, 0.0, 1.0)
    # smooth on the active samples only, then hold the mood through silence
    v_s = _smooth(valence[idx], fps_d, 650.0)
    a_s = _smooth(arousal[idx], fps_d, 500.0)
    v_d = np.interp(np.arange(td), idx, v_s).astype(np.float32)
    a_d = np.interp(np.arange(td), idx, a_s).astype(np.float32)
    grid = (np.arange(td, dtype=np.float32) + 0.5) * step
    out_v = np.interp(np.arange(t), grid, v_d).astype(np.float32)
    out_a = np.interp(np.arange(t), grid, a_d).astype(np.float32)
    return {"valence": out_v, "arousal": out_a}


def classify(valence: float, arousal: float) -> str:
    if valence >= 0.52 and arousal >= 0.48:
        return "bright"
    if valence >= 0.52:
        return "warm"
    if arousal >= 0.48:
        return "intense"
    return "sad" if valence < 0.43 else "balanced"


def label_cn(valence: float, arousal: float) -> str:
    return EMOTION_LABELS[classify(valence, arousal)]
