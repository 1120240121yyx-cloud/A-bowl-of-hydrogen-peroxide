"""Small, dependency-free DSP helpers shared by the new engines.

The classic path (``features.py`` / ``beat.py``) is intentionally left alone;
this module is the toolbox for the v3 light engine and future analyzers.
"""
from __future__ import annotations

import numpy as np


def frames_view(samples: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """(n_frames, frame) strided view with zero tail padding."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    count = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def magnitude(frames: np.ndarray, frame: int) -> np.ndarray:
    win = np.hanning(frame).astype(np.float32)
    return np.abs(np.fft.rfft(frames * win[None, :], axis=1)).astype(np.float32)


def rms_of(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)


def band_edges(sr: int, frame: int, n_bands: int, lo: float = 40.0,
               hi: float | None = None) -> np.ndarray:
    top = min(hi if hi else sr * 0.45, sr * 0.45)
    return np.geomspace(lo, top, n_bands + 1)


def band_matrix(spec: np.ndarray, sr: int, frame: int, n_bands: int = 32,
                lo: float = 40.0, hi: float | None = None) -> np.ndarray:
    """Geometric band means (log-spaced) -> (n_frames, n_bands)."""
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = band_edges(sr, frame, n_bands, lo, hi)
    out = np.zeros((spec.shape[0], n_bands), dtype=np.float32)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            out[:, i] = spec[:, mask].mean(axis=1)
    return out


def band_flux(spec: np.ndarray, sr: int, frame: int, lo: float, hi: float,
              log_compress: bool = True) -> np.ndarray:
    """Positive spectral flux restricted to [lo, hi).

    Low band (40-260 Hz) is dominated by kick/bass, which is what the beat grid
    should lock to; hi-hats and melody onsets live higher and must not move the
    phase (the classic "offbeat lock" failure).
    """
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(spec.shape[0], dtype=np.float32)
    sub = spec[:, mask]
    if log_compress:
        sub = np.log1p(10.0 * sub / (float(sub.max()) + 1e-9))
    d = np.diff(sub, axis=0, prepend=sub[:1])
    flux = np.maximum(d, 0.0).sum(axis=1)
    return flux.astype(np.float32)


def smooth_gauss(x: np.ndarray, fps: float, sigma_ms: float) -> np.ndarray:
    sigma = max(1e-3, sigma_ms / 1000.0) * fps
    r = int(3 * sigma)
    if r < 1:
        return np.asarray(x, dtype=np.float32).copy()
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # edge-pad so the result always has len(x); np.convolve(mode="same") returns
    # max(len(x), len(k)) and broke short tracks (<~4 s) with a shape mismatch
    xp = np.pad(np.asarray(x, dtype=np.float64), r, mode="edge")
    y = np.convolve(xp, k, mode="same")
    return y[r:r + len(xp) - 2 * r].astype(np.float32)


def normalize(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-6:
        b = a + 1e-6
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def envelope(x: np.ndarray, fps: float, attack_ms: float, release_ms: float,
             init: float = 0.0) -> np.ndarray:
    """Asymmetric one-pole follower (fast up, slow down)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    a = float(np.exp(-1.0 / max(fps * attack_ms / 1000.0, 1e-3)))
    d = float(np.exp(-1.0 / max(fps * release_ms / 1000.0, 1e-3)))
    y = np.empty_like(x)
    prev = float(init)
    for i in range(x.size):
        c = a if x[i] > prev else d
        prev = c * prev + (1.0 - c) * x[i]
        y[i] = prev
    return y.astype(np.float32)


def ar_step(prev: float, target: float, dt: float, attack_ms: float,
            release_ms: float) -> float:
    """Single-step version of :func:`envelope` for per-frame use."""
    ms = attack_ms if target > prev else release_ms
    tau = max(ms, 0.5) / 1000.0
    c = float(np.exp(-max(dt, 1e-4) / tau))
    return c * prev + (1.0 - c) * target


def rolling_percentile(values: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Causal rolling percentile (vectorised, O(n log window))."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.size
    if n == 0 or window <= 1:
        return v.astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i + 1 - window)
        out[i] = np.percentile(v[lo:i + 1], pct)
    return out


def autocorr_peak(x: np.ndarray, lo_lag: int, hi_lag: int) -> tuple[int, float]:
    """Best lag in [lo, hi] and its normalised height (0..1)."""
    e = np.asarray(x, dtype=np.float64)
    e = e - e.mean()
    if e.size <= hi_lag + 2:
        return 0, 0.0
    ac = np.correlate(e, e, mode="full")[e.size - 1:]
    base = float(ac[0]) + 1e-9
    lo = max(2, int(lo_lag))
    hi = min(int(hi_lag), ac.size - 1)
    if hi <= lo:
        return 0, 0.0
    seg = ac[lo:hi + 1]
    lag = lo + int(np.argmax(seg))
    return lag, float(np.clip(ac[lag] / base, 0.0, 1.0))
