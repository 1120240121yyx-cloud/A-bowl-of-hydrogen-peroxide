"""Beat tracking by dynamic programming.

Algorithm: Ellis, D. P. W. (2007),
"Beat Tracking by Dynamic Programming",
Journal of New Music Research, 36(1), 51-60.

This is the same family of algorithm used by librosa.beat.beat_track
(librosa is ISC licensed).  Re-implemented here in NumPy so OpenRay
stays dependency-free and can be shipped as a single folder.
"""
from __future__ import annotations

import numpy as np


def onset_strength(spec: np.ndarray, sr: int, frame: int, n_bands: int = 32) -> np.ndarray:
    """Log-compressed multi-band spectral flux (a la SuperFlux / librosa)."""
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    top = min(14000.0, sr * 0.45)
    edges = np.geomspace(40.0, top, n_bands + 1)

    bands = np.zeros((spec.shape[0], n_bands), dtype=np.float32)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            bands[:, i] = spec[:, mask].mean(axis=1)

    logb = np.log1p(10.0 * bands / (float(bands.max()) + 1e-9))
    flux = np.zeros(len(spec), dtype=np.float32)
    flux[1:] = np.maximum(0.0, np.diff(logb, axis=0)).sum(axis=1)

    flux = np.maximum(flux - float(np.median(flux)), 0.0)
    scale = float(flux.std())
    return (flux / (scale if scale > 1e-6 else 1.0)).astype(np.float32)


def track_beats(onset: np.ndarray, fps: float, bpm: float,
                tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis 2007). Returns frame indices."""
    n = len(onset)
    if n < 8 or bpm <= 0:
        return np.zeros(0, dtype=np.int64)

    period = max(2.0, 60.0 / bpm * fps)

    # local score: small Gaussian smoothing around each onset
    sigma = max(1.0, period / 32.0)
    radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    local = np.convolve(onset, g, mode="same")

    # candidate previous beats: from -2 periods to -0.5 period
    lo = int(round(-2.0 * period))
    hi = int(round(-0.5 * period))
    if hi <= lo:
        hi = lo + 1
    offsets = np.arange(lo, hi + 1)
    txcost = -tightness * (np.log(-offsets / period) ** 2)

    cum = np.zeros(n, dtype=np.float64)
    back = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        idx = i + offsets
        valid = idx >= 0
        if not np.any(valid):
            cum[i] = local[i]
            continue
        scores = cum[idx[valid]] + txcost[valid]
        j = int(np.argmax(scores))
        back[i] = idx[valid][j]
        cum[i] = local[i] + scores[j]

    # backtrack from the strongest endpoint
    end = int(np.argmax(cum))
    beats: list[int] = []
    i = end
    guard = 0
    while i >= 0 and guard <= n:
        beats.append(int(i))
        i = int(back[i])
        guard += 1
    beats.reverse()

    arr = np.array(beats, dtype=np.int64)
    if arr.size == 0:
        return arr

    # trim weak leading / trailing beats
    thresh = 0.05 * float(np.percentile(local, 95))
    start, stop = 0, len(arr)
    while start < stop and local[arr[start]] < thresh:
        start += 1
    while stop > start and local[arr[stop - 1]] < thresh:
        stop -= 1
    return arr[start:stop]


def refine_beats(beats: np.ndarray, onset: np.ndarray, fps: float,
                 radius: float = 0.06) -> np.ndarray:
    """Snap each tracked beat to the nearest onset peak (+/- radius seconds)."""
    if beats.size == 0:
        return beats
    r = max(1, int(radius * fps))
    out = []
    for b in beats:
        lo = max(0, int(b) - r)
        hi = min(len(onset), int(b) + r + 1)
        if hi <= lo:
            out.append(int(b))
            continue
        out.append(lo + int(np.argmax(onset[lo:hi])))
    return np.array(out, dtype=np.int64)


def pulse_envelope(beats: np.ndarray, n_frames: int, fps: float,
                   tau: float = 0.085, accent_every: int = 4) -> np.ndarray:
    """Exponential pulse on every beat; every 4th beat gets an accent."""
    env = np.zeros(n_frames, dtype=np.float32)
    if beats.size == 0:
        return env
    length = max(1, int(tau * 4.0 * fps))
    t = np.arange(length, dtype=np.float32) / fps
    decay = np.exp(-t / tau)
    for k, b in enumerate(beats):
        amp = 1.0 if (k % accent_every == 0) else 0.72
        end = min(n_frames, int(b) + length)
        if end <= b:
            continue
        env[int(b):end] = np.maximum(env[int(b):end], amp * decay[: end - int(b)])
    return env


def beat_stats(beats: np.ndarray, fps: float) -> tuple[float, float, float]:
    """(count, median interval seconds, interval std seconds)."""
    if beats.size < 2:
        return float(beats.size), 0.0, 0.0
    intervals = np.diff(beats) / fps
    return float(beats.size), float(np.median(intervals)), float(intervals.std())
