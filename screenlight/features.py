"""Audio loading + feature extraction (RMS / spectral flux / tempo / scene)."""
from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .beat import onset_strength, pulse_envelope, refine_beats, track_beats
from .config import Policy


# ---------------------------------------------------------------- loading

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM wav file -> (mono float32, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        channels = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(n)

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


# ---------------------------------------------------------------- helpers

def _normalize(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    if x.size == 0:
        return x
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-6:
        b = a + 1e-6
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def _envelope(x: np.ndarray, fps: float, attack_ms: float, decay_ms: float) -> np.ndarray:
    """One-pole attack / decay follower."""
    a = float(np.exp(-1.0 / max(fps * attack_ms / 1000.0, 1e-3)))
    d = float(np.exp(-1.0 / max(fps * decay_ms / 1000.0, 1e-3)))
    y = np.empty_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        c = a if v > prev else d
        prev = c * prev + (1.0 - c) * float(v)
        y[i] = prev
    return y


def _band_energy(mag: np.ndarray, sr: int, frame: int, n_bands: int = 24) -> np.ndarray:
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    top = min(14000.0, sr * 0.45)
    edges = np.geomspace(40.0, top, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    # spectral tilt: lift the highs so the bars do not collapse into a ramp
    tilt = (centers / 1000.0) ** 0.55
    out = np.zeros((mag.shape[0], n_bands), dtype=np.float32)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            out[:, i] = mag[:, mask].mean(axis=1)
    return out * tilt[None, :]


def _frames(samples: np.ndarray, frame: int, hop: int):
    if len(samples) < frame:
        samples = np.pad(samples, (0, frame - len(samples)))
    count = 1 + (len(samples) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return samples[idx]


# ---------------------------------------------------------------- tempo / scene

def estimate_bpm(onset: np.ndarray, fps: float) -> float:
    """Autocorrelation with a log-tempo prior centred on 120 BPM."""
    if onset.size < 8 or float(onset.std()) < 1e-6:
        return 0.0
    x = onset - onset.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo = max(1, int(fps * 60.0 / 180.0))
    hi = min(int(fps * 60.0 / 50.0), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lags = np.arange(lo, hi)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.6) ** 2)
    lag = int(lags[int(np.argmax(ac[lo:hi] * prior))])
    bpm = 60.0 * fps / max(lag, 1)
    # octave correction: keep the result inside a musical range
    while bpm < 90.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return float(bpm)


def classify_scenes(samples: np.ndarray, sr: int, win: float = 1.0, hop: float = 0.5):
    """Cheap 3-way heuristic: music / speech / applause.

    v1 heuristic. v2 replaces this with YAMNet (see docs/ARCHITECTURE.md).
    """
    n = int(win * sr)
    h = max(1, int(hop * sr))
    out: list[tuple[float, str]] = []
    for start in range(0, max(1, len(samples) - n + 1), h):
        seg = samples[start:start + n]
        if seg.size < n:
            break
        zcr = float(np.mean(np.abs(np.diff(np.signbit(seg).astype(np.int8)))))
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) + 1e-9
        flat = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
        if flat > 0.32 and zcr > 0.22 and rms > 0.01:
            label = "applause"
        elif zcr < 0.14 and flat < 0.22:
            label = "speech"
        else:
            label = "music"
        out.append((start / sr, label))
    return out


# ---------------------------------------------------------------- offline

@dataclass
class Analysis:
    fps: float
    level: np.ndarray
    bands: np.ndarray
    bpm: float
    scenes: list[tuple[float, str]] = field(default_factory=list)
    duration: float = 0.0
    sr: int = 44100
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def level_at(self, t: float) -> float:
        if self.level.size == 0:
            return 0.0
        i = int(t * self.fps) % self.level.size
        return float(self.level[i])

    def bands_at(self, t: float) -> np.ndarray:
        if self.bands.shape[0] == 0:
            return np.zeros(24, dtype=np.float32)
        i = int(t * self.fps) % self.bands.shape[0]
        return self.bands[i]

    def scene_at(self, t: float) -> str:
        label = "music"
        for start, name in self.scenes:
            if start <= t:
                label = name
            else:
                break
        return label


def analyze_offline(samples: np.ndarray, sr: int, policy: Policy,
                    frame: int = 1024, hop: int = 256) -> Analysis:
    frames = _frames(samples, frame, hop)
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame).astype(np.float32), axis=1))
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    flux = np.zeros(len(spec), dtype=np.float32)
    if len(spec) > 1:
        flux[1:] = np.maximum(0.0, np.diff(spec, axis=0)).sum(axis=1)

    fps = sr / hop
    rms_n = _normalize(rms)
    flux_n = _normalize(flux)

    # kick/bass band drives the flash; broadband flux only adds texture
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    low = freqs <= 260.0
    low_flux = np.zeros(len(spec), dtype=np.float32)
    if len(spec) > 1 and np.any(low):
        low_flux[1:] = np.maximum(0.0, np.diff(spec[:, low], axis=0)).sum(axis=1)
    punch_flux = 0.75 * _normalize(low_flux) + 0.25 * flux_n
    punch_src = np.clip((punch_flux - 0.08) / 0.92, 0.0, 1.0) ** 1.15
    punch = _envelope(
        punch_src, fps,
        max(3.0, policy.attack_ms * 0.4),
        max(60.0, min(180.0, policy.decay_ms * 0.6)),
    )
    body = _envelope(rms_n, fps, max(6.0, policy.attack_ms), policy.decay_ms)

    bpm = estimate_bpm(flux_n, fps)
    onset = onset_strength(spec, sr, frame)
    beats = refine_beats(track_beats(onset, fps, bpm), onset, fps)
    beat_env = pulse_envelope(beats, len(spec), fps)

    if float(beat_env.max()) > 0.0:
        # beat-locked: flash on the beat grid, not on every hat / snare
        env = np.clip(0.80 * beat_env + 0.22 * body, 0.0, 1.0)
    else:
        # fallback: reactive punch (tempo too ambiguous for DP tracking)
        env = np.clip(0.18 * body + 0.95 * punch, 0.0, 1.0)

    level = env.astype(np.float32)  # raw 0..1; shaped by Policy at render time

    bands = _band_energy(spec, sr, frame)
    peak = bands.max(axis=1, keepdims=True) + 1e-6
    bands = np.sqrt(bands / peak).astype(np.float32)

    return Analysis(
        fps=sr / hop,
        level=level,
        bands=bands,
        bpm=bpm,
        scenes=classify_scenes(samples, sr),
        duration=len(samples) / sr,
        sr=sr,
        beats=beats,
    )


# ---------------------------------------------------------------- live

class LiveAnalyzer:
    """Real-time analyser for mic / loopback streams."""

    def __init__(self, sr: int, policy: Policy, frame: int = 1024):
        self.sr = sr
        self.policy = policy
        self.frame = frame
        self.buf = np.zeros(frame * 8, dtype=np.float32)
        self.window = np.hanning(frame).astype(np.float32)
        self.prev_mag: np.ndarray | None = None
        self.body = 0.0
        self.punch = 0.0
        self.rms_peak = 0.05
        self.flux_peak = 1.0
        self.low_peak = 1.0
        self._freqs = np.fft.rfftfreq(frame, 1.0 / sr)
        self._low = self._freqs <= 260.0
        self.last_t = 0.0
        self.last_level = 0.0
        self.last_punch = 0.0

    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size >= self.buf.size:
            self.buf[:] = block[-self.buf.size:]
        else:
            self.buf[:-block.size] = self.buf[block.size:]
            self.buf[-block.size:] = block

    def read(self, now: float) -> tuple[float, np.ndarray]:
        seg = self.buf[-self.frame:] * self.window
        mag = np.abs(np.fft.rfft(seg))
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))

        if self.prev_mag is None:
            flux = 0.0
            low_flux = 0.0
        else:
            diff = np.maximum(0.0, mag - self.prev_mag)
            flux = float(diff.sum())
            low_flux = float(diff[self._low].sum()) if np.any(self._low) else flux
        self.prev_mag = mag

        self.rms_peak = max(rms, self.rms_peak * 0.995, 1e-3)
        self.flux_peak = max(flux, self.flux_peak * 0.995, 1e-3)
        self.low_peak = max(low_flux, self.low_peak * 0.995, 1e-3)
        rms_n = min(rms / self.rms_peak, 1.0)
        flux_n = min(flux / self.flux_peak, 1.0)
        low_n = min(low_flux / self.low_peak, 1.0)
        punch_flux = 0.75 * low_n + 0.25 * flux_n

        dt = max(now - self.last_t, 1e-3)
        self.last_t = now

        body_ms = self.policy.attack_ms if rms_n > self.body else self.policy.decay_ms
        cb = float(np.exp(-dt / (max(body_ms, 2.0) / 1000.0)))
        self.body = cb * self.body + (1.0 - cb) * rms_n

        punch_src = max(0.0, min((punch_flux - 0.08) / 0.92, 1.0)) ** 1.15
        if punch_src > 0.45 and (now - self.last_punch) < 0.09:
            punch_src = 0.0
        if punch_src > self.punch:
            self.last_punch = now
        punch_ms = 4.0 if punch_src > self.punch else max(60.0, min(180.0, self.policy.decay_ms * 0.6))
        cp = float(np.exp(-dt / (punch_ms / 1000.0)))
        self.punch = cp * self.punch + (1.0 - cp) * punch_src

        level = float(np.clip(0.18 * self.body + 0.95 * self.punch, 0.0, 1.0))
        self.last_level = level

        bands = _band_energy(mag[None, :], self.sr, self.frame, 24)[0]
        peak = float(bands.max()) + 1e-6
        bands = np.sqrt(bands / peak).astype(np.float32)
        return level, bands


def shape_level(env: float, policy: Policy) -> float:
    """Legacy mapping: base / gain / gamma + a hard black point.

    Kept for A/B testing and as the classic-engine fallback.  With the
    guardrail off the black point is larger, so the screen goes nearly black
    between beats and snaps to full brightness on the kick -- which is exactly
    what made smooth music look like a few straight segments (see
    ``shape_level_smooth``).
    """
    value = policy.base + policy.gain * float(env)
    value = min(max(value, 0.0), 1.0) ** policy.gamma
    black = {"off": 0.16, "comfort": 0.06, "strict": 0.0}.get(policy.safety, 0.06)
    if black > 0.0:
        value = min(max((value - black) / (1.0 - black), 0.0), 1.0)
    return float(value)


# default S-curve strength per safety level; a policy may override with
# ``curve_contrast`` (>= 0)
_CURVE_CONTRAST = {"off": 0.45, "comfort": 0.22, "strict": 0.08}


def shape_level_smooth(env: float, policy: Policy, contrast: float | None = None) -> float:
    """C1-continuous brightness mapping: base/gain/gamma + a soft S-curve.

    Why not the old hard black point: subtracting a threshold clips every
    energy value below it to exactly 0, so a continuous energy contour becomes
    a flat line with a linear ramp on top -- visually "a few segments".  A
    smoothstep blend deepens blacks and brightens peaks while staying C1, so
    the same contour keeps its curvature.
    """
    value = policy.base + policy.gain * float(env)
    value = min(max(value, 0.0), 1.0) ** max(policy.gamma, 0.05)
    if contrast is None:
        c = getattr(policy, "curve_contrast", -1.0)
        contrast = _CURVE_CONTRAST.get(policy.safety, 0.22) if c is None or c < 0 else c
    contrast = min(max(float(contrast), 0.0), 1.0)
    if contrast > 0.0:
        smooth = value * value * (3.0 - 2.0 * value)
        value = value + (smooth - value) * contrast
    return float(min(max(value, 0.0), 1.0))
