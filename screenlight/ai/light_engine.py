"""Perceptual light engine (v3) — the "quality algorithm" plugin.

Two implementations, one algorithm family:

``PerceptualOfflineEngine``  pre-analyses a whole track: robust AGC energy
    contour + low-band beat grid + music-driven accents + beat confidence.
``PerceptualLiveEngine``     the same idea causally: rolling AGC, online tempo
    estimate with hysteresis, phase-locked beat accents, confidence gating.

Key design points (see ``docs/ARCHITECTURE_V3.md`` §3.3):

* **Continuous energy owns the brightness.**  RMS -> dB -> robust AGC -> one-pole
  attack/release.  Never a spectral flux gate, so smooth music stays smooth.
* **Beats are an additive accent, not the signal.**  A beat pulse is mixed on
  top with weight ``beat_weight`` and gated by ``beat_confidence``; when the
  material has no clear pulse the engine falls back to pure breathing instead of
  inventing a grid.
* **Low-band ODF fixes the offbeat lock.**  The grid is tracked on 40–260 Hz
  flux (kick/bass), so hi-hats and melody onsets cannot move the phase.
* **Accent amplitude follows the music**, not a hard-coded "every 4th beat".
* **Policy-independent features, policy-time shaping.**  Analysis results are
  cached; base/gain/gamma/attack/decay are applied per frame, so Agent commands
  ("温柔一点") take effect immediately.
* Everything is wrapped by ``core.engine.SafeEngine`` and falls back to the
  classic Ellis-DP / reactive path.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..beat import onset_strength, refine_beats, track_beats
from ..core import dsp
from ..core import emotion_dsp
from ..core.frame import N_BANDS, AnalysisFrame
from ..config import ROOT
from ..core.registry import register
from ..features import (LiveAnalyzer, analyze_offline, shape_level,
                        shape_level_smooth)

FRAME = 1024
HOP = 256
LOOP_FADE = 0.06          # seconds of loop-boundary crossfade
ACCENT_TAU = 0.12         # seconds, exponential accent decay (policy-independent)

# per safety level: how much of the picture is beat accent, how smooth the
# envelope is, how punchy the accent looks, how strong the S-curve is
SAFETY_TUNING = {
    "off":     dict(beat_weight=0.42, attack_floor=6.0,  release_floor=150.0,
                    accent_gain=1.00, contrast=0.45),
    "comfort": dict(beat_weight=0.30, attack_floor=15.0, release_floor=220.0,
                    accent_gain=0.85, contrast=0.22),
    "strict":  dict(beat_weight=0.15, attack_floor=40.0, release_floor=420.0,
                    accent_gain=0.55, contrast=0.08),
}


def tuning_for(policy) -> dict:
    # v2.1: the guard was replaced by user sliders.  The response profile is
    # fixed (the most responsive one) so sensitivity/base only change amplitude,
    # never the accent/timing; legacy --safety only maps to sensitivity.
    tune = dict(SAFETY_TUNING["off"])
    bw = getattr(policy, "beat_weight", -1.0)
    if bw is not None and float(bw) >= 0.0:
        tune["beat_weight"] = float(min(max(bw, 0.0), 1.0))
    cc = getattr(policy, "curve_contrast", -1.0)
    if cc is not None and float(cc) >= 0.0:
        tune["contrast"] = float(min(max(cc, 0.0), 1.0))
    return tune


def confidence_gate(conf: float, lo: float = 0.55, hi: float = 0.70) -> float:
    """0 below lo, 1 above hi, smoothstep in between.

    Calibrated on the shipped assets: drum material scores 0.70–0.97 (full
    accent), spoken/ambient material 0.25–0.50 (no accent at all).  The gap
    between 0.45 (smooth pad) and 0.70 (demo_beat) is what keeps the accent off
    on smooth music.
    """
    x = min(max((float(conf) - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------- helpers

def _onset_peaks(env: np.ndarray, fps: float, threshold: float = 0.60,
                 refractory: float = 0.10) -> np.ndarray:
    """Times of *prominent* ODF peaks (the loud hits: kick / snare / accent).

    A high relative threshold matters: with dense weak peaks (bass 16ths) almost
    any tempo "covers" them and the metrical level becomes ambiguous.  The loud
    hits are what a listener perceives as the beat.
    """
    if env.size < 4:
        return np.zeros(0, dtype=np.float64)
    thr = max(float(threshold * env.max()), float(np.percentile(env, 82)))
    out, last = [], -1e9
    for i in range(2, len(env) - 2):
        if env[i] >= thr and env[i] == env[i - 2:i + 3].max():
            t = i / fps
            if t - last > refractory:
                out.append(t)
                last = t
    return np.asarray(out, dtype=np.float64)


def _grid_mean(env: np.ndarray, fps: float, bpm: float, n_phases: int = 16) -> float:
    """Mean onset strength on the best-phase rigid grid (used only for fine
    BPM refinement *around* the metrical level chosen by coverage)."""
    if bpm <= 0:
        return 0.0
    period = 60.0 * fps / bpm
    if period < 2.0 or len(env) < 4.0 * period:
        return 0.0
    best = 0.0
    span = int((len(env) - 1) / period) + 1
    for ph in np.linspace(0.0, period, n_phases, endpoint=False):
        pos = ph + period * np.arange(span)
        pos = pos[pos < len(env) - 1]
        if len(pos) < 4:
            continue
        i = pos.astype(int)
        fr = pos - i
        v = env[i] * (1.0 - fr) + env[i + 1] * fr
        best = max(best, float(v.mean()))
    return best


def _grid_coverage(peaks: np.ndarray, bpm: float, tol: float = 0.055) -> float:
    """Fraction of onset peaks that a rigid grid at ``bpm`` can hit (±tol).

    Coverage is unbiased across tempos (unlike the mean ODF on the grid, which
    favours slower grids) so it separates "every hit is on the beat" from
    "every other hit".  Vectorised: it is called ~13k times per long track.
    """
    if bpm <= 0 or peaks.size < 6:
        return 0.0
    period = 60.0 / bpm
    bins = 36
    idx = ((peaks % period) / period * bins).astype(np.int64) % bins
    hist = np.bincount(idx, minlength=bins).astype(np.float64)
    half = max(1, int(round(tol / period * bins)))
    padded = np.concatenate((hist[-half:], hist, hist[:half]))
    k = np.ones(2 * half + 1)
    best = float(np.convolve(padded, k, mode="valid").max())
    return min(best / float(peaks.size), 1.0)


def estimate_tempo(odf: np.ndarray, fps: float,
                   bpm_lo: float = 60.0, bpm_hi: float = 185.0) -> tuple[float, float]:
    """Tempo by autocorr candidates + grid coverage + log prior.

    Autocorrelation proposes candidates (and their octave relatives); the final
    choice maximises how many **prominent onsets** a rigid grid at that tempo
    actually covers.  That removes both classic failure modes:

    * half/double tempo (a slower grid "covers" every other hit: coverage 50%);
    * 3:4 ambiguity (90 vs 120 BPM on triplet-heavy material): the grid that
      covers the real hits wins.
    """
    odf = np.asarray(odf, dtype=np.float64)
    if odf.size < 16 or float(odf.std()) < 1e-9:
        return 0.0, 0.0
    env = dsp.smooth_gauss(odf, fps, 18.0).astype(np.float64)
    env = env / (float(env.max()) + 1e-9)
    peaks = _onset_peaks(env, fps)
    if peaks.size < 6:
        return 0.0, 0.0

    # full scan is cheap (a histogram of peaks per candidate) and much more
    # robust than chasing autocorrelation peaks and their octaves
    scan = np.arange(bpm_lo, bpm_hi + 1e-9, 1.0)
    cov = np.array([_grid_coverage(peaks, float(b)) for b in scan])
    best_cov = float(cov.max())
    if best_cov < 0.50:                 # not enough evidence of a beat
        return 0.0, best_cov

    # metrical level: the SLOWEST tempo that still covers (almost) all hits.
    # 90 BPM covers the demo beats at 93%; a 60 BPM grid only covers every
    # other hit (32%), and 180 BPM is a subdivision (also 93% but 2x too fast).
    threshold = 0.85 * best_cov
    ok = np.where(cov >= threshold)[0]
    best_bpm = float(scan[int(ok[0])]) if ok.size else float(scan[int(np.argmax(cov))])

    # local refinement (+-3 BPM, 0.1 steps) maximising mean ODF on the grid:
    # coverage alone saturates near the true tempo (118.5 vs 120.2 look equal),
    # but the grid *energy* peaks sharply at the true period because any
    # mismatch accumulates phase drift across the window.
    fine = np.arange(max(bpm_lo, best_bpm - 1.5), min(bpm_hi, best_bpm + 1.5) + 1e-9, 0.25)
    if fine.size:
        vals = np.array([_grid_mean(env, fps, float(b)) for b in fine])
        best_bpm = float(fine[int(np.argmax(vals))]) if vals.max() > 0 else best_bpm

    strength = float(np.clip(_grid_coverage(peaks, best_bpm), 0.0, 1.0))
    return float(best_bpm), strength


def accent_envelope(beats: np.ndarray, amps: np.ndarray, n_frames: int, fps: float,
                    tau: float = ACCENT_TAU, attack: float = 0.020) -> np.ndarray:
    """One exponential accent per beat, amplitude from the music (no 4/4 rule).

    ``attack`` (20 ms) keeps the pulse snappy while avoiding a single-frame
    luminance jump, which at 60 fps reads as a strobe rather than a punch.
    """
    env = np.zeros(n_frames, dtype=np.float32)
    if beats.size == 0:
        return env
    length = max(1, int(tau * 4.0 * fps))
    t = np.arange(length, dtype=np.float32) / fps
    decay = np.exp(-t / max(tau, 1e-3))
    a = max(1, int(round(attack * fps)))
    ramp = np.linspace(0.0, 1.0, a, dtype=np.float32)
    for k, b in enumerate(beats):
        start = int(b)
        if start < 0 or start >= n_frames:
            continue
        amp = float(amps[k]) if k < len(amps) else 0.8
        end = min(n_frames, start + length)
        seg = amp * decay[: end - start]
        head = min(a, len(seg))
        seg[:head] *= ramp[:head]
        env[start:end] = np.maximum(env[start:end], seg)
    return env


def _tempo_strength_table(broad_odf: np.ndarray, fps: float):
    """One FFT autocorrelation of the wideband ODF (~43 Hz), reused by every
    tempo segment.  The old per-segment ``np.correlate`` was O(n^2) and took
    ~2.6 s on a 5-minute track."""
    if len(broad_odf) < 64:
        return np.zeros(1), 43.0
    ds = max(1, int(round(fps / 43.0)))
    x = dsp.smooth_gauss(broad_odf[::ds], fps / ds, 25.0).astype(np.float64)
    x = x - x.mean()
    n = x.size
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    ac = ac / (float(ac[0]) + 1e-9)
    return ac.astype(np.float32), fps / ds


def _tempo_strength_at(ac: np.ndarray, fps_ds: float, bpm: float) -> float:
    if bpm <= 0 or ac.size < 4:
        return 0.0
    lag = 60.0 * fps_ds / bpm
    i = int(round(lag))
    if i < 1 or i >= ac.size:
        return 0.0
    return float(np.clip(np.max(ac[max(0, i - 2):i + 3]), 0.0, 1.0))


def beat_confidence(low_odf: np.ndarray, beats: np.ndarray, fps: float,
                    tempo_strength: float) -> float:
    """How much should we trust the beat grid as *the* rhythm of this music?

    Interval regularity alone is useless here (the DP always returns a regular
    grid, and a smooth pad produces regular swells).  The decisive signal is
    **low-band onset lift**: are the beats actually on strong kick/bass
    transients?  Measured on the shipped assets:

        demo_120bpm  lift 4.57 -> yes         demo_smooth  lift 1.19 -> no
        demo_beat    lift 1.71 -> yes         (speech-like material ~1.1)

    Gate thresholds (0.40 / 0.62) are calibrated so the three drum assets clear
    it and the ambient asset stays below it.
    """
    if beats.size < 8:
        return 0.0
    intervals = np.diff(beats) / fps
    med = float(np.median(intervals))
    if med <= 0.05:
        return 0.0
    reg = 1.0 - min(float(np.std(intervals)) / med, 1.0)
    idx = np.clip(np.round(beats).astype(int), 0, len(low_odf) - 1)
    lift = float(low_odf[idx].mean() / (low_odf.mean() + 1e-9))
    lift_term = float(np.clip((lift - 1.12) / 1.5, 0.0, 1.0))
    conf = 0.60 * lift_term + 0.15 * reg + 0.25 * float(np.clip(tempo_strength, 0.0, 1.0))
    if beats.size < 12:
        conf *= 0.75
    return float(np.clip(conf, 0.0, 1.0))


def _bands_from_spec(spec: np.ndarray, sr: int, frame: int,
                     n_bands: int = N_BANDS) -> np.ndarray:
    b = dsp.band_matrix(spec, sr, frame, n_bands)
    edges = dsp.band_edges(sr, frame, n_bands)
    centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    b *= (centers / 1000.0) ** 0.55          # spectral tilt: stop the ramp look
    peak = b.max(axis=1, keepdims=True) + 1e-6
    return np.sqrt(b / peak).astype(np.float32)


def _scenes_from_spec(spec: np.ndarray, sr: int, frame: int, fps: float) -> list[tuple[float, str]]:
    """Cheap music/speech/applause heuristic on the already computed STFT.

    Percussion-heavy music also has a fairly flat spectrum, so "flatness > 0.22"
    (the old rule) labelled it as applause.  Applause is much flatter still
    (>0.55 median over 1.5 s) and carries little bass; speech gets a stricter
    flatness/centroid + dynamics test.  Everything else is music, which is the
    safe default for a music visualiser.
    """
    step = max(1, int(round(fps * 0.5)))
    rows = spec[::step]
    if rows.size == 0:
        return [(0.0, "music")]
    flat = np.exp(np.mean(np.log(rows + 1e-9), axis=1)) / (np.mean(rows, axis=1) + 1e-9)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr).astype(np.float32)
    centroid = (rows * freqs[None, :]).sum(axis=1) / (rows.sum(axis=1) + 1e-9)
    low = freqs < 250.0
    low_ratio = rows[:, low].sum(axis=1) / (rows.sum(axis=1) + 1e-9)
    rms = np.sqrt((rows * rows).mean(axis=1) + 1e-12)
    # 3-block median smoothing so a single noisy window does not flip the label
    def med(x: np.ndarray, k: int = 3) -> np.ndarray:
        pad = np.pad(x, (k // 2, k // 2), mode="edge")
        return np.array([np.median(pad[i:i + k]) for i in range(len(x))])

    flat_s = med(flat)
    cent_s = med(centroid)
    low_s = med(low_ratio)
    rms_s = med(rms)
    out: list[tuple[float, str]] = []
    for i in range(len(rows)):
        dyn = (np.percentile(rms_s, 90) - np.percentile(rms_s, 10)) / (
            np.percentile(rms_s, 90) + 1e-9)
        if flat_s[i] > 0.55 and low_s[i] < 0.30:
            label = "applause"
        elif 0.02 < flat_s[i] < 0.12 and cent_s[i] < 1200.0 and dyn > 0.55:
            label = "speech"
        else:
            label = "music"
        out.append((float(i * step / fps), label))
    return out


def _loop_blend(curve: np.ndarray, fps: float, fade: float = LOOP_FADE) -> np.ndarray:
    """Crossfade the tail into the head so the winsound loop has no visual jump."""
    n = curve.shape[0]
    k = int(round(fade * fps))
    if k < 3 or n < 2 * k + 2:
        return curve
    w = np.linspace(0.0, 1.0, k, dtype=np.float32)
    head = curve[:k].copy()
    if curve.ndim == 1:
        curve[-k:] = curve[-k:] * (1.0 - w) + head * w
    else:
        curve[-k:] = curve[-k:] * (1.0 - w)[:, None] + head * w[:, None]
    return curve


def local_tempo(low_s: np.ndarray, fps: float, win: float = 8.0,
                hop: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding tempo estimate -> (centre_times, bpms, strengths).

    This is what makes tempo *changes* work: ``demo_beat.wav`` is 120 BPM for
    20 s and 90 BPM afterwards, so a single global BPM puts the second half of
    the track permanently off the beat (the original "does not hit the drums"
    complaint).
    """
    n = len(low_s)
    w = max(16, int(round(win * fps)))
    h = max(1, int(round(hop * fps)))
    if n <= w:
        bpm, strength = estimate_tempo(low_s, fps)
        return np.array([n / (2.0 * fps)]), np.array([bpm]), np.array([strength])
    centers, bpms, strengths = [], [], []
    for a in range(0, n - w + 1, h):
        bpm, strength = estimate_tempo(low_s[a:a + w], fps)
        centers.append((a + w * 0.5) / fps)
        bpms.append(bpm)
        strengths.append(strength)
    centers = np.asarray(centers, dtype=np.float64)
    bpms = np.asarray(bpms, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if bpms.size >= 5:                             # median filter: kill outliers
        bpms = np.asarray([np.median(bpms[max(0, i - 2):i + 3])
                           for i in range(bpms.size)])
    ok = bpms > 0
    if ok.any():
        idx = np.arange(bpms.size)
        bpms = np.interp(idx, idx[ok], bpms[ok])
    return centers, bpms, strengths


@dataclass
class TempoSegment:
    t0: float
    t1: float
    bpm: float
    confidence: float = 0.0
    strength: float = 0.0


def build_segments(centers: np.ndarray, bpms: np.ndarray, strengths: np.ndarray,
                   duration: float, min_len: float = 4.0,
                   change: float = 0.08) -> list[TempoSegment]:
    """Split the track where the local tempo changes by more than ``change``."""
    if bpms.size == 0:
        return [TempoSegment(0.0, duration, 0.0)]
    cuts = [0.0]
    for i in range(1, bpms.size):
        a, b = bpms[i - 1], bpms[i]
        if a <= 0 or b <= 0:
            continue
        if abs(b - a) / a > change:
            cut = float((centers[i - 1] + centers[i]) * 0.5)
            if cut - cuts[-1] >= min_len and duration - cut >= min_len:
                cuts.append(cut)
    cuts.append(duration)

    def bpm_in(a: float, b: float) -> tuple[float, float]:
        m = (centers >= a) & (centers < b)
        if m.any():
            return float(np.median(bpms[m])), float(np.median(strengths[m]))
        i = int(np.argmin(np.abs(centers - (a + b) * 0.5)))
        return float(bpms[i]), float(strengths[i])

    segs: list[TempoSegment] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        bpm, strength = bpm_in(a, b)
        segs.append(TempoSegment(float(a), float(b), bpm, 0.0, strength))

    # merge segments shorter than min_len into their neighbour
    merged: list[TempoSegment] = []
    for s in segs:
        if merged and (s.t1 - s.t0) < min_len:
            merged[-1].t1 = s.t1
            continue
        if merged and (merged[-1].t1 - merged[-1].t0) < min_len:
            merged[-1].t1 = s.t1
            continue
        merged.append(s)
    for s in merged:
        s.bpm, s.strength = bpm_in(s.t0, s.t1)
    return merged


@dataclass
class TrackCurve:
    times: np.ndarray
    energy: np.ndarray
    accent: np.ndarray
    bands: np.ndarray
    duration: float
    scenes: list[tuple[float, str]]
    beats: np.ndarray
    seg_t0: np.ndarray
    seg_t1: np.ndarray
    seg_bpm: np.ndarray
    seg_conf: np.ndarray
    valence: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    arousal: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    @property
    def bpm(self) -> float:
        valid = self.seg_bpm[self.seg_bpm > 0]
        return float(np.median(valid)) if valid.size else 0.0

    @property
    def confidence(self) -> float:
        w = np.maximum(self.seg_t1 - self.seg_t0, 1e-6)
        return float(np.clip(np.average(self.seg_conf, weights=w), 0.0, 1.0))

    def _seg_index(self, t: float) -> int:
        i = int(np.searchsorted(self.seg_t0, t, side="right")) - 1
        return max(0, min(i, len(self.seg_bpm) - 1))

    def bpm_at(self, t: float) -> float:
        return float(self.seg_bpm[self._seg_index(t)])

    def conf_at(self, t: float) -> float:
        return float(self.seg_conf[self._seg_index(t)])

    def valence_at(self, t: float) -> float:
        if self.valence is None or len(self.valence) == 0:
            return 0.5
        return float(np.interp(t, self.times, self.valence))

    def arousal_at(self, t: float) -> float:
        if self.arousal is None or len(self.arousal) == 0:
            return 0.5
        return float(np.interp(t, self.times, self.arousal))

    def emotion_at(self, t: float) -> str:
        return emotion_dsp.label_cn(self.valence_at(t), self.arousal_at(t))

    def at(self, t: float):
        t = float(t) % max(self.duration, 1e-6)
        e = float(np.interp(t, self.times, self.energy))
        a = float(np.interp(t, self.times, self.accent))
        b = np.empty(self.bands.shape[1], dtype=np.float32)
        for i in range(b.shape[0]):
            b[i] = float(np.interp(t, self.times, self.bands[:, i]))
        return e, a, b

    def beat_phase_at(self, t: float) -> float:
        bpm = self.bpm_at(t)
        period = 60.0 / bpm if bpm > 0 else 0.5
        if self.beats.size < 2:
            return float((t % period) / period)
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        anchor = float(self.beats[i]) if i >= 0 else float(self.beats[0]) - period
        return float(min(max((t - anchor) / period, 0.0), 1.0))

    def scene_at(self, t: float) -> str:
        label = "music"
        for start, name in self.scenes:
            if start <= t:
                label = name
            else:
                break
        return label


# ---------------------------------------------------------------- offline engine

ANALYSIS_DIR = ROOT / "work" / "analysis"
CURVE_VERSION = "v7"


def _curve_key(samples: np.ndarray, sr: int) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(int(sr).to_bytes(4, "little"))
    h.update(len(samples).to_bytes(8, "little"))
    h.update(np.ascontiguousarray(samples[:262144]).tobytes())
    h.update(np.ascontiguousarray(samples[-262144:]).tobytes())
    h.update(CURVE_VERSION.encode("ascii"))
    return h.hexdigest()


def _load_curve(key: str) -> TrackCurve | None:
    path = ANALYSIS_DIR / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"][0]))
            return TrackCurve(
                times=z["times"], energy=z["energy"], accent=z["accent"],
                bands=z["bands"], duration=float(meta["duration"]),
                scenes=[(float(a), str(b)) for a, b in meta["scenes"]],
                beats=z["beats"], seg_t0=z["seg_t0"], seg_t1=z["seg_t1"],
                seg_bpm=z["seg_bpm"], seg_conf=z["seg_conf"],
                valence=(z["valence"] if "valence" in z.files else None),
                arousal=(z["arousal"] if "arousal" in z.files else None),
            )
    except Exception:  # noqa: BLE001 - a corrupt cache must never break playback
        return None


def _save_curve(key: str, curve: TrackCurve) -> None:
    try:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        meta = np.array([json.dumps({
            "duration": curve.duration,
            "scenes": [[a, b] for a, b in curve.scenes],
        }, ensure_ascii=False)])
        np.savez_compressed(
            ANALYSIS_DIR / f"{key}.npz",
            times=curve.times, energy=curve.energy, accent=curve.accent,
            bands=curve.bands, beats=curve.beats, seg_t0=curve.seg_t0,
            seg_t1=curve.seg_t1, seg_bpm=curve.seg_bpm, seg_conf=curve.seg_conf,
            valence=(curve.valence if curve.valence is not None
                     else np.zeros(0, dtype=np.float32)),
            arousal=(curve.arousal if curve.arousal is not None
                     else np.zeros(0, dtype=np.float32)),
            meta=meta,
        )
    except Exception:  # noqa: BLE001
        pass


def analyse_track(samples: np.ndarray, sr: int) -> TrackCurve:
    """Policy-independent track analysis (the expensive part, done once).

    44.1/48 kHz sources are decimated by 2 for analysis only (playback keeps the
    original file): the beat grid lives in 40-260 Hz and the canvas in <11 kHz,
    so this halves the analysis cost with no audible/visual change.
    """
    if sr > 32000:
        # pair-average = crude but effective half-band anti-alias; plain
        # subsampling folded hi-hats/bells back into the mid band and made
        # smooth material look like it had a beat
        n = len(samples) // 2 * 2
        samples = samples[:n].reshape(-1, 2).mean(axis=1).astype(np.float32)
        samples = np.ascontiguousarray(samples)
        sr = int(sr) // 2
    key = _curve_key(samples, sr)
    cached = _load_curve(key)
    if cached is not None:
        return cached
    frames = dsp.frames_view(samples, FRAME, HOP)
    spec = dsp.magnitude(frames, FRAME)
    fps = sr / float(HOP)
    rms = dsp.rms_of(frames)
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))

    # ---- continuous energy: robust AGC + light smoothing (fast + slow mix)
    # ^1.22 gives the mid-range some headroom: without it a steady pad sits at
    # ~0.9 and the screen is washed out; drums still reach ~0.9 on a hit.
    db_n = dsp.normalize(db, 5.0, 95.0)
    e_fast = dsp.smooth_gauss(db_n, fps, 60.0)
    e_slow = dsp.smooth_gauss(db_n, fps, 700.0)
    energy = np.clip((0.68 * e_fast + 0.32 * e_slow) ** 1.22, 0.0, 1.0).astype(np.float32)

    # ---- time-varying beat grid: local tempo -> segments -> DP per segment.
    # A single global BPM fails on tracks with a half-time section (the demo
    # excerpt is 120 BPM for 20 s then 90 BPM), which is exactly why the old
    # version drifted off the drums after the first section.
    low = dsp.band_flux(spec, sr, FRAME, 40.0, 160.0)
    low_s = dsp.smooth_gauss(low, fps, 18.0)
    wide = onset_strength(spec, sr, FRAME)
    wide_s = dsp.smooth_gauss(wide, fps, 18.0)
    ac_strength, ac_fps = _tempo_strength_table(wide_s, fps)
    centers, bpms, strengths = local_tempo(wide_s, fps)
    duration = max(len(samples) / float(sr), 1e-3)
    segs = build_segments(centers, bpms, strengths, duration)

    beats_list: list[np.ndarray] = []
    for seg in segs:
        seg.confidence = 0.0
        if seg.bpm <= 0.0:
            continue
        a0 = int(max(0, (seg.t0 - 1.0) * fps))
        b0 = int(min(len(low_s), (seg.t1 + 1.0) * fps))
        sl = low_s[a0:b0]
        if sl.size < 16:
            continue
        try:
            bl = refine_beats(track_beats(sl, fps, seg.bpm), sl, fps, radius=0.04)
        except Exception:  # noqa: BLE001 - a bad grid must not kill the engine
            continue
        bl = bl + a0
        lo, hi = int(seg.t0 * fps), int(seg.t1 * fps)
        seg_beats = bl[(bl >= lo) & (bl < hi)].astype(np.float64)
        strength = _tempo_strength_at(ac_strength, ac_fps, seg.bpm)
        seg.strength = strength
        seg.confidence = beat_confidence(low_s, seg_beats, fps, strength)
        if seg_beats.size:
            beats_list.append(seg_beats)
    beats_f = np.concatenate(beats_list) if beats_list else np.zeros(0, dtype=np.float64)
    beats_s = beats_f / fps

    # ---- accent amplitude from the actual low-band energy at each beat
    if beats_f.size:
        idx = np.clip(np.round(beats_f).astype(int), 0, len(low_s) - 1)
        ref = float(np.percentile(low_s, 92)) + 1e-9
        amps = 0.55 + 0.45 * np.clip(low_s[idx] / ref, 0.0, 1.0)
    else:
        amps = np.zeros(0, dtype=np.float32)
    accent = accent_envelope(beats_f, amps, len(energy), fps)
    accent = _loop_blend(accent, fps)

    # ---- bands + scene, then drop the big spectrogram
    bands = _bands_from_spec(spec, sr, FRAME, N_BANDS)
    scenes = _scenes_from_spec(spec, sr, FRAME, fps)
    times = (np.arange(len(energy), dtype=np.float32) / fps).astype(np.float32)
    # per-frame tempo for the emotion model (same BPM can still feel different)
    bpm_frames = np.zeros(len(energy), dtype=np.float32)
    for seg in segs:
        a0 = int(max(0, seg.t0 * fps))
        b0 = int(min(len(bpm_frames), seg.t1 * fps))
        bpm_frames[a0:b0] = seg.bpm
    if np.any(bpm_frames > 0):
        bpm_frames[bpm_frames <= 0] = float(np.median(bpm_frames[bpm_frames > 0]))
    else:
        bpm_frames[:] = 100.0
    emo = emotion_dsp.emotion_from_spec(spec, rms, fps, sr, FRAME, bpm_frames)
    valence = _loop_blend(emo["valence"], fps)
    arousal = _loop_blend(emo["arousal"], fps)
    energy = _loop_blend(energy, fps)
    bands = _loop_blend(bands, fps)
    curve = TrackCurve(
        times=times, energy=energy, accent=accent, bands=bands,
        duration=duration, scenes=scenes, beats=beats_s,
        seg_t0=np.array([s.t0 for s in segs], dtype=np.float64),
        seg_t1=np.array([s.t1 for s in segs], dtype=np.float64),
        seg_bpm=np.array([s.bpm for s in segs], dtype=np.float64),
        seg_conf=np.array([s.confidence for s in segs], dtype=np.float64),
        valence=valence, arousal=arousal,
    )
    _save_curve(key, curve)
    return curve


class PerceptualOfflineEngine:
    """Whole-track light engine (in-app playback)."""

    name = "perceptual"
    healthy = True

    def __init__(self, samples: np.ndarray, sr: int, policy) -> None:
        self.sr = int(sr)
        self.curve = analyse_track(samples, sr)
        self.duration = self.curve.duration
        self.bpm = self.curve.bpm
        self.scenes = self.curve.scenes
        self.beat_confidence = self.curve.confidence
        # per-frame state (policy is applied at frame time, not baked in)
        self._energy = 0.0
        self._last_t: float | None = None
        self._last_frame = AnalysisFrame(engine=self.name)

    # ------------------------------------------------------------ frame
    def frame_at(self, t: float, policy, now: float | None = None) -> AnalysisFrame:
        tune = tuning_for(policy)
        tt = float(t) % max(self.duration, 1e-6)
        if self._last_t is None:
            dt = 1.0 / 60.0
        else:
            dt = tt - self._last_t
            if dt < 0.0 or dt > 0.25:      # seek / loop / first frame
                dt = 1.0 / 60.0
        self._last_t = tt

        e_raw, a_raw, bands = self.curve.at(tt)
        attack = max(float(policy.attack_ms), tune["attack_floor"])
        release = max(float(policy.decay_ms), tune["release_floor"])
        self._energy = dsp.ar_step(self._energy, e_raw, dt, attack, release)

        conf = self.curve.conf_at(tt)          # per tempo segment: a track can
        gate = confidence_gate(conf)           # go from drum-driven to ambient
        accent_vis = float(np.clip(a_raw * gate * tune["accent_gain"], 0.0, 1.0))
        w = tune["beat_weight"] * gate
        raw = (1.0 - w) * self._energy + w * accent_vis
        level = shape_level_smooth(raw, policy, contrast=tune["contrast"])

        frame = AnalysisFrame(
            t=tt,
            level=level,
            bands=bands,
            bpm=self.curve.bpm_at(tt),
            beat_phase=self.curve.beat_phase_at(tt),
            energy=float(self._energy),
            accent=accent_vis,
            beat_confidence=conf,
            scene=self.curve.scene_at(tt),
            scene_confidence=0.5,
            wake=False,
            engine=self.name,
            valence=self.curve.valence_at(tt),
            arousal=self.curve.arousal_at(tt),
            emotion=self.curve.emotion_at(tt),
        )
        self._last_frame = frame
        return frame.clamped()


# ---------------------------------------------------------------- live engine

class PerceptualLiveEngine:
    """Causal version for mic / system-loopback audio.

    Rolling AGC energy + online tempo estimate + phase-locked accents.  When the
    tempo estimate is weak (smooth music, speech) the accent is gated to zero and
    the screen just breathes with the energy contour.
    """

    name = "perceptual-live"
    healthy = True

    def __init__(self, sr: int, policy) -> None:
        self.sr = int(sr)
        self.frame_size = FRAME
        self.window = np.hanning(FRAME).astype(np.float32)
        self.buf = np.zeros(FRAME * 8, dtype=np.float32)
        self.prev_mag: np.ndarray | None = None
        self._freqs = np.fft.rfftfreq(FRAME, 1.0 / self.sr)
        self._low = (self._freqs >= 40.0) & (self._freqs < 160.0)

        self.db_hist: deque[float] = deque(maxlen=360)     # ~6 s @60fps
        self.onset_hist: deque[float] = deque(maxlen=600)  # ~10 s @60fps
        self.tempo_hist: deque[float] = deque(maxlen=5)

        self.energy = 0.0
        self.accent = 0.0
        self.accent_target = 0.0
        self.bpm = 0.0
        self.confidence = 0.0
        self.phase = 0.0
        self.last_beat = -1e9
        self.last_onset = -1e9
        self.last_t: float | None = None
        self._last_tempo_check = -1e9
        self._onset_peak = 1e-3
        self._frames_seen = 0
        self._valence = 0.5
        self._arousal = 0.5
        self.spec_hist: deque[np.ndarray] = deque(maxlen=24)
        self.rms_hist: deque[float] = deque(maxlen=24)
        self._last_frame = AnalysisFrame(engine=self.name)

    # ------------------------------------------------------------ input
    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        if block.size >= self.buf.size:
            self.buf[:] = block[-self.buf.size:]
        else:
            self.buf[:-block.size] = self.buf[block.size:]
            self.buf[-block.size:] = block

    # ------------------------------------------------------------ helpers
    def _normalized_rms(self, rms: float) -> float:
        db = 20.0 * math.log10(max(rms, 1e-6))
        self.db_hist.append(db)
        if len(self.db_hist) >= 30:
            arr = np.fromiter(self.db_hist, dtype=np.float32)
            lo, hi = np.percentile(arr[-240:], [15.0, 92.0])
        else:
            lo, hi = -46.0, -8.0
        if hi - lo < 6.0:
            hi = lo + 6.0
        return float(min(max((db - lo) / (hi - lo), 0.0), 1.0))

    def _update_tempo(self, now: float) -> None:
        """Rolling tempo estimate with the same coverage test as the offline
        engine.  A plain autocorrelation height is too permissive: a smooth pad
        with periodic swells produces a tall autocorr peak and would turn beat
        accents on for music that has no beat at all (live smooth-music bug)."""
        if now - self._last_tempo_check < 0.5 or len(self.onset_hist) < 240:
            return
        self._last_tempo_check = now
        x = np.fromiter(self.onset_hist, dtype=np.float32)
        bpm, strength = estimate_tempo(x, 60.0)
        if bpm <= 0.0 or strength < 0.62:      # not enough evidence of a pulse
            self.confidence *= 0.55
            return
        if self.tempo_hist:
            ref = float(np.median(self.tempo_hist))
            if bpm / ref > 1.55:              # octave jump protection
                bpm /= 2.0
            elif bpm / ref < 0.65:
                bpm *= 2.0
        self.tempo_hist.append(bpm)
        stability = 1.0
        if len(self.tempo_hist) >= 3:
            arr = np.fromiter(self.tempo_hist, dtype=np.float32)
            stability = 1.0 - min(float(np.std(arr)) / max(float(np.median(arr)), 1e-6), 1.0)
        self.bpm = float(np.median(self.tempo_hist))
        self.confidence = float(np.clip((strength - 0.60) / 0.30, 0.0, 1.0)) * stability

    # ------------------------------------------------------------ frame
    def frame(self, now: float, policy) -> AnalysisFrame:
        dt = 1.0 / 60.0 if self.last_t is None else max(1e-3, min(0.25, now - self.last_t))
        self.last_t = now
        self._frames_seen += 1

        seg = self.buf[-FRAME:] * self.window
        mag = np.abs(np.fft.rfft(seg)).astype(np.float32)
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))

        low_flux = 0.0
        if self.prev_mag is not None:
            diff = np.maximum(0.0, mag - self.prev_mag)
            low_flux = float(diff[self._low].sum()) if np.any(self._low) else float(diff.sum())
        self.prev_mag = mag
        self._onset_peak = max(low_flux, self._onset_peak * 0.996, 1e-3)
        onset_norm = float(min(max(low_flux / self._onset_peak, 0.0), 1.0))
        self.onset_hist.append(onset_norm)

        tune = tuning_for(policy)
        e_target = self._normalized_rms(rms)
        # >=30 ms: at 60 fps a 6 ms attack is effectively a one-frame
        # jump, which reads as flicker in the live (mic/loopback) path
        attack = max(float(policy.attack_ms), tune["attack_floor"], 30.0)
        release = max(float(policy.decay_ms), tune["release_floor"])
        self.energy = dsp.ar_step(self.energy, e_target, dt, attack, release)

        self._update_tempo(now)

        # ---- phase lock: fire an accent on a strong low-band onset that lands
        # near the grid phase; otherwise advance the grid.
        period = 60.0 / self.bpm if self.bpm > 0 else 0.0
        if period > 0:
            self.phase = (self.phase + dt / period) % 1.0
        strong = onset_norm > 0.42
        if strong:
            self.last_onset = now
        gate = confidence_gate(self.confidence)
        fired = False
        if strong and gate > 0.05 and (now - self.last_beat) > 0.16 and period > 0:
            near = self.phase < 0.22 or self.phase > 0.78
            if near:
                amp = 0.55 + 0.45 * onset_norm
                self.accent_target = max(self.accent_target, amp * gate)
                self.phase = 0.0
                self.last_beat = now
                fired = True
        if not fired and period > 0 and self.phase >= 0.999:
            self.phase = 0.0
            if gate > 0.55 and (now - self.last_beat) > 0.45 * period:
                self.accent_target = max(self.accent_target, 0.5 * gate)
                self.last_beat = now

        # 20 ms attack / smooth decay: a single-frame jump reads as a strobe,
        # not a punch (the live path has no precomputed accent envelope)
        self.accent_target *= float(np.exp(-dt / ACCENT_TAU))
        self.accent = dsp.ar_step(self.accent, self.accent_target, dt, 22.0, 60.0)
        accent_vis = float(np.clip(self.accent * tune["accent_gain"], 0.0, 1.0))
        w = tune["beat_weight"] * gate
        raw = (1.0 - w) * self.energy + w * accent_vis
        level = shape_level_smooth(raw, policy, contrast=tune["contrast"])

        bands = _bands_from_spec(mag[None, :], self.sr, FRAME, N_BANDS)[0]
        # emotion: roll a ~2 s spectral history, refresh twice a second
        if self._frames_seen % 6 == 0:
            self.spec_hist.append(mag.astype(np.float32))
            self.rms_hist.append(rms)
            if len(self.spec_hist) >= 4:
                hist = np.stack(self.spec_hist)
                rms_arr = np.fromiter(self.rms_hist, dtype=np.float32,
                                      count=len(self.rms_hist))
                rms_arr = np.resize(rms_arr, hist.shape[0])
                try:
                    emo = emotion_dsp.emotion_from_spec(
                        hist, rms_arr, 10.0, self.sr, FRAME, self.bpm or 100.0)
                    self._valence = float(np.mean(emo["valence"][-4:]))
                    self._arousal = float(np.mean(emo["arousal"][-4:]))
                except Exception:  # noqa: BLE001 - emotion must never kill audio
                    pass
        frame = AnalysisFrame(
            t=max(now, 0.0),
            level=level,
            bands=bands,
            bpm=self.bpm,
            beat_phase=float(self.phase),
            energy=float(self.energy),
            accent=accent_vis,
            beat_confidence=float(self.confidence),
            scene="live",
            scene_confidence=0.5,
            wake=False,
            engine=self.name,
            valence=self._valence,
            arousal=self._arousal,
            emotion=emotion_dsp.label_cn(self._valence, self._arousal),
        )
        self._last_frame = frame
        return frame.clamped()


# ---------------------------------------------------------------- classic adapters

class ClassicOfflineEngine:
    """Adapter over the v1 pipeline (Ellis DP + pulse envelope + shape_level)."""

    name = "classic"
    healthy = True

    def __init__(self, samples: np.ndarray, sr: int, policy) -> None:
        self.analysis = analyze_offline(samples, sr, policy)
        self.duration = max(self.analysis.duration, 1e-3)
        self.bpm = float(self.analysis.bpm)
        self.scenes = self.analysis.scenes
        self.confidence = 1.0 if self.analysis.beats.size > 4 else 0.0
        self.beats_s = self.analysis.beats.astype(np.float64) / self.analysis.fps
        self._prev_raw = 0.0

    def frame_at(self, t: float, policy, now: float | None = None) -> AnalysisFrame:
        tt = float(t) % self.duration
        raw = float(self.analysis.level_at(tt))
        level = shape_level(raw, policy)
        accent = float(np.clip((raw - self._prev_raw) * 4.0, 0.0, 1.0))
        self._prev_raw = raw
        phase = 0.0
        if self.beats_s.size and self.bpm > 0:
            period = 60.0 / self.bpm
            phase = float((tt % period) / period)
        return AnalysisFrame(
            t=tt, level=level, bands=self.analysis.bands_at(tt), bpm=self.bpm,
            beat_phase=phase, energy=raw, accent=accent,
            beat_confidence=self.confidence, scene=self.analysis.scene_at(tt),
            scene_confidence=0.4, engine=self.name,
        ).clamped()

    def scene_at(self, t: float) -> str:
        return self.analysis.scene_at(t)


class ClassicLiveEngine:
    """Adapter over ``features.LiveAnalyzer`` (reactive punch, no tempo model)."""

    name = "classic-live"
    healthy = True

    def __init__(self, sr: int, policy) -> None:
        self.analyzer = LiveAnalyzer(sr, policy)
        self.bpm = 0.0
        self.confidence = 0.0
        self.duration = 0.0
        self.scenes: list[tuple[float, str]] = []
        self._prev = 0.0
        self._period_phase = 0.0

    def push(self, block: np.ndarray) -> None:
        self.analyzer.push(block)

    def frame(self, now: float, policy) -> AnalysisFrame:
        self.analyzer.policy = policy
        raw, bands = self.analyzer.read(now)
        level = shape_level(raw, policy)
        accent = float(np.clip((raw - self._prev) * 4.0, 0.0, 1.0))
        self._prev = raw
        return AnalysisFrame(
            t=max(now, 0.0), level=level, bands=bands, bpm=0.0, beat_phase=0.0,
            energy=raw, accent=accent, beat_confidence=0.0, scene="live",
            scene_confidence=0.3, engine=self.name,
        ).clamped()


# register the plugin chain (import order is handled by core.engine.build_*)
register("lightengine:offline", "perceptual", PerceptualOfflineEngine)
register("lightengine:offline", "classic", ClassicOfflineEngine)
register("lightengine:live", "perceptual", PerceptualLiveEngine)
register("lightengine:live", "classic", ClassicLiveEngine)
