"""Headless checks: analysis, safety limiter, agent parser, v3 light engine."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .agent import parse_rules
from .beat import beat_stats, onset_strength
from .config import ASSETS, Policy
from .core.engine import OFFLINE_KIND, SafeEngine, build_offline
from .features import analyze_offline, load_wav, shape_level
from .mapping import SafetyLimiter


def sparkline(values: np.ndarray, width: int = 72) -> str:
    chars = " .:-=+*#%@"
    if values.size == 0:
        return ""
    idx = np.linspace(0, values.size - 1, min(width, values.size)).astype(int)
    return "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in values[idx])


# ---------------------------------------------------------------- v3 metrics

def _strong_onsets(samples: np.ndarray, sr: int, threshold: float = 0.6) -> np.ndarray:
    frame, hop = 1024, 256
    if len(samples) < frame:
        return np.zeros(0)
    count = 1 + (len(samples) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[idx]
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame).astype(np.float32), axis=1))
    fps = sr / hop
    env = onset_strength(spec, sr, frame)
    if env.max() <= 0:
        return np.zeros(0)
    # 20 ms smoothing so one hit yields one peak, then normalise
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.6) ** 2)
    k /= k.sum()
    env = np.convolve(env, k, mode="same")
    env = env / env.max()
    out, last = [], -1e9
    for i in range(2, len(env) - 2):
        if env[i] > threshold and env[i] == env[i - 2:i + 3].max():
            t = i / fps
            if t - last > 0.10:
                out.append(t)
                last = t
    return np.asarray(out, dtype=np.float64)


def _light_peaks(level: np.ndarray, fps: float, min_rise: float = 0.10) -> np.ndarray:
    rise = max(1, int(round(0.12 * fps)))
    out, last = [], -1e9
    for i in range(rise, len(level) - 1):
        if level[i] >= level[i - 1] and level[i] > level[i + 1] \
                and level[i] - level[i - rise] >= min_rise:
            t = i / fps
            if t - last > 0.12:
                out.append(t)
                last = t
    return np.asarray(out, dtype=np.float64)


def engine_metrics(engine, policy: Policy, samples: np.ndarray, sr: int,
                   seconds: float = 20.0, fps: float = 60.0) -> dict:
    """Beat alignment + smoothness of the *brightness curve* (not the audio)."""
    n = max(4, int(min(seconds, len(samples) / sr) * fps))
    times = np.arange(n, dtype=np.float64) / fps
    levels = np.array([engine.frame_at(float(t), policy).level for t in times])
    d = np.diff(levels)
    onsets = _strong_onsets(samples, sr)
    peaks = _light_peaks(levels, fps)
    onbeat = 0.0
    if len(peaks) and len(onsets):
        hits = sum(1 for p in peaks if float(np.min(np.abs(onsets - p))) <= 0.09)
        onbeat = hits / len(peaks)
    dd = np.diff(levels, n=2) if len(levels) > 2 else np.zeros(1)
    return dict(
        onbeat=onbeat,
        flash_rate=len(peaks) / max(times[-1], 1e-6),
        smooth_p95=float(np.percentile(np.abs(d), 95)) if d.size else 0.0,
        hard_steps=float(np.sum(np.abs(d) > 0.25) / max(times[-1], 1e-6)),
        jerk=float(np.mean(np.abs(dd))) * 1000.0,
        level_mean=float(np.mean(levels)),
        level_range=float(np.max(levels) - np.min(levels)),
    )


def _smooth_test_signal(sr: int = 44100, seconds: float = 8.0) -> np.ndarray:
    """Slow pad: no transients, no percussion -> must not produce beat accents."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    x = (np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 277.2 * t)
         + 0.3 * np.sin(2 * np.pi * 329.6 * t))
    env = 0.5 - 0.5 * np.cos(2 * np.pi * t / seconds * 2.0)
    return (0.25 * x * env).astype(np.float32)



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="screenlight --selftest")
    p.add_argument("--wav", default=str(ASSETS / "demo_120bpm.wav"))
    args = p.parse_args(argv)

    print("=" * 68)
    print("OpenRay v2.3 · selftest")
    print("=" * 68)

    policy = Policy().clamp()
    t0 = time.perf_counter()
    samples, sr = load_wav(args.wav)
    analysis = analyze_offline(samples, sr, policy)
    dt = time.perf_counter() - t0

    print(f"wav        : {args.wav}")
    print(f"duration   : {analysis.duration:.2f}s   sr={sr}   frames={len(analysis.level)}   fps={analysis.fps:.1f}")
    print(f"analysis   : {dt * 1000:.0f} ms   ({analysis.duration / max(dt, 1e-6):.1f}x realtime)")
    print(f"bpm        : {analysis.bpm:.1f}  (expected 120)")
    bcount, bmed, bstd = beat_stats(analysis.beats, analysis.fps)
    print(f"beats      : {int(bcount)}  median={bmed * 1000:.0f} ms  std={bstd * 1000:.0f} ms")
    print(f"level min/mean/max : {analysis.level.min():.3f} / {analysis.level.mean():.3f} / {analysis.level.max():.3f}")
    for safety in ("off", "comfort", "strict"):
        pp = Policy(safety=safety).clamp()
        shaped = np.array([shape_level(v, pp) for v in analysis.level])
        print(
            f"shaped {safety:8s}: min={shaped.min():.2f} p5={np.percentile(shaped, 5):.2f} "
            f"p50={np.percentile(shaped, 50):.2f} p95={np.percentile(shaped, 95):.2f} max={shaped.max():.2f}"
        )
    print(f"envelope   : |{sparkline(analysis.level)}|")

    # beat grid stability (jitter of the intervals, not absolute phase)
    if analysis.beats.size > 1:
        intervals = np.diff(analysis.beats) / analysis.fps
        med = float(np.median(intervals))
        jitter = np.abs(intervals - med)
        print("beat times : " + ", ".join(f"{b / analysis.fps:.2f}" for b in analysis.beats[:12]))
        print(f"beat jitter: mean={jitter.mean() * 1000:.0f} ms  max={jitter.max() * 1000:.0f} ms")

    print("scenes     : " + ", ".join(f"{t:.1f}s={name}" for t, name in analysis.scenes[:8]))

    # safety limiter
    for level in ("off", "comfort", "strict"):
        lim = SafetyLimiter(level)
        out, now, dt = [], 0.0, 0.01
        for i in range(600):  # 6 s, 20 Hz square wave
            now += dt
            out.append(lim.step(1.0 if (i // 2) % 2 == 0 else 0.0, now))
        tail = out[-200:]
        rate = max(abs(b - a) / dt for a, b in zip(out, out[1:]))
        print(f"limiter {level:8s}: tail p-p={max(tail) - min(tail):.2f}  max rate={rate:5.1f}/s  clamped={lim.clamped_events}")

    # ---- v3 light engine: A/B vs classic, beatless material, fallback chain
    pol_off = Policy(safety="off").clamp()
    if analysis.duration >= 4.0:
        t0 = time.perf_counter()
        perc = build_offline(samples, sr, pol_off)
        dt_engine = (time.perf_counter() - t0) * 1000.0
        segs = [(round(float(a), 1), round(float(b), 1), round(float(p), 1), round(float(c), 2))
                for a, b, p, c in zip(perc.curve.seg_t0, perc.curve.seg_t1,
                                      perc.curve.seg_bpm, perc.curve.seg_conf)]
        print(f"engine     : {perc.active_name}  built in {dt_engine:.0f} ms  "
              f"fallbacks={len(perc.failures)}")
        print("tempo segs : " + ", ".join(f"[{a}-{b}s {p:.0f}bpm c{c:.2f}]" for a, b, p, c in segs[:6]))
        classic = build_offline(samples, sr, Policy(engine="classic", safety="off").clamp())
        m_new = engine_metrics(perc, pol_off, samples, sr)
        m_cls = engine_metrics(classic, pol_off, samples, sr)
        for name, m in (("perceptual", m_new), ("classic", m_cls)):
            print(f"  {name:10s}: onbeat={m['onbeat']:.2f}  flash={m['flash_rate']:.2f}/s  "
                  f"p95Δ={m['smooth_p95']:.3f}  hard={m['hard_steps']:.2f}/s  "
                  f"jerk={m['jerk']:.1f}  level={m['level_mean']:.2f}±{m['level_range'] / 2:.2f}")
        ab_ok = (m_new["onbeat"] >= m_cls["onbeat"] - 1e-6
                 and m_new["jerk"] <= max(m_cls["jerk"] * 0.85, 1.0)
                 and m_new["hard_steps"] <= m_cls["hard_steps"] + 1e-6)
        print(f"  A/B       : {'OK' if ab_ok else 'WARN'}  "
              f"(onbeat>=classic, jerk<=85% classic, no extra hard steps)")

        smooth = _smooth_test_signal(sr=sr)
        eng_s = build_offline(smooth, sr, pol_off)
        m_s = engine_metrics(eng_s, pol_off, smooth, sr, seconds=8.0)
        smooth_ok = (m_s["hard_steps"] <= 0.2
                     and (eng_s.curve.confidence < 0.50 or eng_s.curve.beats.size == 0))
        print(f"  beatless  : conf={eng_s.curve.confidence:.2f}  flash={m_s['flash_rate']:.2f}/s  "
              f"p95Δ={m_s['smooth_p95']:.3f}  hard={m_s['hard_steps']:.2f}/s  -> "
              f"{'OK' if smooth_ok else 'WARN'}")
    else:
        print("engine     : skipped (wav shorter than 4 s)")

    bad = SafeEngine(OFFLINE_KIND, "definitely-not-an-engine",
                     lambda f, a, k: f(*a, **k), samples, sr, pol_off)
    fb_ok = bool(bad.healthy and bad.active_name in ("perceptual", "classic") and bad.failures)
    print(f"fallback   : requested=bogus -> active={bad.active_name}  "
          f"failures={len(bad.failures)}  -> {'OK' if fb_ok else 'WARN'}")

    # agent
    cases = [
        "暖色，跟着鼓点，别太刺眼",
        "慢一点，呼吸感，蓝色",
        "打开画布模式，亮度70%",
        "喊一声就显示：欢迎来到 ACC",
        "关闭护栏",
    ]
    base = Policy().clamp()
    print("agent      :")
    for text in cases:
        new, note = parse_rules(text, base)
        print(f"  · {text}")
        print(f"    -> mode={new.mode} palette={new.palette[0]} attack={new.attack_ms:.0f} "
              f"decay={new.decay_ms:.0f} max_dim={new.max_dim:.2f} safety={new.safety} text={new.text!r}")
        print(f"       {note}")

    print(f"shape_level: base={shape_level(0.0, base):.2f} peak={shape_level(1.0, base):.2f}")
    print("=" * 68)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
