"""Optional LLM agent backend (function calling / JSON), same env as agent.py.

The guide's "Agent 策略" calls for ``OpenDev LLM + function calling`` while the
real-time loop stays local.  This module only runs at *configuration* time:
user text -> one short chat call -> tool calls / JSON -> Policy.  If the env
vars are missing or the call fails, ``agent.apply_text`` falls back to the
offline rule parser, so the demo never depends on the network.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from ..config import Policy
from .palette import normalise

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_lighting",
            "description": "Set the screen lighting policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string",
                             "enum": ["dim", "canvas", "dj", "wake"]},
                    "base": {"type": "number"},
                    "gain": {"type": "number"},
                    "attack_ms": {"type": "number"},
                    "decay_ms": {"type": "number"},
                    "gamma": {"type": "number"},
                    "max_dim": {"type": "number"},
                    "safety": {"type": "string",
                               "enum": ["off", "comfort", "strict"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_palette",
            "description": "Set the light palette from a mood or explicit hex colors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {"type": "string"},
                    "colors": {"type": "array",
                               "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_response",
            "description": "Tune beat following / curve contrast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "beat_weight": {"type": "number"},
                    "curve_contrast": {"type": "number"},
                },
            },
        },
    },
]

SYSTEM = """You control a music-driven screen-light demo. Convert the user's
one-sentence request into function calls or a JSON object. Prefer functions.
mode: dim|canvas|dj|wake. base/gain 0..1. attack_ms 1..2000, decay_ms 1..4000,
gamma 0.2..3, max_dim 0..0.98, safety off|comfort|strict. beat_weight and
curve_contrast -1..1 (-1 = auto). palette colors are hex strings.
"""


def available() -> bool:
    return bool(os.getenv("OPENDEV_BASE_URL", "").strip()
                and os.getenv("OPENDEV_API_KEY", "").strip())


def _post(payload: dict, timeout: float) -> dict:
    base = os.getenv("OPENDEV_BASE_URL", "").rstrip("/")
    key = os.getenv("OPENDEV_API_KEY", "")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "OpenRay/1.8 (+https://github.com/)"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _merge(policy: Policy, data: dict) -> tuple[Policy, list[str]]:
    merged = policy.to_dict()
    notes: list[str] = []
    for key in ("mode", "base", "gain", "attack_ms", "decay_ms", "gamma",
                "max_dim", "safety", "beat_weight", "curve_contrast"):
        if key in data and data[key] is not None:
            merged[key] = data[key]
    palette = normalise(data.get("colors") or [])
    if palette:
        merged["palette"] = palette
        notes.append("AI 配色")
    if data.get("mood"):
        from .palette import MOODS
        if data["mood"] in MOODS:
            merged["palette"] = list(MOODS[data["mood"]][1])
            notes.append(f"AI 配色·{MOODS[data['mood']][0]}")
    p = Policy.from_dict(merged).clamp()
    if "mode" in data:
        notes.append(f"模式 {data['mode']}")
    return p, notes


def apply(text: str, policy: Policy, timeout: float = 8.0
          ) -> tuple[Policy, str, str] | None:
    """Return (policy, note, backend) or None when unavailable/failed."""
    if not available():
        return None
    model = os.getenv("OPENDEV_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.2,
        "tool_choice": "auto",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"current policy: {json.dumps(policy.to_dict(), ensure_ascii=False)}\n"
                f"request: {text}"},
        ],
    }
    try:
        body = _post(payload, timeout)
        message = body["choices"][0]["message"]
        p = policy
        notes: list[str] = []
        calls = message.get("tool_calls") or []
        for call in calls:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            name = fn.get("name")
            if name == "set_palette":
                mood = args.get("mood")
                if mood:
                    from .palette import MOODS
                    if mood in MOODS:
                        p.palette = list(MOODS[mood][1])
                        notes.append(f"AI 配色·{MOODS[mood][0]}")
                colors = normalise(args.get("colors") or [])
                if colors:
                    p.palette = colors
                    notes.append("AI 配色")
            elif name == "set_lighting":
                p, n2 = _merge(p, args)
                notes += n2
            elif name == "set_response":
                p, n2 = _merge(p, args)
                notes += n2
        if not calls:
            content = (message.get("content") or "").strip()
            content = re.sub(r"^```(?:json)?|```$", "", content,
                             flags=re.MULTILINE).strip()
            data = json.loads(content)
            p, notes = _merge(p, data)
        return p.clamp(), "、".join(notes) or "LLM 已应用", "llm-tools"
    except Exception:  # noqa: BLE001 - offline / quota / parse -> rules
        return None


def probe(timeout: float = 20.0) -> dict:
    """Connectivity check for the OpenAI-compatible endpoint (``--llm-test``)."""
    base = os.getenv("OPENDEV_BASE_URL", "").rstrip("/")
    key = os.getenv("OPENDEV_API_KEY", "")
    model = os.getenv("OPENDEV_MODEL", "gpt-4o-mini")
    info = {
        "base": base or "(missing OPENDEV_BASE_URL)",
        "model": model,
        "key": (key[:6] + "..." + key[-4:]) if key else "(missing OPENDEV_API_KEY)",
    }
    if not base or not key:
        return {**info, "ok": False, "error": "set OPENDEV_BASE_URL and OPENDEV_API_KEY"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "只回复两个字：收到"}],
    }
    try:
        body = _post(payload, timeout)
        content = body["choices"][0]["message"].get("content", "")
        return {**info, "ok": True, "reply": content.strip()}
    except Exception as exc:  # noqa: BLE001
        return {**info, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


EMOTION_SYSTEM = """You are a music emotion analyst for a light show.
Given the track title and acoustic features, estimate the *perceived* emotion.
Reply with strict JSON only:
{"label":"明亮|温暖|激烈|忧伤|平衡","valence":0.0..1.0,"arousal":0.0..1.0,"reason":"<=20 chars"}
valence: dark/sad -> low, happy/warm -> high. arousal: calm -> low, intense -> high.
"""


def emotion_hint(title: str, features: dict, timeout: float = 12.0) -> dict | None:
    """LLM reads acoustic features + title once per track (never per frame)."""
    if not available():
        return None
    model = os.getenv("OPENDEV_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EMOTION_SYSTEM},
            {"role": "user", "content":
                f"title: {title}\nfeatures: {json.dumps(features, ensure_ascii=False)}"},
        ],
    }
    try:
        body = _post(payload, timeout)
        content = body["choices"][0]["message"].get("content") or "{}"
        data = json.loads(content)
        label = str(data.get("label", "")).strip()
        if label not in ("明亮", "温暖", "激烈", "忧伤", "平衡"):
            return None
        return {
            "label": label,
            "valence": float(min(max(float(data.get("valence", 0.5)), 0.0), 1.0)),
            "arousal": float(min(max(float(data.get("arousal", 0.5)), 0.0), 1.0)),
            "reason": str(data.get("reason", ""))[:24],
            "model": model,
        }
    except Exception:  # noqa: BLE001 - optional, never break playback
        return None
