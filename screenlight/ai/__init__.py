"""Optional AI plugins for OpenRay.

Every module here is optional: it is imported lazily, wrapped in try/except by
the caller, and always paired with a heuristic fallback so the demo never
depends on a model, an API key or a network connection.

    light_engine.py   perceptual brightness/beat engine (numpy only, always on)
    scene_yamnet.py   YAMNet ONNX scene classifier + spectral heuristic fallback
    kws.py            voice wake: sherpa-onnx KWS / Windows SAPI (offline)
    llm.py            optional OpenDev LLM with function calling
    palette.py        mood -> palette JSON (offline library)
    emotion.py        valence / arousal -> response curve
    models.py         model discovery + capability probe
"""
from __future__ import annotations


def capabilities() -> dict:
    from . import models

    return models.status()


def describe() -> str:
    cap = capabilities()
    parts = []
    parts.append("YAMNet" if cap["yamnet_model"] and cap["onnxruntime"]
                 else "scene:heuristic")
    parts.append("KWS:sherpa" if cap["sherpa_kws_model"] and cap["sherpa_onnx"]
                 else ("KWS:sapi" if cap["sapi_zh"] else "KWS:off"))
    parts.append("LLM:on" if _llm_on() else "LLM:rules")
    return " · ".join(parts)


def _llm_on() -> bool:
    try:
        from .llm import available
        return available()
    except Exception:  # noqa: BLE001
        return False
