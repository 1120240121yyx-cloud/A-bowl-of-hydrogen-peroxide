"""Mood -> palette JSON (P1 in the guide: ''「落日感 / 赛博朋克」→ 画布配色变'').

The local library is instant and offline.  ``agent.py`` can also ask the LLM
for a palette; anything the LLM returns is validated through
:func:`normalise` so a bad answer can never crash the renderer.
"""
from __future__ import annotations

import re

MOODS = {
    "sunset": ("落日", ("#FF7A18", "#FF3D5A", "#B36BFF", "#FFC46B")),
    "cyber": ("赛博朋克", ("#00F0FF", "#FF2BD6", "#7B5CFF", "#0B1026")),
    "forest": ("森林", ("#2EE59D", "#A8E063", "#1B7F5A", "#F2FFE0")),
    "ocean": ("海洋", ("#2E8BFF", "#00E5FF", "#6FE3FF", "#0A2A4A")),
    "sacred": ("神圣", ("#E8C46A", "#F6E7B0", "#3A5BA0", "#1A1B3A")),
    "blood": ("血腥", ("#C81E28", "#7A0F18", "#FF5A4E", "#14070A")),
    "carnival": ("狂欢", ("#FF3BD4", "#00F5D4", "#FFE600", "#1B0B2A")),
    "mono": ("极简", ("#FFFFFF", "#D8D8D8", "#8A8A8A", "#111111")),
    "aurora": ("极光", ("#5CF2D6", "#6AA8FF", "#B47CFF", "#0A1020")),
    "warm": ("暖色", ("#FF7A18", "#FF3D00", "#FFD166")),
    "cool": ("冷色", ("#00E5FF", "#3D5AFE", "#B388FF")),
}

_ALIASES = {
    "落日": "sunset", "夕阳": "sunset", "日落": "sunset", "黄昏": "sunset",
    "赛博": "cyber", "赛博朋克": "cyber", "cyberpunk": "cyber", "霓虹": "cyber",
    "森林": "forest", "自然": "forest", "清新": "forest",
    "海洋": "ocean", "海": "ocean", "蓝色": "ocean", "冷色": "cool",
    "神圣": "sacred", "教堂": "sacred", "金色": "sacred", "金": "sacred",
    "血腥": "blood", "暗黑": "blood", "血": "blood",
    "狂欢": "carnival", "派对": "carnival", "糖果": "carnival",
    "极简": "mono", "白色": "mono", "黑白": "mono",
    "极光": "aurora", "暖": "warm", "暖色": "warm",
}

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


def normalise(colors, limit: int = 5) -> list[str]:
    out: list[str] = []
    for c in colors or []:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if _HEX.match(c):
            out.append("#" + c.lstrip("#").upper())
        if len(out) >= limit:
            break
    return out


def mood_for_text(text: str) -> str | None:
    low = text.lower()
    for name, (_label, colors) in MOODS.items():
        if name in low:
            return name
    for alias, name in _ALIASES.items():
        if alias in text:
            return name
    return None


def palette_for_text(text: str) -> tuple[str, list[str]] | None:
    name = mood_for_text(text)
    if name is None:
        return None
    label, colors = MOODS[name]
    return label, list(colors)


def palette_json(name: str) -> dict:
    if name not in MOODS:
        name = "sunset"
    label, colors = MOODS[name]
    return {"mood": name, "label": label, "palette": list(colors)}


def list_moods() -> list[dict]:
    return [{"id": key, "label": label, "palette": list(colors)}
            for key, (label, colors) in MOODS.items()]
