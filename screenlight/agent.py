"""Agent layer: natural language -> lighting policy.

Two backends:
  * rule-based  (always on, offline, instant)
  * OpenDev LLM (optional, set OPENDEV_BASE_URL / OPENDEV_API_KEY / OPENDEV_MODEL)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import replace

from .config import Policy
from .mapping import SAFETY_LABELS
from .ai import llm as ai_llm
from .ai import palette as ai_palette

SYSTEM_PROMPT = """你是 OpenRay 的灯光策略助手。把用户的一句话转换成 JSON 灯光策略。
只输出 JSON，不要解释。字段与取值范围：
mode: dim|canvas|wake
base: 0..1        常亮底亮度
gain: 0..1        随声音起伏的增益
attack_ms: 1..2000
decay_ms: 1..4000
gamma: 0.2..3
max_dim: 0..0.98   画面最多压暗多少
palette: 2-4 个十六进制颜色
text: 唤醒时显示的文字
safety: off|comfort|strict

规则：用户说“别太刺眼/柔和”-> max_dim<=0.75, gamma>=1.0；
说“快/炸/激烈”-> attack_ms<=15, decay_ms<=140, gain>=0.9；
说“慢/呼吸/温柔”-> attack_ms>=50, decay_ms>=500, gain<=0.7。
"""

WARM = ["暖", "橙", "黄", "红", "落日", "夕阳", "温暖", "火"]
COOL = ["冷", "蓝", "青", "紫", "赛博", "科技", "冰"]
GREEN = ["绿", "自然", "森林", "清新"]
WHITE = ["白", "极简", "干净", "纯色"]

FAST = ["快", "激烈", "炸", "嗨", "强", "猛烈", "动感"]
SLOW = ["慢", "柔", "呼吸", "舒缓", "安静", "温柔", "轻"]

SOFT = ["别太刺眼", "不刺眼", "柔和", "低调", "暗", "护眼", "舒服"]

PALETTES = {
    "warm": ["#FF7A18", "#FF3D00", "#FFD166"],
    "cool": ["#00E5FF", "#3D5AFE", "#B388FF"],
    "green": ["#00E676", "#AEEA00", "#1DE9B6"],
    "white": ["#FFFFFF", "#E0E0E0", "#BDBDBD"],
}


def _has(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def parse_rules(text: str, policy: Policy) -> tuple[Policy, str]:
    p = replace(policy)
    notes: list[str] = []
    t = text.strip()

    mood = ai_palette.palette_for_text(t)
    if mood is not None:
        p.palette = list(mood[1])
        notes.append(f"AI 配色·{mood[0]}")
    elif _has(t, WARM):
        p.palette = list(PALETTES["warm"]); notes.append("暖色")
    elif _has(t, COOL):
        p.palette = list(PALETTES["cool"]); notes.append("冷色")
    elif _has(t, GREEN):
        p.palette = list(PALETTES["green"]); notes.append("自然绿")
    elif _has(t, WHITE):
        p.palette = list(PALETTES["white"]); notes.append("极简白")

    if _has(t, FAST):
        p.attack_ms, p.decay_ms, p.gain = 12.0, 130.0, 0.95
        notes.append("快速响应")
    elif _has(t, SLOW):
        p.attack_ms, p.decay_ms, p.gain = 60.0, 600.0, 0.62
        notes.append("缓慢呼吸")

    if _has(t, ["更炸", "再猛", "幅度大", "剧烈", "全力", "拉满", "更闪"]):
        p.base, p.gain = 0.02, 0.98
        p.attack_ms, p.decay_ms = 3.0, 100.0
        p.max_dim = 0.99
        p.gamma = 0.90
        notes.append("最大幅度")
    elif _has(t, ["温柔", "轻一点", "别太闪", "低调一点"]):
        p.base, p.gain = 0.14, 0.66
        p.attack_ms, p.decay_ms = 40.0, 420.0
        p.max_dim = min(p.max_dim, 0.68)
        notes.append("柔和幅度")

    if _has(t, SOFT):
        p.max_dim = min(p.max_dim, 0.72)
        p.gamma = max(p.gamma, 1.05)
        notes.append("柔和不刺眼")

    if _has(t, ["画布", "音波", "频谱", "波形"]):
        p.mode = "canvas"; notes.append("画布模式")
    elif _has(t, ["唤醒", "喊", "叫", "文字", "亮起来"]):
        p.mode = "wake"; notes.append("唤醒模式")
    elif _has(t, ["跟随", "氛围", "呼吸灯"]):
        p.mode = "dim"; notes.append("跟随模式")

    m = re.search(r"(?:屏幕|亮度|最亮)[^\d]{0,4}(\d{1,3})\s*%", t)
    if m:
        peak = min(max(int(m.group(1)) / 100.0, 0.05), 1.0)
        p.base = round(peak * 0.2, 3)
        p.gain = round(peak * 0.8, 3)
        notes.append(f"峰值亮度 {int(peak * 100)}%")

    m = re.search(r"(?:显示|写上|打字)[:：]?\s*[“\"「]?([^”\"」]{1,20})", t)
    if m:
        p.text = m.group(1).strip()
        p.mode = "wake"
        notes.append(f"显示「{p.text}」")

    if _has(t, ["关掉护栏", "关闭护栏", "不要护栏", "关闭保护"]):
        p.safety = "off"; notes.append("安全护栏关闭")
    elif _has(t, ["严格", "最安全"]):
        p.safety = "strict"; notes.append("严格护栏")

    return p.clamp(), ("、".join(notes) if notes else "保持当前策略")


def ask_opendev(text: str, policy: Policy, timeout: float = 8.0) -> tuple[Policy, str] | None:
    base = os.getenv("OPENDEV_BASE_URL", "").rstrip("/")
    key = os.getenv("OPENDEV_API_KEY", "")
    model = os.getenv("OPENDEV_MODEL", "gpt-4o-mini")
    if not base or not key:
        return None

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"当前策略：{json.dumps(policy.to_dict(), ensure_ascii=False)}\n用户：{text}"},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "OpenRay/1.8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        data = json.loads(content)
        merged = policy.to_dict()
        merged.update({k: v for k, v in data.items() if k in merged})
        return Policy.from_dict(merged), "LLM 已应用"
    except Exception:  # network / quota / parse -> fall back to rules
        return None


def apply_text(text: str, policy: Policy, use_llm: bool = True) -> tuple[Policy, str, str]:
    """Returns (policy, note, backend)."""
    if use_llm:
        got = ai_llm.apply(text, policy)
        if got is not None:
            return got[0], got[1], got[2]
        got = ask_opendev(text, policy)
        if got is not None:
            return got[0], got[1], "llm"
    p, note = parse_rules(text, policy)
    return p, note, "rules"


def describe(policy: Policy) -> str:
    return (
        f"{policy.mode} · 感度{int(policy.sensitivity * 100)}% · "
        f"底亮{int(policy.base_brightness * 100)}% · "
        f"{int(policy.attack_ms)}/{int(policy.decay_ms)}ms"
    )
