# OpenRay 架构说明（v1）

## 1. 总览

```
┌──────────────── 音源层 IAudioSource ────────────────┐
│  InAppSource      内置 wav，自有时钟，零依赖         │
│  MicSource        sounddevice，外部声音              │
│  LoopbackSource   pyaudiowpatch，WASAPI 系统内录     │
└──────────────────────────┬───────────────────────────┘
                           │ PCM (mono float32)
                           ▼
┌──────────────── 特征层 features.py ──────────────────┐
│  RMS · 低频 punch · 对数频带 · BPM · 场景启发式      │
│  输出：env ∈ [0,1] + bands[24]                      │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────── 策略层 config.py / agent.py ─────────┐
│  Policy: base / gain / attack / decay / gamma        │
│  Agent: 自然语言 → Policy（规则 + 可选 LLM）         │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────── 安全层 mapping.py ───────────────────┐
│  一阶低通（max_hz）+ 变化率限制（slew）              │
│  三档：off / comfort / strict                        │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────── 渲染层 renderer.py ──────────────────┐
│  dim: 全屏黑色遮罩，alpha = (1-level) * max_dim      │
│  canvas: 背景色随 level 变亮 + 28 根彩色音波条       │
│  wake: 背景色随 level 变亮 + 大字 + 副标题           │
└──────────────────────────────────────────────────────┘
```

所有层之间只传数值与 `Policy`，不互相依赖平台 API —— 这是后续移植 Android 的基础。

## 2. 关键接口

### 音源

```python
class BaseSource:
    key: str                  # inapp | mic | loop
    label: str
    available: bool
    reason: str               # 不可用原因，直接显示在面板
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def position(self) -> float: ...          # 仅 InAppSource
    def latest(self, n: int) -> np.ndarray | None: ...  # 仅流式音源
```

新增音源只需要实现这 5 个成员，主循环不用改。

### 策略

```json
{
  "mode": "dim",
  "base": 0.08,
  "gain": 0.92,
  "attack_ms": 6,
  "decay_ms": 130,
  "gamma": 0.95,
  "max_dim": 0.96,
  "palette": ["#FF7A18", "#FF3D00", "#FFD166"],
  "text": "看见声音",
  "safety": "comfort"
}
```

`base + gain * env` 先算目标亮度，再取 `gamma`，然后过「黑场压缩」，最后进安全层。

### 2.1 亮度映射（v1.1）

```python
# features.py
low_flux = 低频(<=260Hz) 的频谱通量
punch    = envelope(0.75*low_flux + 0.25*broadband_flux, attack=3ms, decay=~80ms)
body     = envelope(rms, attack=6ms, decay=130ms)
env      = clip(0.18*body + 0.95*punch, 0, 1)

# features.shape_level()
v = (base + gain*env) ** gamma
black = {"off": 0.16, "comfort": 0.06, "strict": 0.0}[safety]
v = clip((v - black) / (1 - black), 0, 1)
```

- **低频 punch** 让屏幕跟底鼓/贝斯闪，而不是被军鼓/镲片带偏
- **黑场压缩** 在关闭护栏时最深，实测 alpha 0.06–0.81（屏幕亮度 19%–94%）
- 舒适档实测 alpha 0.19–0.64；严格档更平缓

### 安全层参数

```python
SAFETY_PRESETS = {
  "off":     {"max_hz": 0, "slew": 80},   # 无低通、无变化率限制
  "comfort": {"max_hz": 3, "slew": 5},
  "strict":  {"max_hz": 1, "slew": 1},
}
```

- `max_hz`：一阶低通截止频率，0 表示旁路
- `slew`：每秒允许的最大亮度变化量
- 实测（20 Hz 方波输入）：off p-p=1.00，comfort=0.10，strict=0.02

## 3. 延迟预算

| 环节 | 目标 |
|---|---|
| WASAPI loopback 缓冲 | 10–30 ms |
| FFT 帧（1024 @ 44.1k） | 23 ms |
| 平滑 / 映射 | < 1 ms |
| 遮罩 alpha 更新 | 1 帧（16 ms） |
| **合计** | **< 60 ms** |

内置曲目模式不走采集，直接读预计算曲线，延迟只取决于渲染帧率。

## 4. AI 接入点（v2）

### 4.1 场景识别 → YAMNet

替换 `features.classify_scenes()`：

```python
# 模型：YAMNet ONNX（约 3.7 MB，AudioSet 521 类）
# 输入：16 kHz 单声道，0.975 s 窗口
# 输出：521 类分数 → 映射到 music / speech / applause / noise
```

建议放在独立线程，每 0.5 s 推理一次，不阻塞渲染。

### 4.2 语音唤醒 → sherpa-onnx KWS

替换 `App.auto_wake()` 的能量阈值：

```python
# 模型：sherpa-onnx 中文关键词（如「亮起来」「开灯」）
# 输入：16 kHz，流式
# 输出：命中关键词 → trigger_wake()
```

能量阈值保留为兜底（模型未加载时）。

### 4.3 节拍跟踪 → madmom

替换 `features` 里的 `flux` 峰值：

```python
# madmom.features.beats.RNNBeatProcessor
# 输出 beat 时间点 → 直接驱动 envelope
```

v1 用频谱通量 + 自相关估 BPM，120 BPM 测试曲误差约 16 ms。

### 4.4 Agent → OpenDev

`agent.ask_opendev()` 已实现 OpenAI 兼容的 `/chat/completions` 调用，
输出 JSON 策略并与当前策略合并。没有 Key 时自动回退规则解析。

## 5. 多人联动（未实现，设计已定）

**主机编排 + 从机当灯泡**，不做 P2P：

```
笔记本（Host）
├── 音频分析
├── WebSocket Server  ws://<lan-ip>:8765
└── 二维码 http://<lan-ip>:8000/join?room=ACC

手机（浏览器从机）
└── 收到 {"type":"wave","period":120,"color":"#FF7A18"} → 全屏变色
```

- 主机按 index 下发延迟，实现「依次亮 / 波浪 / 全体脉冲」
- 现场用笔记本热点，避免场馆 Wi-Fi 的 AP 隔离
- 从机用 `navigator.wakeLock` 防息屏
- 后续再做 App 从机、时钟校准、P2P 接棒

## 6.5 内置音频引擎（下载即用）

`vendor/pyaudiowpatch/` 预解压了 cp310–cp315 的 `pyaudiowpatch`（含 `_portaudiowpatch.*.pyd`）。
`screenlight/vendor.py` 按 `sys.version_info` 选择对应目录并加入 `sys.path`，因此：

- 不需要 pip install
- 麦克风和 WASAPI loopback 都能用
- 只支持 Windows x64 + Python 3.10–3.15

麦克风打不开时，优先检查 Windows 隐私设置里的「允许桌面应用访问麦克风」。

## 6. Android 移植要点

| 能力 | 方案 | 风险 |
|---|---|---|
| 应用内播放 | ExoPlayer / AudioTrack，PCM 双写 | 无 |
| 麦克风 | `AudioRecord` + `RECORD_AUDIO` | 无 |
| 系统内录 | `MediaProjection` + `AudioPlaybackCapture` | 播放方 App 可拒绝；DRM 不可采 |
| 调光 | `SYSTEM_ALERT_WINDOW` 悬浮遮罩（快）/ `WRITE_SETTINGS`（慢） | 需用户授权 |
| 前台服务 | Android 14+ `FOREGROUND_SERVICE_MEDIA_PROJECTION` | 必须 |

iOS：第三方 App 既不能采其他 App 音频，也不能全局调亮度 → 不做。

## 7. 设计取舍

- **为什么用遮罩而不是改背光**：背光 API 慢（50–200 ms）、外接屏兼容差；遮罩 16 ms 一帧，任何屏幕都能跑。背光可作为 v2 增强。
- **为什么内置曲目要「PCM 双写」**：播放和分析同一份数据，天然零延迟，不依赖任何采集权限。
- **为什么护栏做低通而不是硬截断**：硬截断会产生台阶感；低通在视觉上更自然，同时把高频能量压下去。
- **为什么 LLM 不进实时环**：实时环 60 fps，LLM 延迟 300–2000 ms。LLM 只做「配置」，实时靠本地规则。
