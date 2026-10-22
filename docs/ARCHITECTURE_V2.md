# OpenRay 架构 v2（可持续演进版）

> 目标：音源、分析、策略、渲染四条管线可替换；AI 作为插件挂在分析/策略两侧；平台差异只存在于 platform 层。

---

## 1. 现状（v1.1）

```
app.py            控制器 / 控制面板 / 主循环 / 快捷键
├── audio.py      InAppSource / MicSource / LoopbackSource
├── features.py   RMS / 低频 punch / BPM / 场景启发式
│   └── beat.py   Ellis DP 节拍跟踪（新）
├── mapping.py    亮度映射 / 三档安全护栏 / 配色
├── renderer.py   dim / canvas / wake 全屏渲染
├── agent.py      规则解析 + 可选 OpenDev LLM
├── config.py     Policy 数据类
├── vendor.py     内置 pyaudiowpatch 加载器
└── selftest.py   无界面自检
```

依赖方向：

```
app ──> audio / features / mapping / renderer / agent / config
features ──> beat / config
renderer ──> mapping
audio ──> vendor
```

**问题**（不是现在就必须改，但要知道）：

- `app.py` 同时管 UI、编排、采样、状态机，加 Android 会膨胀
- `features.py` 混了 onset、tempo、beat、scene 四件事
- `renderer.py` 三种模式共用一套 Tk 逻辑，加硬件亮度会继续膨胀

---

## 2. 目标分层

```
core/          时钟、编排、事件、插件注册表
sources/       AudioSource 实现（内置 / 麦克风 / 内录）
analysis/      Analyzer 链（onset → tempo → beat → scene）
policy/        Policy、Safety、Agent
render/        DimRenderer / CanvasRenderer / WakeRenderer
platform/      Windows overlay、硬件亮度、Android
ai/            KWS、场景模型、LLM 适配器
```

### 依赖规则（很重要）

1. `core` 只依赖 **Protocol 接口**，不 import 任何具体实现
2. 具体实现通过 `registry.register("source", "mic", MicSource)` 注册
3. `ai/*` 只被 `analysis` / `policy` 调用，**加载失败必须 fallback**
4. `platform/*` 只被 `render` 调用，业务层不碰 Win32/Android API
5. 禁止 `sources` 反向 import `app`

### 接口草案

```python
class AudioSource(Protocol):
    key: str
    label: str
    available: bool
    reason: str
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, n: int) -> np.ndarray | None: ...
    def position(self) -> float: ...

class Analyzer(Protocol):
    def process(self, samples: np.ndarray, now: float) -> AnalysisFrame: ...

class Renderer(Protocol):
    def render(self, frame: AnalysisFrame, policy: Policy) -> None: ...

class Agent(Protocol):
    def apply(self, text: str, policy: Policy) -> Policy: ...
```

`AnalysisFrame` 统一承载：

```python
@dataclass
class AnalysisFrame:
    level: float          # 0..1 目标亮度
    bands: np.ndarray     # 画布频谱
    bpm: float
    beat_phase: float     # 0..1，当前拍内相位
    scene: str            # music / speech / applause
    confidence: float
    wake: bool            # 是否触发唤醒
```

---

## 3. 新增模块怎么做（以 YAMNet 为例）

1. 新建 `ai/scene_yamnet.py`，实现 `SceneClassifier` 接口
2. 在 `core/registry.py` 注册 `scene:yamnet`
3. 配置里写 `"scene": "yamnet"`，或启动时探测模型文件自动启用
4. 模型缺失 / 推理失败 → 自动回退 `scene:heuristic`
5. **不改** `app.py`、`renderer.py`、`audio.py`

新增音源同理：实现 `AudioSource` → 注册 → 配置里选。新增渲染器同理。

---

## 4. AI 现在在哪、什么时候加

### 现状（v1.1）

| 模块 | 现在是什么 | 是不是 AI |
|---|---|---|
| Agent 自然语言 | 规则解析 + 可选 OpenDev LLM | **部分是真 AI**（配了 Key 才是 LLM） |
| 语音唤醒 | 能量阈值（`App.auto_wake`） | ❌ 不是 AI |
| 场景识别 | 频谱启发式（`features.classify_scenes`） | ❌ 不是 AI |
| 节拍 | Ellis DP + 相位对齐（`beat.py`） | ❌ 经典算法，不是 ML |
| 配色 | 规则 | ❌ |

### 正式版规划

| 模块 | 正式方案 | 依赖 | 建议时间 |
|---|---|---|---|
| Agent 策略 | OpenDev LLM + function calling | 已有 | ✅ 已接 |
| 语音唤醒 | sherpa-onnx 中文 KWS | ONNX 模型 ~10MB | **优先**，10.24 后 |
| 场景识别 | YAMNet ONNX（521 类） | ~4MB | 10.24 后 |
| 节拍 | Ellis DP（已够用） | 无 | ✅ 已接 |
| 下拍/小节 | madmom / Beat-Transformer | 重 | 以后 |
| 情绪/配色 | CLAP 零样本 / LLM | 中 | 以后 |

**结论**：是的，AI 模型是「初步原型跑通后」再作为插件加进去。原因不是偷懒，而是：

- 比赛截止 10.22，核心 demo 不能依赖模型下载/推理稳定性
- 插件 + fallback 的结构保证：模型挂了，demo 仍然能跑
- KWS 最直观，优先级最高；YAMNet 次之

---

## 5. 系统屏幕亮度：能不能直接调？

### 四条路

| 方案 | 覆盖范围 | 延迟 | 权限 | 问题 |
|---|---|---|---|---|
| **全屏遮罩（现在）** | 所有屏幕/窗口 | 16ms | 无 | 只压暗，不减背光功耗 |
| **Gamma Ramp** | 整个桌面 | 快 | 无 | 改色彩对比，不是真背光 |
| **WMI** `WmiMonitorBrightnessMethods` | 笔记本内屏 | 50–200ms | 无 | 外接屏不支持，调用会阻塞 |
| **DDC/CI** `SetMonitorBrightness` | 支持 DDC 的外接屏 | 20–100ms | 无 | 显示器兼容性参差 |

### 推荐（以后做）

**混合模式**：

- 节拍级快速变化（60fps）→ 继续用**遮罩**
- 慢速环境光（≤2Hz）→ 调用 **WMI / DDC** 改真实背光
- 启动时探测能力，失败自动回退遮罩
- 放在 `platform/brightness.py`，实现 `BrightnessBackend` 接口

**现在不做**：WMI 调用会阻塞主线程、外接屏兼容性不可控，demo 前不值得冒险。

---

## 6. 节拍算法：为什么之前像“乱闪”

旧版是**反应式包络**：只要频谱有能量上升就亮，于是底鼓、军鼓、镲片、人声都触发，实测约 **5–8 次/秒**，看起来没有节拍感。

v1.2 换成：

1. **对数压缩多频带 onset**（SuperFlux 思路，`beat.onset_strength`）
2. **Ellis (2007) 动态规划节拍跟踪**（`beat.track_beats`）
3. **相位对齐**：把每个拍点吸附到最近的 onset 峰值（`beat.refine_beats`）
4. **拍点脉冲包络**：每拍一个指数衰减，每 4 拍重音（`beat.pulse_envelope`）

实测（你的 demo 音频，60s）：

```
bpm        : 120.2
beats      : 120
median     : 499 ms
std        : 14 ms
jitter     : 8 ms
flash rate : 2.0 /s   ← 正好等于 120 BPM / 60
```

参考：D. P. W. Ellis, *Beat Tracking by Dynamic Programming*, JNMR 2007；
librosa 的 `beat_track` 属于同一算法家族（ISC 许可）。本实现是 NumPy 重写，无额外依赖。

---

## 7. 迁移计划（不重写，按模块抽）

| 阶段 | 做什么 | 风险 |
|---|---|---|
| **现在（0.5 天）** | 把 `features.py` 拆成 `analysis/onset.py`、`tempo.py`、`beats.py`、`scene.py`，保持函数签名不变 | 低 |
| 10.24 后 | 加 `ai/kws.py` + `ai/scene_yamnet.py`，注册 + fallback | 中 |
| 下个版本 | 拆 `sources/`、`render/`；加 `platform/brightness.py` | 中 |
| Android | 实现 `AudioSource` / `Renderer` 的 Android 版，复用 `analysis` / `policy` | 高 |

**原则**：先有插件点，再拆文件；不要为了“架构好看”在比赛前大改。
