# OpenRay 架构 v3 — 接手 AI 的梳理与改造方案

> 版本：v3.0 草案（2026-10-20）｜ 适用：v1.3 代码库
> 原则：**不重写核心、旧入口全兼容、新增能力一律「插件 + fallback」**、每一步 selftest + 三模式 smoke 保持绿。

---

## 0. 结论摘要（先看这 5 行）

1. 代码量只有 ~1.7k 行，模块划分**不差**，但分析层**没有统一契约**：`app.py` 直接调用 `analyze_offline` / `LiveAnalyzer.read`，渲染层靠位置参数传递。以后每加一个模块（KWS、YAMNet、硬件亮度、多人联动）都要改 `app.py` —— 这是扩展性的最大瓶颈。
2. **离线 / 实时两套分析质量不一致**：内置曲目有 DP 拍点，系统内录/麦克风只有「反应式 punch + 90ms 不应期」。用户在「平滑、无鼓点」的音乐上听到/看到的是后者 —— 这就是「没踩在鼓点上 / 像用粗糙折线描述连续曲线」的直接根因。
3. **策略与特征耦合**：`analyze_offline(samples, sr, policy)` 把 attack/decay/黑场烧进了预先算好的曲线；Agent 改 Policy 只影响 limiter，不影响已算好的曲线 → 「温柔一点」对内置曲目不生效。
4. **`shape_level` 的硬黑场分界 + 接近线性的 gamma** 把连续能量曲线切成「平段 + 斜段」；`level_at()` 用最近邻取样（无插值）→ 阶跃感。
5. 画布只是「单色背景 + 28 根纯色矩形」，每帧 28×(`coords`+`itemconfig`) 无缓存 → 38–46fps 且观感普通。

**v3 = 3 个结构性改动 + 2 个算法改动 + 1 个质量回归机制**，全部增量：

| # | 改动 | 文件 | 解决问题 |
|---|---|---|---|
| S1 | `core/frame.py` 统一 `AnalysisFrame` + `core/registry.py` 插件注册表 + `SafeAnalyzer` fallback 链 | 新增 `screenlight/core/` | 扩展性 |
| S2 | `ai/light_engine.py` 统一光引擎：**连续能量主曲线 + 置信度门控的节拍重音**，离线/在线同一套算法 | 新增 `screenlight/ai/` | 没踩鼓点、折线感、off 档混乱 |
| S3 | 渲染层拆 `render/scene_canvas.py`（`CanvasScene`），`renderer.py` 只做窗口与调度 | 新增 `screenlight/render/` | 画布观感、帧率 |
| S4 | `shape_level` 改为 C1 平滑 S 曲线（去硬黑场死区）+ 子帧插值采样 | `features.py`、`app.py` | 分段折线感 |
| S5 | selftest 增加**视觉质量指标**（拍点对齐度 / 曲线平滑度 / 闪烁率 / 引擎回退次数）+ `--diag` 离线曲线导出 | `selftest.py`、`tools/diag_curve.py` | 以后改动画不回归 |

---

## 1. 现状盘点（v1.3）

```
app.py (386)            控制器 + Tk 面板 + tick 循环 + 状态机 + 热键分发 + 自动唤醒
├── audio.py (258)      InAppSource / MicSource / LoopbackSource / 设备枚举
├── features.py (265)   load_wav / analyze_offline / LiveAnalyzer / shape_level / estimate_bpm / classify_scenes
│   └── beat.py (117)   onset_strength / track_beats(Ellis DP) / refine_beats / pulse_envelope
├── mapping.py (88)     SafetyLimiter / 颜色工具
├── renderer.py (178)   Overlay：Win32 点击穿透 + dim/canvas/wake 三模式绘制
├── agent.py (143)      规则解析 + 可选 OpenDev LLM
├── config.py (54)      Policy dataclass + JSON 读写
├── hotkeys.py (67)     Win32 全局热键线程
├── vendor.py (20)      pyaudiowpatch 按 Python 版本加载
└── selftest.py (84)    无界面分析 + limiter + agent 检查
```

两条数据通路（当前）：

```
内置曲目:  load_wav -> analyze_offline(整首, policy) -> Analysis.level@172fps
            -> app.level_at(t) 最近邻 -> shape_level -> SafetyLimiter -> Overlay
实时音源:  ring buffer -> LiveAnalyzer.read()（RMS+低频 punch，90ms 不应期）
            -> level -> shape_level -> SafetyLimiter -> Overlay
```

---

## 2. 结构性问题清单（按影响排序）

### P1 缺少统一分析契约与插件注册表（扩展性第一瓶颈）
- `app.py` 同时知道 `analyze_offline`、`Analysis.level_at`、`LiveAnalyzer.read` 三种具体 API；`tick()` 里手工组装 `level/bands/mode/palette/text`。
- `ARCHITECTURE_V2.md` 里的 `core/registry`、`AnalysisFrame` 只停留在文档。
- 后果：加任何一个模块都要动 `app.py` 的主循环（最高风险区域）。

### P2 离线/实时两套质量（用户抱怨的直接来源）
- 离线：`beat.track_beats`（Ellis DP）+ `pulse_envelope`，拍点锁定。
- 实时：`LiveAnalyzer` 只有 `body`（RMS 跟随）+ `punch`（低频通量，90ms 不应期），**完全没有 tempo 模型**。平滑音乐里 punch 会被旋律音符/气口触发 → 亮度乱跳；有鼓点的音乐也只能「见峰就闪」，5–8 次/秒。
- 用户在「正常听歌刷视频」场景（`run_background.bat` → 系统内录 → 实时路径）感受最强。

### P3 策略与特征耦合
- `analyze_offline(samples, sr, policy)` 把 `policy.attack_ms/decay_ms` 烧进曲线，黑场阈值在 `shape_level` 里按 `safety` 再切一刀。
- `App.on_agent()` 只 `self.live.policy = policy`，**不重算** `self.analysis`。
- 后果：Agent 说「温柔一点」→ 内置曲目亮度曲线不变，只有 limiter 变；语义不一致。
- 正确分层：**特征与策略无关**（能量/频带/beat grid 预计算），**shaping 每帧做**（依赖当前 Policy）。

### P4 硬黑场分界 + 最近邻取样 = 折线/阶跃感
```python
black = {"off": 0.16, ...}
value = min(max((value - black) / (1.0 - black), 0.0), 1.0)   # 低于阈值全部压成 0
```
- 连续能量 < 阈值时输出恒为 0（平段），超过后线性上升（斜段）→ 视觉上就是「用几段直线近似曲线」。
- `Analysis.level_at()`：`i = int(t * self.fps) % size`，最近邻、循环边界无插值 → 阶跃 + 循环点闪跳。
- gamma 默认 0.95 ≈ 线性，没有提供平滑的对比度压缩。

### P5 节拍重音「规则化」
- `pulse_envelope`：每 4 拍固定 `amp=1.0`，其余 `0.72`，与音乐真实重音无关（3/4 拍、切分、慢歌全错）。
- `tau=0.085s` 太短 → 观感是「闪光」而不是「力道/呼吸」；`off` 档下平均 90% 时间是黑的，只有 1–2 帧亮 → 乱。

### P6 渲染层单类三模式 + 扁平画布
- `Overlay` 200 行里混了 Win32 点击穿透、三模式显隐、逐帧动画。
- canvas：28 根 `create_rectangle`，每帧 28 次 `coords` + 28 次 `itemconfig`，无颜色缓存；背景是单个矩形纯色。38–46fps，观感是「频谱柱状图」而非「灯光画布」。
- 无光晕、无倒影、无峰值保持、无节拍扩散环、无渐变。

### P7 没有视觉质量回归测试
- `selftest` 只验证 BPM / 节拍 jitter / limiter / Agent 解析，**不验证**「亮度曲线是否踩拍」「曲线是否平滑」「每秒闪烁次数」。
- 因此 P2/P4/P5 这类问题改坏了也发现不了。

### P8 时钟与循环边界
- `InAppSource.position()` 用 `perf_counter() - t0`，`winsound` 的启动延迟未补偿；`% duration` 在循环点可能跳 1–2 帧。
- `level_at` 的 `% size` 与 `bands_at` 同理，循环点无交叉淡化。

### P9 配置/文档漂移
- `ARCHITECTURE_V2.md` 描述的目标分层未实现；`README.md` 的 canvas fps 没有测量支撑；`selftest.py` 重复 `import shape_level`；`PROJECT_STATE.json` 未跟随 v1.3 更新细节。
- `presets/default.json` 编码正常（已核对 codepoints），控制台乱码只是 chcp 显示问题，无需改。

### P10 扩展点缺失
- 没有 `sources/`、`analysis/`、`render/`、`platform/` 目录；WAV-only（用户的 `demo音源集/*.mp3` 是 M4A 容器，当前无法解码，只能靠内录）；没有分析缓存；热键表硬编码在 `app.py`。

---

## 3. 目标架构 v3（插件 + fallback）

```
screenlight/
  core/
    frame.py       AnalysisFrame（唯一跨层数据契约）
    registry.py    极简注册表：register(kind, name, factory) / resolve(kind, name, fallbacks)
    safe.py        SafeAnalyzer：try/except + 失败计数 + 自动降级（面板可见）
  ai/
    light_engine.py  PerceptualLightEngine（离线 + 在线）—— 本次核心算法
    kws.py           （后续）sherpa-onnx 关键词，失败回退能量阈值
    scene_yamnet.py  （后续）YAMNet，失败回退 heuristic
  render/
    scene_canvas.py  CanvasScene：渐变背景 / 胶囊柱 / 光晕 / 倒影 / 峰值 / 节拍环
  analysis/          （后续把 features.py 按 onset/tempo/beats/scene 拆出）
  platform/          （后续）brightness.py: BrightnessBackend（WMI/DDC，失败回退遮罩）
  —— 以下保持原文件，作为兼容层，不删不改签名 ——
  features.py / beat.py / mapping.py / renderer.py / audio.py / agent.py / config.py
```

### 3.1 唯一数据契约

```python
@dataclass
class AnalysisFrame:
    t: float                 # 播放位置（秒）
    level: float             # 0..1 目标亮度（已按 Policy shaping，未过 limiter）
    bands: np.ndarray        # (24,) 0..1 画布频带
    bpm: float
    beat_phase: float        # 0..1 拍内相位（画布扩散环用）
    energy: float            # 0..1 连续慢能量（呼吸感）
    accent: float            # 0..1 拍点重音（鼓点力度）
    beat_confidence: float   # 0..1；低 = 无鼓点/平滑音乐 → 重音权重自动降为 0
    scene: str               # music / speech / applause / live
    scene_confidence: float
    wake: bool
    engine: str              # 实际生效引擎名（诊断/面板显示）
```

`Overlay.render(frame, policy)` 是唯一渲染入口（旧方法签名保留兼容重载）。

### 3.2 插件 + fallback 链

```python
policy.engine = "auto"        # auto | perceptual | classic
# auto 解析顺序：
#   analyzer:offline -> ai.perceptual@offline -> classic@offline
#   analyzer:live    -> ai.perceptual@live    -> classic@live
# resolve 时每个实现包一层 SafeAnalyzer：
#   - 构造失败 / 首帧异常 / 连续 3 次异常 -> 记录原因，降级到下一个
#   - 面板状态栏显示 "引擎: perceptual" 或 "引擎: classic（perceptual 失败: ...）"
```

### 3.3 统一光引擎算法（S2，离线与在线共用同一套「三段式」）

```
                ┌─ 连续能量主曲线 energy(t) ───────────────┐
PCM → 特征 →    │  RMS→dB → 4–8s 自适应 AGC → attack/release │ → 混合 → shaping → limiter
                ├─ 节拍网格 beats(t) + confidence ──────────┤
                └─ 拍点重音 accent(t)（置信度门控）────────┘
```

- **能量主曲线（保证「连续、呼吸」）**：对数域 RMS + 滑窗分位 AGC（离线按 4s 窗，在线按 4s 滚动）+ 非对称一阶滤波（attack 取 Policy，release 取 `max(Policy.decay, 120ms)`，避免 off 档过碎）。
- **节拍网格（保证「踩点」）**：
  - 离线：沿用 Ellis DP，但 **ODF 换成低频段谱通量**（底鼓/贝斯定相位，踩镲不骗），并计算 **beat_confidence**（间隔规整度 + 拍点处 ODF 显著性 + 速度自相关峰值比）。
  - 在线：8s onset 环缓冲，每 0.5s 自相关估速（60–180 BPM，带滞回）+ PLL 预测下一拍；低置信自动停止重音。
- **重音加速度（音乐驱动，不再「每 4 拍」）**：每个拍点一个快起慢落的脉冲，幅度 = `0.55 + 0.45 * 该拍低频能量归一值`（离线可查表）；`tau` 由 safety 决定：off 140ms / comfort 110ms / strict 90ms。这样「重音跟着真实鼓的力度」，而不是固定第 1 拍。
- **置信度门控（解决平滑音乐）**：`beat_conf < 0.35` → 重音权重线性降到 0，只剩连续能量曲线 + 慢呼吸振荡；不会出现「折线段」。
- **混合权重**：`raw = (1-w)*energy + w*accent`，`w` 由 safety/policy 决定（off 0.42 / comfort 0.30 / strict 0.15）。
- **shaping（S4）**：`v = base + gain*raw` → `v**gamma` → `smoothstep` 混合（对比度），**无死区**、C1 连续；`off` 档用 `gamma 1.15 + contrast 0.45` 达到「更深黑、更亮峰」但不切段。
- **采样**：曲线存 172fps，`sample(t)` 用 `np.interp` 线性插值；循环点做 50ms 交叉淡化。

### 3.4 画布渲染（S3）

`CanvasScene`：
- 背景：6 段竖向渐变条（深色→palette），亮度随 `level` 平滑变化；中心径向光晕椭圆随 `energy/accent` 缩放。
- 频谱：24–28 根「胶囊柱」（`create_line` + round cap）三层：光晕（宽 2.4×，暗）、主体、底部倒影；EMA 平滑 + 邻域空间插值；峰值保持帽（缓降）。
- 节拍：2 个扩散环（`beat_phase` 驱动）+ 轻微全屏提亮（由 `accent` 驱动，不做频闪）。
- 性能：每帧只更新变化超过阈值的 item；颜色字符串缓存；目标 ≥50fps。

---

## 4. 症状 → 根因 → 方案 对照表（用户三条抱怨）

| 用户症状 | 直接根因 | v3 方案 |
|---|---|---|
| 「关闭护栏后很混乱、没踩鼓点」 | 硬黑场 0.16 + 85ms 脉冲 + 最近邻取样 + 每 4 拍固定重音 + off 档无任何平滑 | S4 平滑 S 曲线去死区；S2 低频 ODF 定拍 + 音乐驱动重音（tau 140ms）；插值采样；off 档仍保留 ≥120ms release |
| 「没鼓点的平滑音乐像粗糙分段线」 | 实时路径无 tempo 模型，punch 触发任意音符；硬黑场把低能量压成 0；最近邻取样 | S2 置信度门控（低→纯连续呼吸曲线）+ 4s AGC + 插值 + 无死区 shaping |
| 「画布效果太一般」 | 单色背景 + 28 根纯色平柱 + 无缓存 | S3 CanvasScene：渐变/光晕/胶囊柱/倒影/峰值帽/扩散环/平滑频谱 |

---

## 5. 增量迁移计划（每一步都可运行）

| 步骤 | 内容 | 验收 |
|---|---|---|
| S0 | 基线：selftest + 三模式 smoke（已完成，OK） | exit 0 / `OK` |
| S1 | `core/frame.py` + `core/registry.py` + `core/safe.py`；注册 classic 引擎；`app.py` 改为通过 registry 取 analyzer（行为不变） | selftest 绿、三 smoke 绿 |
| S2 | `ai/light_engine.py` 离线版；`features.py` 增加 `shape_level_smooth`；app 默认 `engine=auto`；`--engine` 参数；selftest 同时打印 classic / perceptual 指标 | 指标：拍点对齐↑、平滑度↑；三 smoke 绿 |
| S3 | 在线版（滚动 AGC + 在线 tempo/PLL + 置信度门控）；live 路径接入 | 实时模拟：闪烁率 ≤ 4/s，曲线 jerk 下降 |
| S4 | `render/scene_canvas.py` + `renderer.py` 调度改造 | 画布截图对比；fps ≥ 50 |
| S5 | `tools/diag_curve.py`（CSV + 指标）、`--diag` 参数；selftest 增加断言式指标 | 输出 `outputs/diag/*`，全部 PASS |
| S6 | 文档：README / HANDOFF / PROJECT_STATE / ARCHITECTURE_V3 同步 | 文档与代码一致 |

---

## 6. 兼容与风险控制

- **不删** `vendor/pyaudiowpatch`、`assets/demo_beat.wav`；**不硬编码** 任何 Key；LLM 仍只在配置时调用。
- `beat.py` 的 DP 参数**不动**；新引擎只在外面换 ODF 与加置信度，selftest 用同一份 `demo_beat.wav` 对比证明更好。
- 所有新模块 `try/except`，失败自动回退 classic，面板显示引擎名与回退原因；`tick()` 最外层异常捕获保留。
- 性能预算：离线分析增加 ≤ 100ms（60s 曲目目标 < 350ms）；实时每帧 ≤ 2ms（Tk 渲染另有 ~16ms 预算）。
- 默认 `engine=auto`：若新引擎异常，用户看到的仍是旧效果，不会崩。

---

## 7. 未来模块接入点（本次先留好缝）

| 模块 | 接入位置 | 契约 |
|---|---|---|
| KWS 语音唤醒 | `ai/kws.py` | 实现 `WakeDetector.detect(samples) -> bool`，注册 `wake:kws`，失败回退 `wake:energy` |
| YAMNet 场景 | `ai/scene_yamnet.py` | 实现 `SceneClassifier.classify(window) -> (label, conf)`，回退 `scene:heuristic` |
| 硬件亮度 | `platform/brightness.py` | 实现 `BrightnessBackend.set(level)`；render 后调用；失败回退遮罩 |
| 多人联动 | `core/events.py` | 订阅 `AnalysisFrame` → WebSocket 广播；不反向依赖 app |
| 非 WAV 解码 | `sources/decode.py` | 可选 ffmpeg/MediaFoundation；失败回退系统内录 |
| 分析缓存 | `analysis/cache.py` | `track.npz`（level/bands/beats/confidence），按文件哈希缓存 |
| 新渲染风格 | `render/scene_*.py` | 实现 `Scene.update(frame, policy)`，配置 `canvas_style` 选择 |

---

## 8. 本次不做（避免比赛前冒险）

- 不拆 `app.py` 成多文件（只把分析/渲染的接缝抽出来）。
- 不引入 librosa/madmom/torch 等重依赖（保持零安装）。
- 不上 KWS/YAMNet/硬件亮度/Android（架构留缝，10.24 后做）。
- 不改 `winsound` 播放方案（循环点交叉淡化在采样层解决）。

---

## 9. 实施状态（2026-10-20 已落地）

| 项 | 状态 | 位置 |
|---|---|---|
| S1 AnalysisFrame + registry + SafeEngine fallback | ✅ | `screenlight/core/{frame,registry,engine,dsp}.py` |
| S2 感知光引擎（离线 + 在线，时变速度 / 置信度门控 / 音乐驱动重音） | ✅ | `screenlight/ai/light_engine.py` |
| S3 CanvasScene（渐变/光晕/胶囊柱/倒影/峰值帽/节拍环） | ✅ | `screenlight/render/scene_canvas.py`，`renderer.py` 委托 + legacy 回退 |
| S4 平滑 S 曲线 + 子帧插值 + 循环交叉淡化 | ✅ | `features.shape_level_smooth`，`core/dsp.py`，`TrackCurve` |
| S5 视觉质量回归（selftest 指标 + diag 工具 + 真屏抓取） | ✅ | `selftest.py`，`tools/diag_curve.py`，`tools/visual_check.py` |
| S6 文档同步 | ✅ | `README.md` / `HANDOFF.md §13` / `AI_CONTEXT.md` / `PROJECT_STATE.json` |

关键结果：离线 demo 主脉冲 90 BPM（不是旧的 120.2）、onbeat 0.40→0.96、jerk 69→19；
平滑音乐 0 闪烁；画布 1280×720 208 fps；selftest + 三模式 smoke 全绿。
尚未做（按 §8 约定）：KWS / YAMNet / 硬件亮度 / Android / 多人联动 → 接入点见 §7。
