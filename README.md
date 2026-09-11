# 基于流体动力学的泥石流坡度滚落物理动态分析系统

**Debris-Flow Slope Runout Physical Dynamics Analysis System** — 第一版 (v1.0)

在浏览器端完成泥石流等非牛顿流体**从斜坡滚落**的参数调节、物理模拟、三维可视化
与数据导出，并预留 AI 快速预测接口。

---

## 目录

- [1. 系统能力概览](#1-系统能力概览)
- [2. 快速开始](#2-快速开始)
- [3. 项目结构](#3-项目结构)
- [4. 物理内核](#4-物理内核)
- [5. 求解后端与回退策略](#5-求解后端与回退策略)
- [6. 前后端通信协议](#6-前后端通信协议)
- [7. REST / WebSocket API](#7-rest--websocket-api)
- [8. 数据导出](#8-数据导出)
- [9. AI 预测接口](#9-ai-预测接口)
- [10. 性能与已知限制](#10-性能与已知限制)
- [11. 测试](#11-测试)
- [12. 许可证](#12-许可证)

---

## 1. 系统能力概览

| 规格要求 | 实现情况 |
|---|---|
| 坡度角 0–60° 可调 | ✅ `slope_angle_deg`，滑块 + 输入框，硬约束校验 |
| 坡长、Manning 粗糙度、体积浓度、屈服应力、粘性系数、密度、模拟时长、粒子数量可调 | ✅ 全部在参数面板暴露 |
| 非牛顿本构（优先 Bingham 或 Cross） | ✅ Bingham（Papanastasiou 正则化）、Cross、Herschel-Bulkley、幂律，四种可切换 |
| 坡面启动 → 加速 → 滚落 → 扩散 → 堆积 | ✅ 解析软坡脚地形 + 屈服应力停止判据，完整复现全过程 |
| Three.js 实时渲染 SPH 粒子 | ✅ `Points` + 自定义着色器，按流速着色 |
| 流深热力图 | ✅ 深度栅格 → `DataTexture` → 贴地着色器 |
| 流速矢量场 | ✅ `InstancedMesh` 箭头场，按格点降采样 |
| 旋转 / 缩放 / 平移 | ✅ `OrbitControls`，另有复位 / 侧视 / 俯视预设 |
| ECharts 流速时间曲线 | ✅ |
| ECharts 流深距离剖面 | ✅ 沿程中线剖面 |
| ECharts 冲击力时间曲线 | ✅ 动压 + 静压积分 |
| 导出 CSV / JSON | ✅ 时间序列 / 粒子快照 / 流深剖面 / 完整记录 |
| 场景截图 | ✅ 视口 PNG 下载 |
| 前端 REST 提交参数 | ✅ `POST /api/simulations` |
| 后端创建任务并调度 PySPH / SynxFlow | ✅ `TaskManager` + 适配器 + 自动回退 |
| WebSocket 推送关键帧 | ✅ 二进制帧，默认**每 10 个时间步**一帧 |
| 前端降采样 | ✅ 后端 `particle_stride` + 前端二次降采样 |
| AI 预测接口预留（PyTorch / CNN-LSTM / DR-LSTM） | ✅ 接口就绪，物理代理开箱可用，网络定义与训练脚本已提供 |
| 保留开源许可证声明 | ✅ 见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| 无 GPU 时用 CPU 浅水方程求解器替代 SynxFlow | ✅ 且作为**默认**后端 |

---

## 2. 快速开始

### 环境要求

| | 版本 | 说明 |
|---|---|---|
| Node.js | ≥ 20（实测 24.18） | 前端构建 |
| pnpm | ≥ 9（实测 11.7） | 也可用 npm，见下文 |
| Python | 3.11 – 3.13（**推荐 3.13**） | 后端 |

> **为什么推荐 Python 3.13 而不是 3.14？**
> 本系统只依赖 NumPy + FastAPI，两个版本都能跑；但 3.13 的科学计算与
> 可选后端（PyTorch / PySPH 源码编译）wheel 覆盖更完整。

### 一键启动（Windows PowerShell）

```powershell
cd debris-flow-system
.\scripts\start.ps1
```

### 一键启动（Linux / macOS）

```bash
cd debris-flow-system
chmod +x scripts/start.sh
./scripts/start.sh
```

启动脚本会：创建 Python 虚拟环境 → 安装后端依赖 → 安装前端依赖 →
同时拉起 FastAPI (`:8000`) 与 Vite (`:5173`)。

### 手动启动

**后端**

```bash
cd backend
py -3.13 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**前端**

```bash
cd frontend
pnpm install
pnpm dev
```

打开 <http://127.0.0.1:5173> 。Vite 已把 `/api` 与 `/ws` 代理到
`http://127.0.0.1:8000`，浏览器始终处于单一 origin。后端交互式 API 文档：
<http://127.0.0.1:8000/docs> 。

> 用 npm 代替 pnpm：`npm install && npm run dev`。若 npm 因
> `allowBuilds` 拒绝执行依赖脚本，请确认 esbuild 的 postinstall 已运行
> （见 `frontend/pnpm-workspace.yaml` 注释）。

### 第一次使用建议

1. 保持默认参数（30° 坡、20 m 坡长、150 m³ 泥石流），点「**开始仿真**」。
2. 默认后端是 **CPU 深度积分浅水方程**，约 **1× 实时**，12 s 工况十几秒跑完。
3. 观察三条曲线与三维粒子；用鼠标左键旋转、滚轮缩放、右键平移。
4. 切换到「**AI 预测**」页签，点「快速预测」对比毫秒级预测与数值结果的量级。
5. 在「④ 数据导出」里导出 CSV / JSON。

### 界面截图

`docs/screenshots/` 中的三张图由 `backend/scripts/ui_screenshot.py`
（无头 Chrome + CDP 驱动，真实点击「开始仿真」）自动生成：

| 文件 | 内容 |
|---|---|
| `01_initial.png` | 初始状态：参数面板、三维坡面地形、空曲线 |
| `02_running.png` | 运行中：坡面泥石流粒子（按流速着色）+ 三条实时曲线 |
| `03_ai.png` | AI 预测面板：跑距 / 峰值流速 / 到达时间 / 冲击力 |

重新生成：

```bash
cd backend
python scripts/ui_screenshot.py --url http://127.0.0.1:5173
```

### 常见问题：脚本报「意外的标记」/ 中文乱码

**现象**（Windows PowerShell 5.1）：

```
start.ps1:42 字符: 13
+ Write-Host "娉ョ煶娴佸潯搴︽粴钀界墿鐞嗗姩鎬佸垎鏋愮郴缁?鈥?鍚姩涓? -ForegroundColor Green
表达式或语句中包含意外的标记"娉ョ煶娴佸潯..."
```

**原因**：`start.ps1` 含中文，而 Windows PowerShell 5.1 对**没有 BOM** 的文件
按 ANSI 代码页（简中环境为 GBK）解码。UTF-8 的汉字字节被当成 GBK 解读后
产生乱码，乱码字节会打断引号配对，于是报「意外的标记」，并级联出
「缺少右 }」等一堆看似无关的错误。

**修复**：`start.ps1` 必须以 **UTF-8 with BOM** 保存（本仓库已如此，
文件开头注释也写明了）。若编辑器改写了该文件，重新另存为
「UTF-8 with BOM」即可；也可用：

```powershell
$p = '.\scripts\start.ps1'
$text = [System.IO.File]::ReadAllText($p, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($p, $text, [System.Text.UTF8Encoding]::new($true))
```

> `backend/app/console.py` 解决的是**输出端**的同类问题：Python 脚本在传统
> 代码页控制台下 `print` 中文会乱码。所有命令行脚本启动时都会调用
> `use_utf8_console()`，无需再手动设置 `PYTHONIOENCODING`。

### 常见问题：`%1 is not a valid Win32 application`

**现象**：

```
Start-Process : 由于出现以下错误，无法运行此命令: %1 is not a valid Win32 application。
+     $frontendJob = Start-Process -FilePath 'pnpm' `
```

**原因**：Windows 上 npm 的全局 bin 目录里**同时存在三个同名文件**：

```
%APPDATA%\npm\pnpm        403 字节，Unix bash 脚本（无扩展名）
%APPDATA%\npm\pnpm.cmd    Windows 批处理
%APPDATA%\npm\pnpm.ps1    PowerShell 脚本
```

`Start-Process -FilePath 'pnpm'` 把名字直接交给 `CreateProcess`，它会按
**精确文件名**匹配，命中的是那个无扩展名的 bash 脚本 —— 而 Windows 根本
无法执行它，于是报 `%1 is not a valid Win32 application`（“%1 不是有效的
Win32 应用程序”）。

> 这个故障与 PATH 中目录的先后顺序有关，因此在某些终端里能跑通、在另一些
> 终端里报错。凡是依赖 `Start-Process <裸命令名>` 的写法都有这个隐患。

**修复**（`start.ps1` 已实现）：不再传裸命令名，而是
1. 用 `Resolve-NodeLauncher` 显式解析出 `.cmd` / `.exe` 的**完整路径**
   （自动跳过无扩展名的 Unix shim），找不到 pnpm 时回退到 npm；
2. 通过 `cmd.exe /c <完整路径>` 启动批处理 —— `CreateProcess` 本身不能
   直接执行 `.cmd`；
3. 退出时用 `taskkill /F /T` 结束**整棵进程树**，避免 `cmd.exe → node.exe`
   里的子进程变成占着端口的孤儿。

**前端「拒绝连接」**：那是上一条的连带结果 —— Vite 从未启动，所以 5173
没有监听。修好启动器后会同时看到：

```
    后端已就绪
    前端已就绪
就绪：
  前端      http://127.0.0.1:5173
  （请用上面的前端地址访问界面；直接打开 :8000 只会看到后端 JSON）
  后端 API  http://127.0.0.1:8000
```

脚本现在还会在启动前检查端口占用，并在 30 秒内确认两个端口真的起来了，
而不是打印一行“就绪”就假设成功。

---

## 3. 项目结构

```
debris-flow-system/
├── README.md                      本文件
├── THIRD_PARTY_NOTICES.md         第三方许可证声明
├── LICENSE                        本项目许可证 (MIT)
├── .gitignore
│
├── backend/                       Python FastAPI 后端
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── main.py                FastAPI 入口、CORS、生命周期
│   │   ├── config.py              全局设置与物理常数
│   │   ├── schemas.py             ★ API 契约（前端 TS 镜像见 frontend/src/types）
│   │   ├── api/
│   │   │   ├── routes_sim.py      任务生命周期、后端探测、本构探针
│   │   │   ├── routes_export.py   CSV / JSON 导出
│   │   │   ├── routes_ai.py       AI 代理模型接口
│   │   │   └── ws.py              WebSocket 关键帧推送
│   │   ├── physics/
│   │   │   ├── rheology.py        ★ 非牛顿本构（Bingham/Cross/HB/幂律）+ 基底阻力
│   │   │   ├── terrain.py         软坡脚解析地形、栅格、释放区
│   │   │   ├── sph_numpy.py       ★ 内置向量化 SPH 求解器（保底，含完整本构耦合）
│   │   │   ├── shallow_water.py   ★ CPU 深度积分浅水方程（SynxFlow 的 CPU 替代）
│   │   │   ├── pysph_backend.py   可选 PySPH 适配器
│   │   │   ├── synxflow_backend.py 可选 SynxFlow GPU 适配器
│   │   │   ├── backends.py        后端探测与工厂（自动回退）
│   │   │   └── engine.py          统一仿真引擎、关键帧抽取、诊断量
│   │   ├── services/
│   │   │   ├── task_manager.py    任务注册表、线程调度、帧广播
│   │   │   └── frames.py          WebSocket 二进制帧编解码
│   │   └── ai/
│   │       ├── physics_surrogate.py 物理启发式代理（开箱可用）
│   │       ├── models.py            CNN-LSTM / DR-LSTM 网络定义
│   │       └── predictor.py         模型选择门面
│   ├── scripts/
│   │   ├── generate_dataset.py    拉丁超立方采样 + 批量数值模拟生成训练集
│   │   └── train_surrogate.py     PyTorch 训练脚本
│   ├── tests/test_physics.py      56 项物理与协议测试
│   └── runs/                      运行产物（导出缓存）
│
├── frontend/                      Vue 3 + TypeScript 前端
│   ├── package.json / vite.config.ts / tsconfig.json
│   └── src/
│       ├── main.ts / App.vue      应用外壳与三栏布局
│       ├── types/simulation.ts    ★ API 契约的 TS 镜像
│       ├── api/client.ts          REST 客户端
│       ├── api/ws.ts              WebSocket 二进制帧解码 + 二次降采样
│       ├── stores/simulation.ts   Pinia：参数、任务、帧流（帧旁路响应式）
│       ├── three/
│       │   ├── SceneManager.ts    ★ 地形 / 粒子 / 热力图 / 矢量场 / 截图
│       │   └── terrain.ts         地形解析式的 TS 孪生实现
│       ├── charts/options.ts      三张 ECharts 图的 option 构造器
│       └── components/
│           ├── ParameterPanel.vue Element Plus 参数面板 + 导出
│           ├── Viewport.vue       Three.js 视口与图层开关
│           ├── ChartsPanel.vue    三张曲线图 + 关键指标
│           └── AiPanel.vue        AI 预测与模型状态
│
├── scripts/
│   ├── start.ps1                  Windows 一键启动
│   └── start.sh                   Linux/macOS 一键启动
│
└── docs/
    └── OPTIONAL_BACKENDS.md       PySPH / SynxFlow / PyTorch 安装与验证
```

★ = 核心文件

---

## 4. 物理内核

### 4.1 地形与坐标系

床面用**软坡脚解析式**（softplus 铰链），使坡面到水平堆积区连续可导，
既避免转折处的数值速度脉冲，又保留解析梯度便于施加基底摩擦：

```
z_b(x) = R · tanθ · softplus((x_toe - x) / R)
```

- `x` 沿坡向下为正，坡顶在 `x = 0`，`z` 竖直向上
- `x_toe = L·cosθ`，坡顶高度 `H = L·sinθ`
- `R = max(0.5, 0.25·x_toe)` 为坡脚转折半径
- `x ≪ x_toe` 时退化为严格线性坡面；`x ≫ x_toe` 时趋于水平

### 4.2 非牛顿本构（`physics/rheology.py`）

四种本构统一以**表观粘度** `μ_app(γ̇)` 暴露，供两个求解器共用：

| 模型 | 公式 | 说明 |
|---|---|---|
| **Bingham** | `τ = τ_y(1 − e^(−mγ̇)) + μ_p γ̇` | Papanastasiou 指数正则化，`m = 1000 s`，消除 `τ_y/γ̇` 在 `γ̇→0` 的奇异 |
| **Cross** | `μ_app = μ_∞ + (μ_0 − μ_∞)/(1 + (λγ̇)^n)` | 四参数模型，覆盖全剪切速率区间 |
| **Herschel-Bulkley** | `μ_app = τ_y/γ̇ + Kγ̇^(n−1)` | 屈服 + 幂律，n<1 剪切稀化 |
| **Power-Law** | `μ_app = Kγ̇^(n−1)` | 无屈服应力 |

剪切速率下限 `γ̇_min = 1e-3 s⁻¹`，表观粘度上限 `μ_max` 由
`max_apparent_viscosity_pa_s` 控制（默认 20000 Pa·s），保证时间积分稳定。

### 4.3 基底阻力

屈服应力之外，床面阻力按 **Voellmy–Coulomb + Manning** 组合施加：

```
τ_b = τ_y + μ_bed·ρ·g·h·cosβ + ρ·g·n²·|v_t|² / h^(1/3)
a_b = τ_b / (ρ·h)
```

第一项是 Bingham 塞流最终**停止**的物理来源，第二项为库仑摩擦，第三项为
Manning 湍流阻力。`cosβ` 取局地床面法向竖直分量，因此在坡脚过渡段自动连续变化。

### 4.4 求解器 A：内置 NumPy SPH（`physics/sph_numpy.py`）

- **核函数**：3D 三次样条核，支撑域 `2h`，`h = 1.3·dx`
- **密度**：`ρ_i = C · m Σ_j W_ij`，`C` 由**初始点阵**的 95 分位核和标定
  - 标定的必要性：三次样条核在 `q=1` 处二阶导不连续，点阵求积存在
    O(10%) 偏差，不标定会被 Tait 方程放大成虚假超压
  - 采用**单一全局比例**（而非逐粒子修正）的额外好处：自由面粒子核和偏小
    → `ρ < ρ₀` → `p = max(p,0)` 自动给出大气压，无需显式自由面检测；
    逐粒子修正则必须截断，而截断后的粒子一旦被流动埋入内部就会产生压力尖峰
- **δ-SPH 密度扩散**（`δ = 0.10`）：核加权密度平均，抑制点阵噪声
- **状态方程**：Tait，`p = B[(ρ/ρ₀)^7 − 1]`，`B = ρ₀c₀²/7`，`c₀ = 10·√(2gH)`（马赫数 ≈0.1）
- **压力梯度**：对称形式 `−Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇_i W_ij`
- **粘性**：Morris 等 (1997) 拉普拉斯算子，粘度取本构的 `μ_app(γ̇)`，
  `γ̇` 由 SPH 速度梯度张量 `L_ab = −Σ_j V_j (v_j−v_i)_a ∂W/∂x_b` 的对称部分算出；
  对角项**半隐式**积分，允许远大于显式粘性稳定限的时间步
- **人工粘性**：Monaghan，`α = 0.15`，抑制激波与粒子穿透
- **邻域搜索**：均匀网格链表 + 稳定排序 + 27 邻域，全向量化；
  粗筛距离在生成 `(M,3)` 分离矢量**之前**完成（网格模板约 6× 过取，
  提前过滤可省下大量内存带宽）
- **床面接触**：非弹性法向投影 + 床面层内施加基底阻力，含「屈服停止」判据
- **稳定时间步**：`dt = min(CFL·h/(c₀+v_max), 0.25√(h/|a|_max))`

### 4.5 求解器 B：CPU 深度积分浅水方程（`physics/shallow_water.py`）

规格要求「SynxFlow 需要 CUDA GPU，没有 GPU 时先用 CPU 浅水方程求解器替代」，
本求解器即该替代实现，并且是**默认后端**。

守恒形式：

```
∂h/∂t   + ∂(hu)/∂x + ∂(hv)/∂y = 0
∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = −gh·∂z_b/∂x − τ_bx/ρ
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = −gh·∂z_b/∂y − τ_by/ρ
```

非牛顿本构以两条途径进入：

1. **基底阻力** `τ_b`：同 §4.3。采用半隐式离散 + 显式**屈服停止判据**
   （`a_res·dt ≥ |V|` 直接置零），既不受刚性限制，又能真实堆积。
2. **深度平均粘性扩散**：`∂(hu)/∂t += ∇·(ν_app h ∇u)`，
   其中 `ν_app = μ_app(γ̇)/ρ`，剪切速率取浅水流动的主导尺度 **`γ̇ ≈ |V|/h`**
   （垂向剪切）——与 SPH 共用同一个本构模块。

数值方法：守恒变量有限体积 + **Rusanov 近似黎曼通量**（对干湿前锋稳健）+
与压力通量离散一致的床面坡度源项（满足静水平衡）+ 自适应时间步。

**拉格朗日示踪粒子**：浅水方程本身是欧拉的，为满足「Three.js 渲染 SPH 粒子」，
按初始释放体布置 `particle_count` 个示踪粒子，用双线性插值的速度场推进；
局部泥深低于干阈值即冻结（沉积）。这样两种后端都能给出粒子级可视化，
且物理上是一致的（示踪粒子是被动标量，不反馈到流场）。

---

## 5. 求解后端与回退策略

| 后端 | 依赖 | 相对实时速度（默认参数） | 说明 |
|---|---|---|---|
| `shallow_water` | 仅 NumPy | **≈ 1×** | **默认**。SynxFlow 的 CPU 替代 |
| `numpy_sph` | 仅 NumPy | ≈ 0.02× | 完整粒子级非牛顿本构，精度高但慢 |
| `pysph` | PySPH + C++ 工具链 | 未实测 | 可选加速 SPH |
| `synxflow` | SynxFlow + CUDA | 未实测 | 可选 GPU 求解 |
| `auto` | — | — | 有 CUDA→SynxFlow；否则有 PySPH→PySPH；否则→`shallow_water` |

> **实测数据（本机，默认参数，Python 3.13 / NumPy 2.5）**
> - 浅水方程：12 s 工况约 1900 步，墙钟 ≈ 11.5 s（≈1.0× 实时）
> - NumPy SPH：N≈2166 粒子，约 0.053 s/步，dt≈9.2e-4 s → ≈0.018× 实时
> - 交叉验证：t = 1.0 s 时 SPH 峰值流速 6.31 m/s，浅水方程 7.17 m/s（差 12%）

**两级回退保证系统永不因缺少可选依赖而不可用：**

1. **创建时**：显式请求 `pysph` / `synxflow` 但不可用 → HTTP 409 + 明确提示；
   `auto` 则静默选可用者。
2. **运行时**：加速后端在运行中抛异常 → `TaskManager` 捕获并用
   `shallow_water` 重启，把回退原因写进 `backend.notes` 与任务消息。

安装与验证可选后端见 [`docs/OPTIONAL_BACKENDS.md`](docs/OPTIONAL_BACKENDS.md)。

---

## 6. 前后端通信协议

### 6.1 二进制关键帧

SPH 数据量大：以 JSON 数组传 4000 个粒子的位置 + 速度约 240 KB/帧，
同样数据用 `float32` 二进制只要 96 KB，且浏览器端解码是一次
`Float32Array` 视图，没有 `JSON.parse` 开销。

```
+----------------+--------------------------+---------------------------+
| uint32 header  |  UTF-8 JSON header       |  concatenated float32     |
| length (LE)    |  (已填充到 4 字节对齐)     |  arrays, in header order  |
+----------------+--------------------------+---------------------------+
```

> JSON 头会**填充空格到 4 字节对齐**。否则客户端
> `new Float32Array(buffer, headerEnd, count)` 会因起始偏移不是 4 的倍数
> 抛 `RangeError`。尾随空格是合法 JSON 空白。

一帧包含的数组：`pos (3N)`、`vel (3N)`、`depth (ny·nx)`、`vx`、`vy`。

### 6.2 帧率与降采样

- 后端**每 `frame_stride` 个时间步产出一帧**（规格默认 **10**）
- 后端按 `max_frame_particles`（默认 6000）做第一级 `particle_stride` 降采样
- 前端 `downsample()` 做第二级降采样
- 前端把帧**合并到每个 animation frame 一次**再渲染（求解器可能快于显示）

### 6.3 帧流的响应式旁路

帧以数十赫兹到达。若让帧走 Vue 响应式，每帧都会触发整棵树重渲染。
因此：

- 解码后的帧**直接走回调**（`store.onFrame`）进入 Three.js 缓冲区；
  几何体仅在粒子数变化时重新分配
- 只有**汇总值**（计数器、最新统计、时间序列）是响应式的
- 三张 ECharts 图按 **4 Hz 节流**重绘

---

## 7. REST / WebSocket API

完整交互式文档：<http://127.0.0.1:8000/docs>

### 系统

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/api/system/backends` | 可选依赖探测结果与后端标签 |
| GET | `/api/system/defaults` | 默认参数 |
| POST | `/api/system/rheology` | 本构探针曲线（表观粘度 vs 剪切速率） |

### 仿真任务

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/simulations` | **创建并启动**；返回 `run_id`、地形、栅格、后端 |
| GET | `/api/simulations` | 任务列表 |
| GET | `/api/simulations/{id}` | 任务状态与进度 |
| POST | `/api/simulations/{id}/pause` \| `/resume` \| `/stop` | 运行控制 |
| DELETE | `/api/simulations/{id}` | 停止并删除 |
| GET | `/api/simulations/{id}/result` | 完整结果（参数 + 时间序列 + 指标） |
| GET | `/api/simulations/{id}/depth-profile` | 沿程流深剖面 |

### 流

| 方法 | 路径 | 说明 |
|---|---|---|
| WS | `/ws/simulations/{run_id}` | 关键帧推送 |

服务端 → 客户端：二进制 `meta`（参数/地形/栅格/后端/床面高程）、
二进制 `frame`、文本 `completed` / `error`。
客户端 → 服务端：`{"type":"pause"|"resume"|"stop"|"ping"}`（文本 JSON）。

> 订阅队列**有界**（容量 8）。客户端消费不过来时丢弃最旧的帧，而不是拖慢物理求解。

### 示例

```bash
curl -X POST http://127.0.0.1:8000/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"slope_angle_deg":35,"slope_length_m":25,"rheology":"bingham",
       "yield_stress_pa":600,"viscosity_pa_s":80,"manning_n":0.035,
       "density_kg_m3":2200,"volume_concentration":0.45,
       "duration_s":10,"particle_count":5000}'
```

---

## 8. 数据导出

`GET /api/simulations/{id}/export?kind=<kind>&fmt=<fmt>[&frame_index=N]`

| `kind` | `fmt` | 内容 |
|---|---|---|
| `time_series` | csv / json | `t, max_speed, mean_speed, front_x, max_depth, kinetic_energy, impact_force_kn, deposited_volume` |
| `particles` | csv / json | 指定关键帧的 `x,y,z,vx,vy,vz,speed`（默认最后一帧） |
| `depth_profile` | csv / json | 沿程中线流深剖面 |
| `full` | json | 参数 + 后端 + 地形 + 栅格 + 指标 + 全部时间序列 + 剖面 |

- CSV 带 **UTF-8 BOM**（`utf-8-sig`），Excel 直接双击不乱码
- 中文文件名用 **RFC 5987** 编码（`filename*=UTF-8''…`），浏览器正确还原
- 前端通过 `Blob` + `URL.createObjectURL` 下载，不离开 SPA

**场景截图**：`SceneManager.captureScreenshot()` 用
`preserveDrawingBuffer: true` 的 WebGL 上下文调用 `toDataURL('image/png')`，
直接下载 PNG。

---

## 9. AI 预测接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/ai/models` | 可用代理模型列表 |
| POST | `/api/ai/predict` | 快速预测 |
| POST | `/api/ai/features` | 调试：参数 → 特征向量 |

返回：跑距、峰值流速、到达监测断面时间、最大泥深、峰值冲击力、影响范围面积、置信度。

### 三个模型

| 名称 | 类型 | 状态 |
|---|---|---|
| `heuristic-v1` | 物理启发式 | **开箱可用** |
| `cnn-lstm-v1` | PyTorch CNN-LSTM | 接口就绪，需训练权重 |
| `dr-lstm-v1` | PyTorch 双分辨率 LSTM | 接口就绪，需训练权重 |

`heuristic-v1` **不是**黑箱回归：它把质量中心当作沿程一维质点，用与完整求解器
**完全相同的非牛顿本构与基底阻力公式**积分动量方程，坡面积分到坡脚得 `v_toe`，
再在水平堆积区减速到零得跑距。毫秒级返回，量级正确，适合滑块拖动时的实时预判。

### 启用学习型代理

```bash
cd backend
python scripts/generate_dataset.py --samples 400 --out data/dataset.npz
python scripts/train_surrogate.py --data data/dataset.npz --model cnn-lstm-v1
# 权重写入 backend/weights/cnn_lstm_v1.pt，重启后端即自动启用
```

`AIPredictor` 会检测权重文件是否存在来决定用哪个模型，前端
`AiPanel.vue` 会明确显示**实际作答的模型**与「已训练 / 物理代理」标签——
这个区别永远不会被隐藏。

---

## 10. 性能与已知限制

### 已实测

| 项 | 数值 |
|---|---|
| 浅水方程（默认，12 s 工况，221×140 栅格） | ≈ 10–12 s 墙钟（≈1.0× 实时），约 1900 步 |
| 质量守恒（浅水方程） | 相对误差 < 1e-6 |
| NumPy SPH（N≈2166） | ≈ 0.053 s/步，≈0.018× 实时 |
| SPH vs 浅水方程峰值流速（t=1 s） | 6.31 vs 7.17 m/s（差 12%） |
| SPH 稳定性 | 无 NaN，密度恒在 `[0.5ρ₀, 3ρ₀]` |
| 单帧二进制大小（4000 粒子 + 221×140 栅格） | ≈ 400 KB |
| 端到端（`scripts/smoke_e2e.py`） | 全部通过：74 帧、5 种导出、AI、参数校验 |
| 前端（无头 Chrome） | WebGL 2.0 可用，3 张 ECharts 画布，104 帧实时渲染 |

**默认参数下的参考结果**（30°、20 m 坡、150 m³、12 s）：
峰值流速 **8.5 m/s**、跑距（坡脚下游）**18.6 m**、峰值冲击力 **≈680 kN**、
约 7 s 后完全停止堆积。

### AI 代理模型精度（诚实数字）

用 `backend/scripts/validate_surrogate.py` 对 8 组工况做「代理 vs 完整模拟」对比：

| 指标 | 平均绝对误差 | 平均偏差 |
|---|---|---|
| 跑距 | **15.1%** | −0.4%（基本无偏） |
| 峰值流速 | 21.8% | +21.8%（系统性偏高） |

单个工况单次预测耗时 **≈1 ms**（对比完整模拟 6–19 s，快约 4 个数量级）。
系数由 `scripts/calibrate_surrogate.py` 对 8 组参考解做网格搜索标定，
两个系数在 `physics_surrogate.py` 顶部以注释公开，不是隐藏的调参。
峰值流速系统性偏高，是因为一维质点模型不解析二维横向展宽，
坡面段的 Manning 阻力被低估——这一点在启用学习型代理模型（CNN-LSTM）后
可望改善。

### 已知限制（诚实清单）

1. **NumPy SPH 慢于实时**（≈0.018×）。这是 WCSPH 的固有刚度：声速
   `c₀ ≈ 140 m/s` 使 `dt ≈ 9e-4 s`。要实时须安装 PySPH（Cython/C++ 编译版，
   通常快 50–100×）或改用默认的浅水方程后端。UI 会在选择 SPH 时给出
   预估墙钟时间。
2. **SPH 的 `h/dx = 1.3` 偏小**，点阵求积偏差约 10%，靠密度标定补偿。
   提高到 `h = 1.7dx` 可把偏差压到 2% 以内，但邻域数从 73 涨到 164，
   代价约 2 倍。第一版选择速度。
3. **浅水方程是深度积分近似**，在 45°–60° 陡坡上静水压假设偏弱；
   SPH 后端在该区间更可信。
4. **无侵蚀 / 卷吸模块**。真实泥石流会沿程冲刷床面增大体积；第一版体积守恒。
5. **地形是单一平面坡 + 水平堆积区**，不含沟道弯曲、障碍物或真实 DEM。
6. **示踪粒子是被动标量**，不反馈到浅水流场（这是正确的做法，但意味着
   浅水后端的「粒子」不是真正的 SPH 粒子）。
7. **前端渲染上限**：默认最多 6000 个粒子/帧；更大规模需调
   `maxFrameParticles`，但低端 GPU 可能掉帧。
8. **单机、单进程**。任务注册表在内存中，进程重启后丢失。
9. **PySPH / SynxFlow 适配器未经运行时验证**（本机无 C++ 工具链 / 无 CUDA）。
   两者都做了防御性降级，失败会被捕获并回退，不影响系统可用性。

---

## 11. 测试

```bash
cd backend
.venv\Scripts\python -m pytest tests -q
```

覆盖 32 项：本构模型的量纲与单调性、地形解析梯度与几何、
两个求解器的质量守恒与稳定性、SPH 密度-压力响应回归、
二进制帧编解码与 4 字节对齐、参数边界校验、AI 代理的单调性。

端到端冒烟测试（起一个真实 uvicorn，走完 REST → WebSocket → 导出 → AI 全链路）：

```bash
cd backend
python scripts/smoke_e2e.py
```

代理模型精度校验：

```bash
python scripts/validate_surrogate.py      # 8 组工况，约 90 s
python scripts/calibrate_surrogate.py     # 重新标定两个结构系数
```

前端类型检查与构建：

```bash
cd frontend
pnpm run typecheck     # vue-tsc
pnpm run build         # 类型检查 + 生产构建
```

---

## 12. 许可证

本项目自有代码以 **MIT** 许可发布（见 [`LICENSE`](LICENSE)）。
所依赖与参考的第三方项目及其许可证见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

其中需特别注意：**PySPH (BSD-3-Clause)** 与 **SynxFlow (MIT)** 是可选后端，
本项目未复制其源码；**TITAN2D (GPL-3.0)** 仅参考公式，未复制代码，
因此本项目不受 GPL 传染。
