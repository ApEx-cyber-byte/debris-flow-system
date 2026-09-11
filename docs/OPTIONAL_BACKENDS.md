# 可选后端安装与验证

可选依赖的安装、验证与排错。**三者都不是运行必需**——不装任何一个，
系统都会自动回退到内置的纯 NumPy CPU 求解器，功能完整、结果物理一致。

---

## 0. 先看当前状态

```bash
curl http://127.0.0.1:8000/api/system/backends
```

或在界面上看右下角的状态栏芯片（`PySPH` / `SynxFlow` / `PyTorch`，
绿色为可用）。

探测逻辑在 `backend/app/physics/backends.py::probe()`：

| 判定 | 条件 |
|---|---|
| `pysph` 可用 | `importlib.util.find_spec("pysph")` 成功 |
| `synxflow` 可用 | 能导入 `synxflow` **且** `cupy.cuda.runtime.getDeviceCount() > 0` |
| `torch` 可用 | `find_spec("torch")` 成功 |

---

## 1. PySPH（可选加速 SPH）

- 项目：<https://github.com/sweeneyde/pysph>
- 许可证：**BSD-3-Clause**
- 用途：把本项目的非牛顿本构接入 PySPH 的编译型求解框架，
  通常比内置 NumPy SPH 快 **50–100×**

### 安装

PyPI 上**只有源码包**（`pysph-1.0b2.tar.gz`），没有 wheel，因此必须能编译
Cython + C++：

```bash
# Windows：需要 Visual Studio Build Tools（含 "使用 C++ 的桌面开发"）
#          + 与 Python 版本匹配的 Windows SDK
# Linux：  sudo apt install build-essential python3-dev cython3
pip install cython numpy
pip install pysph
```

### 验证

```bash
python -c "import pysph; print('pysph', pysph.__version__)"
```

或在界面里把「求解后端」显式选为 **PySPH**：若未安装，创建任务会返回
**HTTP 409** 并给出明确提示（而不是静默降级）。

### 已知情况

`backend/app/physics/pysph_backend.py` 已按 PySPH 的公开 API
（`Application` / `Equation` / `Group` / `Solver`）写好方程集：
`SummationDensity` → `ShearRateTensor`（自定义）→ `TaitEquationOfState`
→ `NonNewtonianViscosity`（自定义 Morris 粘性，逐粒子 `μ_app`）
→ `BasalFrictionForce`（自定义 Voellmy-Coulomb + Manning）→ `MomentumEquation`。

> ⚠️ **该路径在开发机上未经运行时验证**（无 C++ 工具链）。
> 若运行时失败，`TaskManager` 会捕获异常、自动用 `shallow_water` 重启，
> 并把回退原因写入 `backend.notes` 与任务消息——不会导致任务失败。

---

## 2. SynxFlow（可选 GPU 求解）

- 项目：<https://github.com/SynxFlow/SynxFlow>
- 许可证：**MIT**
- 用途：GPU 深度积分浅水方程求解

### 前置条件

1. **NVIDIA GPU** + 驱动
2. **CUDA Toolkit**（版本需与 cupy 匹配）
3. **cupy**：`pip install cupy-cuda12x`（按 CUDA 版本选择）

### 安装

```bash
pip install cupy-cuda12x
pip install synxflow
```

### 验证

```bash
python -c "import cupy; print('devices:', cupy.cuda.runtime.getDeviceCount())"
```

设备数为 0 时后端会被判为不可用，系统按规格要求回退到
**CPU 浅水方程求解器**（`backend/app/physics/shallow_water.py`）。

> ⚠️ SynxFlow 的 Python 求解接口在不同版本间差异较大。
> `synxflow_backend.py` 把 GPU 绑定集中在
> `SynxFlowSolver._gpu_advance()` **一个方法**里作为唯一集成点，
> 并做了防御性降级。针对具体 SynxFlow 版本适配时只需替换该方法体，
> 系统其余部分不变。

---

## 3. PyTorch（AI 代理模型）

- 许可证：**BSD-3-Clause**

```bash
# CPU 版（够用，训练小规模代理模型）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 版
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

装好后 `/api/ai/models` 会把 `cnn-lstm-v1` / `dr-lstm-v1` 列为可选，
但**仍需训练权重**才会真正启用（见下）。

### 训练流程

```bash
cd backend

# 1) 生成样本（拉丁超立方采样 + 批量数值模拟，300 个样本约 20–40 分钟）
python scripts/generate_dataset.py --samples 300 --out data/dataset.npz

#    冒烟测试：
python scripts/generate_dataset.py --samples 4 --quick --out data/smoke.npz

# 2) 训练
python scripts/train_surrogate.py --data data/dataset.npz --model cnn-lstm-v1

# 3) 权重写入 backend/weights/cnn_lstm_v1.pt，重启后端即自动启用
```

`backend/app/ai/predictor.py` 通过**权重文件是否存在**决定用哪个模型，
前端 `AiPanel.vue` 会明确显示实际作答模型与「已训练 / 物理代理」标签。

---

## 4. 回退语义（重要）

系统有两级回退，保证永不因缺少可选依赖而不可用：

1. **创建任务时**
   - `backend="pysph"` 但 PySPH 不可用 → **HTTP 409**，提示安装方法
   - `backend="synxflow"` 但无 CUDA → **HTTP 409**，提示改用 `shallow_water`
   - `backend="auto"` → 静默选择可用者：SynxFlow → PySPH → `shallow_water`

2. **运行过程中**
   - 加速后端抛异常 → `TaskManager._execute()` 捕获，
     用 `params.model_copy(update={"backend": "shallow_water"})` 重启，
     把原因追加到 `backend.notes`

因此「后端失败」永远表现为**降级 + 明确提示**，而不是任务失败。

---

## 5. 性能参考（本机实测，默认参数，Python 3.13 / NumPy 2.5）

| 后端 | 12 s 工况墙钟 | 相对实时 |
|---|---|---|
| `shallow_water`（221×140 栅格） | ≈ 11.5 s | ≈ 1.0× |
| `numpy_sph`（N ≈ 2166） | ≈ 12 min | ≈ 0.018× |

SPH 慢是 WCSPH 的固有刚度：声速 `c₀ ≈ 140 m/s` 使 `dt ≈ 9e-4 s`。
要在粒子级精度下接近实时，PySPH（编译版）是正道。
