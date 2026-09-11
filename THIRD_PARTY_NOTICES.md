# 第三方开源项目与许可证声明

Third-Party Open-Source Notices
===============================

本项目的**自有代码**（`backend/`、`frontend/` 下的全部源码）为第一版原创实现。
下表列出本项目在设计、公式参考与依赖层面涉及的第三方开源项目及其许可证。

> ⚠️ **重要说明**：本项目**没有复制**下列项目的源代码。PySPH 与 SynxFlow 是
> **可选**依赖（通过适配器接入，未安装时自动回退到自研 CPU 求解器）；
> 其余项目仅在算法思想、公式或依赖库层面被引用。

---

## 一、运行时依赖（必需）

| 项目 | 版本约束 | 许可证 | 用途 | 链接 |
|---|---|---|---|---|
| NumPy | >= 2.1 | BSD-3-Clause | 全部数值计算 | https://numpy.org |
| FastAPI | >= 0.115 | MIT | REST / WebSocket 服务框架 | https://github.com/fastapi/fastapi |
| Uvicorn | >= 0.32 | BSD-3-Clause | ASGI 服务器 | https://github.com/encode/uvicorn |
| Pydantic | >= 2.9 | MIT | 参数校验与 API 契约 | https://github.com/pydantic/pydantic |
| Starlette | (随 FastAPI) | BSD-3-Clause | ASGI 工具集 | https://github.com/encode/starlette |
| Vue | ^3.5 | MIT | 前端框架 | https://github.com/vuejs/core |
| Vite | ^5.4 | MIT | 前端构建工具 | https://github.com/vitejs/vite |
| Three.js | ^0.169 | MIT | 三维渲染 | https://github.com/mrdoob/three.js |
| Element Plus | ^2.8 | MIT | 参数面板 UI 组件库 | https://github.com/element-plus/element-plus |
| ECharts | ^5.5 | Apache-2.0 | 曲线图表 | https://github.com/apache/echarts |
| Pinia | ^2.2 | MIT | 前端状态管理 | https://github.com/vuejs/pinia |
| TypeScript | ~5.6 | Apache-2.0 | 类型系统 | https://github.com/microsoft/TypeScript |

## 二、可选加速后端（未安装时自动回退，见 `docs/OPTIONAL_BACKENDS.md`）

| 项目 | 许可证 | 本项目如何使用 | 链接 |
|---|---|---|---|
| **PySPH** | BSD-3-Clause | 可选 SPH 求解后端。`backend/app/physics/pysph_backend.py` 通过其公开 API（`Application` / `Equation` / `Solver`）接入本项目的非牛顿本构。**未复制其源码。** | https://github.com/sweeneyde/pysph |
| **SynxFlow** | MIT | 可选 GPU 深度积分浅水方程求解器。`backend/app/physics/synxflow_backend.py` 预留单一 GPU 集成点；无 CUDA 时按规格要求回退到自研 CPU 浅水方程求解器。**未复制其源码。** | https://github.com/SynxFlow/SynxFlow |
| **PyTorch** | BSD-3-Clause | 可选，用于 CNN-LSTM / DR-LSTM 代理模型的训练与推理（`backend/app/ai/models.py`）。 | https://github.com/pytorch/pytorch |

## 三、算法 / 公式参考（仅参考，未使用其代码）

| 项目 / 文献 | 许可证 | 引用内容 |
|---|---|---|
| **TITAN2D** | GPL-3.0 | 深度积分泥石流模型中的 Voellmy–Coulomb 基底阻力公式形式。**仅公式参考。**<br>https://github.com/TITAN2D/titan2d |
| Monaghan, J.J. (1992), *Smoothed Particle Hydrodynamics*, Annu. Rev. Astron. Astrophys. 30, 543–574 | — | SPH 基本方程、人工粘性 |
| Morris, J.P. et al. (1997), *Modeling low Reynolds number incompressible flows using SPH*, J. Comput. Phys. 136 | — | SPH 粘性拉普拉斯算子 |
| Papanastasiou, T.C. (1987), *Flows of materials with yield*, J. Rheol. 31(5), 385–404 | — | Bingham 屈服面正则化 |
| Cross, M.M. (1965), *Rheology of non-Newtonian fluids*, J. Colloid Sci. 20 | — | Cross 四参数本构 |
| Antuono, M. et al. (2010) | — | δ-SPH 密度扩散思想 |
| Rusanov, V.V. (1961) | — | 近似黎曼通量（local Lax-Friedrichs） |

## 四、被参考但**未使用**的前端脚手架与渲染项目

题目中提到的下列项目属于「可复用参考」，第一版出于**依赖精简与可维护性**
考虑，采用官方 Vite + Vue 3 脚手架自建工程结构，未直接使用其代码：

| 项目 | 许可证 | 说明 |
|---|---|---|
| icegl-three-vue-tres | MIT | Three.js + Vue 3 集成脚手架。<br>https://github.com/icegl/icegl-three-vue-tres |
| craftscape-threejs | MIT | 流体/水体渲染技法参考。<br>https://github.com/erosmarchetti/craftscape-threejs |
| DL-for-landslide-propagation-early-warning | MIT | CNN-LSTM / DR-LSTM 代理模型思路参考。<br>https://github.com/lizhixu123/DL-for-landslide-propagation-early-warning |

若后续版本引入其代码，必须在此处补充对应许可证全文与版权声明。

---

## 五、Harness 插件（开发环境，非本项目运行时依赖）

本项目在 DSH（DeepSeek Harness）Web 界面中开发，安装了以下 DSH 插件，
它们属于**开发工具链**，不参与本系统的运行：

| 插件 | 版本 | 许可证 | 链接 |
|---|---|---|---|
| `@liustack/modlens` | 3.17.2 | 见其仓库 | https://github.com/liustack/modlens |
| `@nanmicoder/dsh-plugin-market` | 0.2.0 | MIT | https://github.com/NanmiCoder/dsh-plugin-market |
| `dsh-better-sidebar` | 0.19.0 | 见其仓库 | https://github.com/omdsh-dev/DSH-better-sidebar |
| `dsh-browser-use` | 0.1.0 | MIT | https://github.com/zoahdev/dsh-browser-use |

---

## 许可证合规提示

1. **PySPH (BSD-3-Clause)**：若分发包含 PySPH 的二进制/源码，需保留其版权声明与免责声明。
2. **SynxFlow (MIT)**：若启用 GPU 后端并分发，需附 MIT 许可证全文。
3. **TITAN2D (GPL-3.0)**：本项目**仅参考公式、未复制代码**，因此不受 GPL 传染。
   若未来引入其任何源码，本项目对应模块须改为 GPL-3.0 兼容许可。
4. **ECharts / TypeScript (Apache-2.0)**：分发时需保留 NOTICE 文件（若上游提供）。
5. 前端生产构建产物中会内联部分依赖的许可证注释，Vite 构建可通过
   `vite-plugin-license` 等工具导出完整清单——第一版未启用该步骤。
