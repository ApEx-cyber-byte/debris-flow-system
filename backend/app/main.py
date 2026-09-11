"""FastAPI 应用入口 / FastAPI application entry point.

启动::

    uvicorn app.main:app --reload --port 8000

交互式 API 文档: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import routes_ai, routes_export, routes_sim, ws
from .config import settings
from .physics.backends import environment_report
from .services.task_manager import task_manager

DESCRIPTION = """
**基于流体动力学的泥石流坡度滚落物理动态分析系统** —— 后端服务。

* **物理内核**：SPH 粒子法（内置 NumPy 实现 / 可选 PySPH 加速）+ 深度积分浅水方程
  （CPU 实现 / 可选 SynxFlow GPU 加速）
* **非牛顿本构**：Bingham（Papanastasiou 正则化）、Cross、Herschel-Bulkley、幂律
* **实时推流**：WebSocket 二进制关键帧（默认每 10 个时间步一帧）
* **AI 预测**：物理启发式代理开箱可用；CNN-LSTM / DR-LSTM 接口已预留
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind the running event loop so worker threads can publish frames."""
    task_manager.bind_loop(asyncio.get_running_loop())
    report = environment_report()
    missing = [name for name in ("pysph", "synxflow") if not report[name]["available"]]
    if missing:
        print(
            "[dfs] 可选求解后端未启用: "
            + ", ".join(missing)
            + " —— 已自动使用内置 CPU 求解器（详见 /api/system/backends）"
        )
    yield
    task_manager.shutdown()


app = FastAPI(
    title="泥石流坡度滚落物理动态分析系统",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_sim.router)
app.include_router(routes_export.router)
app.include_router(routes_ai.router)
app.include_router(ws.router)


@app.get("/", tags=["meta"], summary="服务信息")
def root() -> dict:
    return {
        "name": "基于流体动力学的泥石流坡度滚落物理动态分析系统",
        "version": app.version,
        "docs": "/docs",
        "backends": environment_report(),
    }


@app.get("/health", tags=["meta"], summary="健康检查")
def health() -> dict:
    return {"status": "ok", "runs": len(task_manager.list())}
