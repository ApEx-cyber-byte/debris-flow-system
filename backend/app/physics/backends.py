"""求解后端探测与工厂 / Backend probing and factory.

系统提供四个求解后端，按「能跑就快、跑不了就退」的原则选择：

============  ==========================  ==============================
后端           依赖                         说明
============  ==========================  ==============================
``numpy_sph``  仅 NumPy（总是可用）         内置向量化 SPH，完整非牛顿本构
``shallow_water`` 仅 NumPy（总是可用）      深度积分浅水方程，CPU 实时
``pysph``      PySPH + C++ 编译器           可选加速 SPH
``synxflow``   SynxFlow + CUDA GPU          可选 GPU 深度积分求解
============  ==========================  ==============================

``auto`` 的解析顺序：SynxFlow（有 GPU 时）→ PySPH → CPU 浅水方程。
选择 CPU 浅水方程作为默认回退，是因为规格明确要求
「SynxFlow 需要 CUDA GPU，没有 GPU 时先用 CPU 浅水方程求解器替代」。
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from ..schemas import BackendInfo, SimulationParams, SolverBackend


class BackendUnavailable(RuntimeError):
    """Raised when an explicitly requested backend cannot be loaded."""


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendAvailability:
    pysph: bool
    synxflow: bool
    torch: bool
    cuda: bool
    notes: list[str]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


_AVAILABILITY: BackendAvailability | None = None


def probe(force: bool = False) -> BackendAvailability:
    """Detect optional dependencies once per process."""
    global _AVAILABILITY
    if _AVAILABILITY is not None and not force:
        return _AVAILABILITY

    notes: list[str] = []
    pysph = _module_available("pysph")
    synxflow = _module_available("synxflow")
    torch = _module_available("torch")
    cuda = False

    if not pysph:
        notes.append(
            "PySPH 未安装：内置 NumPy SPH 求解器已接管 "
            "(pip install pysph 需要 Cython 与 C++ 编译工具链)"
        )
    if not synxflow:
        notes.append(
            "SynxFlow 未安装：CPU 浅水方程求解器已接管 "
            "(pip install synxflow 需要 CUDA Toolkit 与 cupy)"
        )
    else:
        try:  # pragma: no cover - only runs where SynxFlow exists
            import cupy  # type: ignore

            cuda = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:  # noqa: BLE001 - any failure means "no usable CUDA"
            cuda = False
        if not cuda:
            notes.append("SynxFlow 已安装但未检测到可用 CUDA 设备，回退到 CPU 浅水方程求解器")
    if not torch:
        notes.append("PyTorch 未安装：AI 预测走内置物理启发式代理模型")

    _AVAILABILITY = BackendAvailability(
        pysph=pysph, synxflow=synxflow and cuda, torch=torch, cuda=cuda, notes=notes
    )
    return _AVAILABILITY


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
BACKEND_LABELS = {
    "numpy_sph": "内置 NumPy SPH（CPU，完整非牛顿本构）",
    "shallow_water": "CPU 深度积分浅水方程（SynxFlow 的 CPU 替代）",
    "pysph": "PySPH（可选加速 SPH）",
    "synxflow": "SynxFlow（GPU 深度积分浅水方程）",
}


def resolve(params: SimulationParams) -> tuple[str, BackendInfo]:
    """Map a requested backend onto one that can actually run here."""
    avail = probe()
    requested = params.backend

    if requested is SolverBackend.PYSPH:
        if not avail.pysph:
            raise BackendUnavailable(
                "请求的 PySPH 后端不可用：未安装 PySPH。"
                "请先 pip install pysph，或改用 backend=auto。"
            )
        active = "pysph"
    elif requested is SolverBackend.SYNXFLOW:
        if not avail.synxflow:
            raise BackendUnavailable(
                "请求的 SynxFlow 后端不可用：未安装 SynxFlow 或没有可用 CUDA GPU。"
                "请改用 backend=shallow_water（CPU 替代）或 backend=auto。"
            )
        active = "synxflow"
    elif requested is SolverBackend.NUMPY_SPH:
        active = "numpy_sph"
    elif requested is SolverBackend.SHALLOW_WATER:
        active = "shallow_water"
    else:  # AUTO
        if avail.synxflow:
            active = "synxflow"
        elif avail.pysph:
            active = "pysph"
        else:
            active = "shallow_water"

    info = BackendInfo(
        requested=requested.value,
        active=active,
        accelerated=active in ("pysph", "synxflow"),
        notes=list(avail.notes),
    )
    return active, info


def create_solver(params: SimulationParams):
    """Instantiate the solver for ``params`` (resolving ``auto`` first)."""
    active, info = resolve(params)

    if active == "pysph":
        from .pysph_backend import PySPHSolver  # noqa: PLC0415 - lazy optional import

        return PySPHSolver(params), info

    if active == "synxflow":
        from .synxflow_backend import SynxFlowSolver  # noqa: PLC0415

        return SynxFlowSolver(params), info

    if active == "numpy_sph":
        from .sph_numpy import NumPySPHSolver  # noqa: PLC0415

        return NumPySPHSolver(params), info

    from .shallow_water import ShallowWaterSolver  # noqa: PLC0415

    return ShallowWaterSolver(params), info


def environment_report() -> dict[str, object]:
    """Diagnostics surfaced by ``GET /api/system/backends``."""
    avail = probe()
    return {
        "pysph": {"available": avail.pysph, "url": "https://github.com/sweeneyde/pysph"},
        "synxflow": {
            "available": avail.synxflow,
            "cuda": avail.cuda,
            "url": "https://github.com/SynxFlow/SynxFlow",
        },
        "torch": {"available": avail.torch, "url": "https://pytorch.org"},
        "default_backend": os.environ.get("DFS_BACKEND", "auto"),
        "labels": BACKEND_LABELS,
        "notes": avail.notes,
        "cpu_count": os.cpu_count(),
    }
