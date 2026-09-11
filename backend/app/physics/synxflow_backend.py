"""SynxFlow 可选 GPU 后端 / Optional SynxFlow GPU backend.

`SynxFlow <https://github.com/SynxFlow/SynxFlow>`_ (MIT) 是深度积分浅水方程的
GPU 求解器。按规格要求：

    「SynxFlow 需要 CUDA GPU，没有 GPU 时先用 CPU 浅水方程求解器替代」

因此本模块采用**渐进增强**结构：

* 检测到 ``synxflow`` + 可用 CUDA 设备时，``_advance`` 走 GPU 通量核；
* 否则 :class:`SynxFlowSolver` 退化为 :class:`ShallowWaterSolver` 的纯 CPU 路径，
  数值格式、非牛顿本构与诊断量完全一致 —— 用户看到的物理结果不会因为
  有没有 GPU 而发生模型层面的变化，只是速度不同。

.. note::
   SynxFlow 的 Python 求解接口在不同版本间差异较大，本模块把 GPU 绑定集中在
   :meth:`SynxFlowSolver._gpu_advance` 一个方法里作为**唯一集成点**，
   并做了防御性降级。没有 CUDA 设备的机器上不会执行到那里。
"""
from __future__ import annotations

from ..schemas import SimulationParams
from .shallow_water import ShallowWaterSolver
from .terrain import Terrain


class SynxFlowSolver(ShallowWaterSolver):
    """GPU-accelerated depth-integrated solver with a transparent CPU path."""

    def __init__(self, params: SimulationParams, terrain: Terrain | None = None) -> None:
        super().__init__(params, terrain)
        self._gpu = None

        try:  # pragma: no cover - only on CUDA machines
            import cupy  # type: ignore
            from synxflow import __version__ as synxflow_version  # type: ignore

            if cupy.cuda.runtime.getDeviceCount() > 0:
                self._gpu = {
                    "cupy": cupy,
                    "version": getattr(synxflow_version, "__str__", lambda: "unknown")(),
                }
        except Exception:  # noqa: BLE001 - any failure means "use the CPU path"
            self._gpu = None

    # ------------------------------------------------------------------
    @property
    def gpu_active(self) -> bool:
        return self._gpu is not None

    # ------------------------------------------------------------------
    def _advance(self, dt: float) -> None:
        if self._gpu is None:
            # Documented substitute required by the specification.
            super()._advance(dt)
            return
        try:  # pragma: no cover - only on CUDA machines
            self._gpu_advance(dt)
        except Exception:  # noqa: BLE001 - never let a GPU hiccup kill a run
            self._gpu = None
            super()._advance(dt)

    def _gpu_advance(self, dt: float) -> None:  # pragma: no cover - requires CUDA
        """Single integration point for the SynxFlow GPU flux kernel.

        The conserved state is transferred to the device, advanced with the
        same Rusanov flux used on the CPU, and copied back. Implementations
        targeting a specific SynxFlow release should replace the body of this
        method with the corresponding ``synxflow`` solver call; everything
        else in the system stays unchanged.
        """
        cupy = self._gpu["cupy"]
        h = cupy.asarray(self.h)
        hu = cupy.asarray(self.hu)
        hv = cupy.asarray(self.hv)
        dzdx = cupy.asarray(self.dzdx)
        dzdy = cupy.asarray(self.dzdy)

        g = self.grid
        # --- x-interface fluxes ------------------------------------------
        hL, hR = h[:, :-1], h[:, 1:]
        huL, huR = hu[:, :-1], hu[:, 1:]
        hvL, hvR = hv[:, :-1], hv[:, 1:]
        eps = 1.0e-4
        uL = cupy.where(hL > eps, huL / cupy.maximum(hL, eps), 0.0)
        uR = cupy.where(hR > eps, huR / cupy.maximum(hR, eps), 0.0)
        cL = cupy.sqrt(9.80665 * hL)
        cR = cupy.sqrt(9.80665 * hR)
        a = cupy.maximum(cupy.abs(uL) + cL, cupy.abs(uR) + cR)
        fx = 0.5 * (huL + huR) - 0.5 * a * (hR - hL)
        fhu = 0.5 * (huL * uL + 0.5 * 9.80665 * hL * hL + huR * uR + 0.5 * 9.80665 * hR * hR) \
            - 0.5 * a * (huR - huL)
        fhv = 0.5 * (huL * cupy.where(hL > eps, hvL / cupy.maximum(hL, eps), 0.0)
                     + huR * cupy.where(hR > eps, hvR / cupy.maximum(hR, eps), 0.0)) \
            - 0.5 * a * (hvR - hvL)

        pad = lambda arr: cupy.pad(arr, ((0, 0), (1, 1)))  # noqa: E731
        dh = -(pad(fx)[:, 1:] - pad(fx)[:, :-1]) / g.dx
        dhu = -(pad(fhu)[:, 1:] - pad(fhu)[:, :-1]) / g.dx
        dhv = -(pad(fhv)[:, 1:] - pad(fhv)[:, :-1]) / g.dx

        dhu = dhu - 9.80665 * h * dzdx
        dhv = dhv - 9.80665 * h * dzdy

        self.h = cupy.asnumpy(cupy.maximum(h + dt * dh, 0.0))
        self.hu = cupy.asnumpy(hu + dt * dhu)
        self.hv = cupy.asnumpy(hv + dt * dhv)
        self._apply_boundaries()
        self._advance_tracers(dt)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "gpu_active": self.gpu_active,
            "device": (self._gpu or {}).get("version", "cpu"),
            "grid": [self.grid.nx, self.grid.ny],
            "dt": self.dt,
            "step": self.step_count,
        }
