"""PySPH 可选加速后端 / Optional PySPH accelerated backend.

本模块把系统的非牛顿本构（:mod:`app.physics.rheology`）接入
`PySPH <https://github.com/sweeneyde/pysph>`_ (BSD-3-Clause) 的求解框架。

.. warning::
   **该后端是可选加速路径，不是系统的必需依赖。**
   PySPH 在 PyPI 上只提供源码包（``pysph-1.0b2.tar.gz``），安装需要
   Cython + C++ 编译工具链。本机当前未安装 PySPH，因此这条代码路径
   **未经运行时验证**；它的失败被 :class:`app.services.task_manager` 捕获
   并自动回退到内置 NumPy SPH 求解器，不会影响系统的可用性。
   安装并验证方法见 ``docs/OPTIONAL_BACKENDS.md``。

Equation set
------------
1. ``SummationDensity``                     -- PySPH 内置求和密度
2. ``ShearRateTensor``                      -- 自定义：SPH 速度梯度张量 -> gamma_dot
3. ``TaitEquationOfState``                  -- PySPH 内置 Tait 状态方程
4. ``NonNewtonianViscosity``                -- 自定义：mu_app(gamma_dot) 的 Morris 粘性
5. ``BasalFrictionForce``                   -- 自定义：Voellmy-Coulomb + Manning 基底摩擦
6. ``MomentumEquation``                     -- PySPH 内置对称压力梯度动量方程
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRAVITY
from ..schemas import SimulationParams
from .rheology import apparent_viscosity, basal_shear_stress
from .terrain import Terrain, build_grid, build_terrain


def _require_pysph():
    try:
        import pysph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only on machines without PySPH
        raise RuntimeError(
            "PySPH 后端被请求但 PySPH 未安装。请执行 pip install pysph，"
            "或使用 backend=auto / shallow_water。"
        ) from exc


class PySPHSolver:
    """Adapter exposing the common solver interface over a PySPH application.

    The class builds a PySPH ``Application`` lazily on first use and mirrors
    the ``step``/diagnostics surface of :class:`NumPySPHSolver` so the rest of
    the system is backend-agnostic.
    """

    def __init__(self, params: SimulationParams, terrain: Terrain | None = None) -> None:
        _require_pysph()

        self.params = params
        self.terrain = terrain if terrain is not None else build_terrain(params)
        self.grid = build_grid(self.terrain)
        self.rho0 = float(params.density_kg_m3)
        self.time = 0.0
        self.step_count = 0

        self._app = self._build_application()
        self.dx_mean = self._app.dx_mean
        self.volume_particle = self._app.volume_particle
        self.mass_particle = self._app.mass_particle

    # ------------------------------------------------------------------
    def _build_application(self):  # pragma: no cover - requires PySPH
        from pysph.base.kernels import CubicSpline
        from pysph.base.utils import get_particle_array
        from pysph.solver.application import Application
        from pysph.solver.solver import Solver
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.equation import Equation, Group
        from pysph.sph.wc.basic import MomentumEquation, TaitEquationOfState

        params = self.params
        terrain = self.terrain
        rho0 = self.rho0
        self_ref = self

        # -- particle seeding (same lattice as the NumPy solver) --------
        x0, x1, y0, y1 = terrain.release_bounds(params)
        thickness = terrain.release_thickness(params)
        volume_total = float(params.initial_volume_m3)
        target = int(params.particle_count)
        spacing = (volume_total / max(target, 1)) ** (1.0 / 3.0)

        nx = max(1, int(round((x1 - x0) / spacing)))
        ny = max(1, int(round((y1 - y0) / spacing)))
        nz = max(1, int(round(thickness / spacing)))
        xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        zs = (np.arange(nz) + 0.5) * thickness / nz
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

        x = gx.ravel()
        y = gy.ravel()
        z = gz.ravel() + terrain.height(x)
        n = x.size
        m = rho0 * volume_total / n
        h = 1.3 * (volume_total / n) ** (1.0 / 3.0)

        self.dx_mean = (volume_total / n) ** (1.0 / 3.0)
        self.volume_particle = volume_total / n
        self.mass_particle = m

        class ShearRateTensor(Equation):
            """SPH velocity-gradient tensor -> shear rate -> apparent viscosity."""

            def __init__(self, dest, sources):
                super().__init__(dest, sources)

            def initialize(self, d_idx, d_gammadot, d_mu):
                d_gammadot[d_idx] = 0.0
                d_mu[d_idx] = 0.0

            def loop(
                self, d_idx, s_idx, d_L, s_m, s_rho, d_u, d_v, d_w,
                s_u, s_v, s_w, DWIJ, XIJ,
            ):
                # Accumulate  dv_a/dx_b  via  -sum_j V_j (v_j-v_a) dW/dx_b
                vol_j = s_m[s_idx] / s_rho[s_idx]
                dv = (s_u[s_idx] - d_u[d_idx], s_v[s_idx] - d_v[d_idx], s_w[s_idx] - d_w[d_idx])
                base = d_idx * 9
                for a in range(3):
                    for b in range(3):
                        d_L[base + a * 3 + b] -= vol_j * dv[a] * DWIJ[b]

            def post_loop(self, d_idx, d_L, d_gammadot, d_mu):
                base = d_idx * 9
                L = np.array([d_L[base + k] for k in range(9)]).reshape(3, 3)
                D = 0.5 * (L + L.T)
                gd = math.sqrt(max(0.0, 2.0 * float(np.sum(D * D))))
                d_gammadot[d_idx] = gd
                d_mu[d_idx] = self_ref._mu_of(gd)

        class NonNewtonianViscosity(Equation):
            """Morris et al. (1997) laminar viscosity with a per-particle mu."""

            def loop(
                self, d_idx, s_idx, d_au, d_av, d_aw, d_u, d_v, d_w,
                s_u, s_v, s_w, d_rho, s_rho, d_mu, s_mu, s_m, R2IJ, DWIJ, XIJ, EPS,
            ):
                mu_bar = 0.5 * (d_mu[d_idx] + s_mu[s_idx])
                fac = 2.0 * s_m[s_idx] * mu_bar / (d_rho[d_idx] * s_rho[s_idx])
                du = d_u[d_idx] - s_u[s_idx]
                dv = d_v[d_idx] - s_v[s_idx]
                dw = d_w[d_idx] - s_w[s_idx]
                dot = XIJ[0] * du + XIJ[1] * dv + XIJ[2] * dw
                fac2 = fac * dot / (R2IJ + 0.01 * EPS)
                d_au[d_idx] += fac2 * DWIJ[0]
                d_av[d_idx] += fac2 * DWIJ[1]
                d_aw[d_idx] += fac2 * DWIJ[2]

        class BasalFrictionForce(Equation):
            """Voellmy-Coulomb + Manning basal resistance, bed layer only."""

            def __init__(self, dest, sources, params, gravity=GRAVITY):
                super().__init__(dest, sources)
                self.params = params
                self.gravity = gravity

            def initialize(self, d_idx, d_au, d_av, d_aw, d_u, d_v, d_w,
                           d_hcol, d_cosslope, d_z, d_zb, d_dx):
                depth = max(float(d_hcol[d_idx]), 1.0e-3)
                speed = math.sqrt(d_u[d_idx] ** 2 + d_v[d_idx] ** 2 + d_w[d_idx] ** 2)
                if speed < 1.0e-9:
                    return
                tau = float(
                    basal_shear_stress(
                        np.array([depth]), np.array([speed]),
                        np.array([d_cosslope[d_idx]]), self.params, self.gravity,
                    )[0]
                )
                decel = tau / (self.params.density_kg_m3 * depth)
                # Distribute over the bed layer (one particle diameter).
                scale = min(1.0, decel / speed)
                d_au[d_idx] -= d_u[d_idx] * scale
                d_av[d_idx] -= d_v[d_idx] * scale
                d_aw[d_idx] -= d_w[d_idx] * scale

        class DebrisFlowApp(Application):
            def create_particles(self):
                pa = get_particle_array(
                    name="debris", x=x, y=y, z=z, h=h, m=m, rho=rho0,
                )
                n_local = pa.num_real_particles
                pa.add_property("gammadot", zeros=n_local)
                pa.add_property("mu", zeros=n_local)
                pa.add_property("L", zeros=9 * n_local)
                pa.add_property("hcol", zeros=n_local)
                pa.add_property("cosslope", zeros=n_local)
                pa.add_property("zb", zeros=n_local)
                pa.add_property("dx", zeros=n_local)
                pa.hcol[:] = self_ref.terrain.release_thickness(self_ref.params)
                pa.cosslope[:] = self_ref.terrain.cos_slope(pa.x)
                pa.zb[:] = self_ref.terrain.height(pa.x)
                pa.dx[:] = self_ref.dx_mean
                pa.mu[:] = self_ref.params.viscosity_pa_s
                return [pa]

            def create_solver(self):
                kernel = CubicSpline(dim=3)
                solver = Solver(
                    dim=3,
                    kernel=kernel,
                    integrator="symplectic",
                    dt=2.0e-4,
                    tf=self_ref.params.duration_s,
                    adaptive_timestep=True,
                    cfl=self_ref.params.cfl_number,
                )
                return solver

            def create_equations(self):
                return [
                    Group(equations=[SummationDensity(dest="debris", sources=["debris"])]),
                    Group(equations=[
                        ShearRateTensor(dest="debris", sources=["debris"]),
                        TaitEquationOfState(
                            dest="debris", sources=None,
                            rho0=rho0, gamma=7.0, b=self_ref._tait_b(),
                        ),
                    ]),
                    Group(equations=[
                        NonNewtonianViscosity(dest="debris", sources=["debris"]),
                        MomentumEquation(dest="debris", sources=["debris"], alpha=0.15),
                        BasalFrictionForce(
                            dest="debris", sources=None, params=self_ref.params
                        ),
                    ]),
                ]

        return DebrisFlowApp()

    # ------------------------------------------------------------------
    def _mu_of(self, gamma_dot: float) -> float:  # pragma: no cover
        return float(apparent_viscosity(np.array([gamma_dot]), self.params)[0])

    def _tait_b(self) -> float:  # pragma: no cover
        v_scale = max(1.0, math.sqrt(2.0 * GRAVITY * max(self.terrain.crest_height_m, 1.0)))
        c0 = 10.0 * v_scale
        return self.rho0 * c0 * c0 / 7.0

    # ------------------------------------------------------------------
    # Common solver interface
    # ------------------------------------------------------------------
    def step(self, dt: float | None = None) -> float:  # pragma: no cover - requires PySPH
        self._app.solver.step(self._app.solver.t)
        self.time = float(self._app.solver.t)
        self.step_count += 1
        return float(self._app.solver.dt)

    def _arrays(self):  # pragma: no cover
        pa = self._app.particles[0]
        return pa

    def particle_state(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        pa = self._arrays()
        pos = np.stack([pa.x, pa.y, pa.z], axis=1)
        vel = np.stack([pa.u, pa.v, pa.w], axis=1)
        return pos, vel

    def depth_field(self) -> np.ndarray:  # pragma: no cover
        pos, _ = self.particle_state()
        return _rasterise(pos, self.grid, self.volume_particle)

    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        pos, vel = self.particle_state()
        return _rasterise_velocity(pos, vel, self.grid)

    def impact_force(self, depth: np.ndarray, vx: np.ndarray) -> float:  # pragma: no cover
        g = self.grid
        col = int(np.clip(
            (self.terrain.toe_x_m + self.params.barrier_distance_m - g.x0) / g.dx, 0, g.nx - 1
        ))
        h = depth[:, col]
        u = np.abs(vx[:, col])
        return float(np.sum(self.rho0 * h * u * u + 0.5 * self.rho0 * GRAVITY * h * h) * g.dy)

    def summary(self, depth: np.ndarray) -> dict[str, float]:  # pragma: no cover
        pos, vel = self.particle_state()
        speed = np.linalg.norm(vel, axis=1)
        moving = speed > 0.05
        return {
            "max_speed": float(speed.max()) if speed.size else 0.0,
            "mean_speed": float(speed[moving].mean()) if np.any(moving) else 0.0,
            "front_x": float(np.percentile(pos[:, 0], 99.5)) if pos.size else 0.0,
            "max_depth": float(depth.max()) if depth.size else 0.0,
            "kinetic_energy": float(np.sum(0.5 * self.mass_particle * speed * speed)),
            "deposited_volume": float(np.sum(~moving) * self.volume_particle),
        }


# ---------------------------------------------------------------------------
def _rasterise(pos: np.ndarray, grid, volume_particle: float) -> np.ndarray:  # pragma: no cover
    ix = np.floor((pos[:, 0] - grid.x0) / grid.dx).astype(np.int64)
    iy = np.floor((pos[:, 1] - grid.y0) / grid.dy).astype(np.int64)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    counts = np.zeros(grid.nx * grid.ny)
    if np.any(ok):
        counts = np.bincount(iy[ok] * grid.nx + ix[ok], minlength=grid.nx * grid.ny).astype(float)
    return (counts * volume_particle / (grid.dx * grid.dy)).reshape(grid.ny, grid.nx)


def _rasterise_velocity(pos, vel, grid):  # pragma: no cover
    ix = np.floor((pos[:, 0] - grid.x0) / grid.dx).astype(np.int64)
    iy = np.floor((pos[:, 1] - grid.y0) / grid.dy).astype(np.int64)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    size = grid.nx * grid.ny
    vx = np.zeros(size)
    vy = np.zeros(size)
    if np.any(ok):
        cell = iy[ok] * grid.nx + ix[ok]
        counts = np.bincount(cell, minlength=size)
        vx = np.bincount(cell, weights=vel[ok, 0], minlength=size)
        vy = np.bincount(cell, weights=vel[ok, 1], minlength=size)
        nz = counts > 0
        vx[nz] /= counts[nz]
        vy[nz] /= counts[nz]
    return vx.reshape(grid.ny, grid.nx), vy.reshape(grid.ny, grid.nx)
