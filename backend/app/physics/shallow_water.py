"""CPU 深度积分浅水方程求解器 / CPU depth-integrated shallow-water solver.

当没有 CUDA GPU（因而无法运行 SynxFlow）时，本求解器按规格要求**替代**
GPU 深度积分求解器，在纯 CPU 上以实时速度求解泥石流。

Governing equations
-------------------
质量守恒与两个方向的动量守恒（深度积分、静水压假设）::

    d(h)/dt   + d(hu)/dx + d(hv)/dy = 0
    d(hu)/dt  + d(hu^2 + g h^2/2)/dx + d(huv)/dy = -g h dz_b/dx - tau_bx/rho
    d(hv)/dt  + d(huv)/dx + d(hv^2 + g h^2/2)/dy = -g h dz_b/dy - tau_by/rho

非牛顿本构以两种方式进入方程：

1. **基底阻力** ``tau_b``：屈服应力 ``tau_y`` + 库仑摩擦 ``mu_bed*rho*g*h*cos(beta)``
   + Manning 湍流 ``rho*g*n^2*|V|^2/h^(1/3)``，采用半隐式离散以避免刚性限制，
   并施加「屈服停止」判据实现真实堆积。
2. **深度平均粘性扩散**：表观粘度取自与 SPH 完全相同的本构模块，
   剪切速率用浅水流动的主导尺度 ``gamma_dot ~ |V|/h`` 估计::

       d(hu)/dt += d/dx( nu_app * h * du/dx ) + d/dy( nu_app * h * du/dy )

数值方法
--------
* 守恒变量 ``(h, hu, hv)`` 的有限体积法
* Rusanov (local Lax-Friedrichs) 近似黎曼通量, 对干湿前锋稳健
* 床面坡度源项与压力通量离散一致, 满足静水平衡 (well-balanced)
* 显式时间推进, 自适应时间步受 CFL + 粘性扩散双重限制

Reference: SynxFlow (https://github.com/SynxFlow/SynxFlow, MIT) —— 本模块是
其 CPU 替代实现，未使用 SynxFlow 源码。
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRAVITY
from ..schemas import BoundaryCondition, SimulationParams
from .rheology import apparent_viscosity, basal_shear_stress
from .terrain import Terrain, build_grid, build_terrain

H_DRY = 1.0e-4  # m, dry-cell threshold


class ShallowWaterSolver:
    """Explicit finite-volume depth-integrated solver with Lagrangian tracers."""

    def __init__(self, params: SimulationParams, terrain: Terrain | None = None) -> None:
        self.params = params
        self.terrain = terrain if terrain is not None else build_terrain(params)
        self.grid = build_grid(self.terrain)

        self.rng = np.random.default_rng(params.seed)
        self.rho0 = float(params.density_kg_m3)
        self.time = 0.0
        self.step_count = 0
        self.dt = 1.0e-3

        g = self.grid
        self.xs = g.x0 + (np.arange(g.nx) + 0.5) * g.dx
        self.ys = g.y0 + (np.arange(g.ny) + 0.5) * g.dy
        self.bed = np.asarray(self.terrain.height(self.xs))[None, :].repeat(g.ny, axis=0)
        self.dzdx = np.gradient(self.bed, g.dx, axis=1)
        self.dzdy = np.gradient(self.bed, g.dy, axis=0)
        self.cos_slope = np.asarray(self.terrain.cos_slope(self.xs))[None, :].repeat(g.ny, axis=0)

        self.h = np.zeros((g.ny, g.nx))
        self.hu = np.zeros((g.ny, g.nx))
        self.hv = np.zeros((g.ny, g.nx))
        # Apparent viscosity field from the previous step; drives the
        # adaptive viscous-diffusion time-step limit.
        self.mu_app = np.full((g.ny, g.nx), float(params.viscosity_pa_s))
        self.seed_initial_condition()

        # Lagrangian tracer particles used for the Three.js particle view.
        self._init_tracers()
        self.dt = self._stable_dt()

    # ------------------------------------------------------------------
    def seed_initial_condition(self) -> None:
        g = self.grid
        x0, x1, y0, y1 = self.terrain.release_bounds(self.params)
        thickness = self.terrain.release_thickness(self.params)

        xx = self.xs[None, :].repeat(g.ny, axis=0)
        yy = self.ys[:, None].repeat(g.nx, axis=1)
        inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        self.h = np.where(inside, thickness, 0.0)

        # If the raster is too coarse to hold the release block, widen it so
        # the prescribed volume is preserved.
        volume = float(np.sum(self.h) * g.dx * g.dy)
        if volume > 1.0e-9:
            self.h *= float(self.params.initial_volume_m3) / volume
        self.hu[:] = 0.0
        self.hv[:] = 0.0

    # ------------------------------------------------------------------
    def _init_tracers(self) -> None:
        g = self.grid
        n = int(self.params.particle_count)
        x0, x1, y0, y1 = self.terrain.release_bounds(self.params)
        thickness = self.terrain.release_thickness(self.params)

        self.tracer_x = self.rng.uniform(x0, x1, size=n)
        self.tracer_y = self.rng.uniform(y0, y1, size=n)
        self.tracer_alpha = self.rng.uniform(0.02, 1.0, size=n)
        self.tracer_z = self.terrain.height(self.tracer_x) + self.tracer_alpha * thickness
        self.tracer_active = np.ones(n, dtype=bool)

    # ------------------------------------------------------------------
    def _stable_dt(self) -> float:
        g = self.grid
        speed = np.sqrt(np.maximum(self.hu, 0.0) ** 2 + self.hv ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = np.where(self.h > H_DRY, speed / np.maximum(self.h, H_DRY), 0.0)
        wave = np.sqrt(GRAVITY * np.maximum(self.h, 0.0))
        c_max = float(np.max(wave + vel)) if self.h.size else 1.0
        c_max = max(c_max, 1.0)

        cell = min(g.dx, g.dy)
        dt_cfl = self.params.cfl_number * cell / c_max

        # Viscous diffusion limit. The diffusive coefficient in the momentum
        # equation is ``nu * h``, not ``nu``: a cell holding virtually no
        # material cannot constrain the step no matter how large its apparent
        # viscosity grows. Using bare ``nu`` here would let stagnant near-dry
        # cells pin dt to the viscosity cap and cost ~7x in run time.
        nu = self.mu_app / self.rho0
        diffusion = float(np.max(nu * np.maximum(self.h, 0.0)))
        dt_visc = 0.25 * cell * cell / max(diffusion, 1.0e-9)

        return float(max(1.0e-5, min(dt_cfl, dt_visc, 0.05)))

    # ------------------------------------------------------------------
    def step(self, dt: float | None = None) -> float:
        """Advance one step; ``dt`` defaults to the internally computed limit."""
        used = self.dt if dt is None else min(float(dt), self.dt)
        self._advance(used)
        self.time += used
        self.step_count += 1
        self.dt = self._stable_dt()
        return used

    # ------------------------------------------------------------------
    def _rusanov_x(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rusanov fluxes at x-interfaces, shape ``(ny, nx+1)``."""
        g = self.grid
        h = self.h
        hu = self.hu
        hv = self.hv

        hL = h[:, :-1]
        hR = h[:, 1:]
        huL, huR = hu[:, :-1], hu[:, 1:]
        hvL, hvR = hv[:, :-1], hv[:, 1:]

        uL = np.where(hL > H_DRY, huL / np.maximum(hL, H_DRY), 0.0)
        uR = np.where(hR > H_DRY, huR / np.maximum(hR, H_DRY), 0.0)
        vL = np.where(hL > H_DRY, hvL / np.maximum(hL, H_DRY), 0.0)
        vR = np.where(hR > H_DRY, hvR / np.maximum(hR, H_DRY), 0.0)

        cL = np.sqrt(GRAVITY * hL)
        cR = np.sqrt(GRAVITY * hR)
        a = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

        fh = 0.5 * (huL + huR) - 0.5 * a * (hR - hL)
        fhu = 0.5 * (huL * uL + 0.5 * GRAVITY * hL * hL + huR * uR + 0.5 * GRAVITY * hR * hR) \
            - 0.5 * a * (huR - huL)
        fhv = 0.5 * (huL * vL + huR * vR) - 0.5 * a * (hvR - hvL)

        out = (np.zeros((g.ny, g.nx + 1)), np.zeros((g.ny, g.nx + 1)), np.zeros((g.ny, g.nx + 1)))
        out[0][:, 1:-1] = fh
        out[1][:, 1:-1] = fhu
        out[2][:, 1:-1] = fhv
        return out

    def _rusanov_y(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rusanov fluxes at y-interfaces, shape ``(ny+1, nx)``."""
        g = self.grid
        h, hu, hv = self.h, self.hu, self.hv

        hL, hR = h[:-1, :], h[1:, :]
        huL, huR = hu[:-1, :], hu[1:, :]
        hvL, hvR = hv[:-1, :], hv[1:, :]

        uL = np.where(hL > H_DRY, huL / np.maximum(hL, H_DRY), 0.0)
        uR = np.where(hR > H_DRY, huR / np.maximum(hR, H_DRY), 0.0)
        vL = np.where(hL > H_DRY, hvL / np.maximum(hL, H_DRY), 0.0)
        vR = np.where(hR > H_DRY, hvR / np.maximum(hR, H_DRY), 0.0)

        cL = np.sqrt(GRAVITY * hL)
        cR = np.sqrt(GRAVITY * hR)
        a = np.maximum(np.abs(vL) + cL, np.abs(vR) + cR)

        gh = 0.5 * (hvL + hvR) - 0.5 * a * (hR - hL)
        ghu = 0.5 * (huL * vL + huR * vR) - 0.5 * a * (huR - huL)
        ghv = 0.5 * (hvL * vL + 0.5 * GRAVITY * hL * hL + hvR * vR + 0.5 * GRAVITY * hR * hR) \
            - 0.5 * a * (hvR - hvL)

        out = (np.zeros((g.ny + 1, g.nx)), np.zeros((g.ny + 1, g.nx)), np.zeros((g.ny + 1, g.nx)))
        out[0][1:-1, :] = gh
        out[1][1:-1, :] = ghu
        out[2][1:-1, :] = ghv
        return out

    # ------------------------------------------------------------------
    def _advance(self, dt: float) -> None:
        g = self.grid
        fx, fhu, fhv = self._rusanov_x()
        gy, ghu, ghv = self._rusanov_y()

        dh = -(fx[:, 1:] - fx[:, :-1]) / g.dx - (gy[1:, :] - gy[:-1, :]) / g.dy
        dhu = -(fhu[:, 1:] - fhu[:, :-1]) / g.dx - (ghu[1:, :] - ghu[:-1, :]) / g.dy
        dhv = -(fhv[:, 1:] - fhv[:, :-1]) / g.dx - (ghv[1:, :] - ghv[:-1, :]) / g.dy

        # --- bed-slope source (consistent with the pressure flux) ---------
        dhu -= GRAVITY * self.h * self.dzdx
        dhv -= GRAVITY * self.h * self.dzdy

        h_new = self.h + dt * dh
        hu_new = self.hu + dt * dhu
        hv_new = self.hv + dt * dhv

        # --- depth-averaged non-Newtonian viscous diffusion ---------------
        h_pos = np.maximum(h_new, 0.0)
        u = np.where(h_pos > H_DRY, hu_new / np.maximum(h_pos, H_DRY), 0.0)
        v = np.where(h_pos > H_DRY, hv_new / np.maximum(h_pos, H_DRY), 0.0)
        speed = np.sqrt(u * u + v * v)

        # Shallow flows are dominated by vertical shear: gamma_dot ~ |V|/h.
        gamma_dot = speed / np.maximum(h_pos, 0.05)
        mu_app = apparent_viscosity(gamma_dot, self.params)
        self.mu_app = mu_app
        nu = mu_app / self.rho0

        nu_h = nu * np.maximum(h_pos, 0.0)
        lap_u = (
            (nu_h * np.roll(u, -1, axis=1) - 2.0 * nu_h * u + nu_h * np.roll(u, 1, axis=1)) / (g.dx * g.dx)
            + (nu_h * np.roll(u, -1, axis=0) - 2.0 * nu_h * u + nu_h * np.roll(u, 1, axis=0)) / (g.dy * g.dy)
        )
        lap_v = (
            (nu_h * np.roll(v, -1, axis=1) - 2.0 * nu_h * v + nu_h * np.roll(v, 1, axis=1)) / (g.dx * g.dx)
            + (nu_h * np.roll(v, -1, axis=0) - 2.0 * nu_h * v + nu_h * np.roll(v, 1, axis=0)) / (g.dy * g.dy)
        )
        hu_new += dt * lap_u
        hv_new += dt * lap_v

        # --- basal resistance, semi-implicit with an exact arrest criterion -
        h_pos = np.maximum(h_new, 0.0)
        u = np.where(h_pos > H_DRY, hu_new / np.maximum(h_pos, H_DRY), 0.0)
        v = np.where(h_pos > H_DRY, hv_new / np.maximum(h_pos, H_DRY), 0.0)
        speed = np.sqrt(u * u + v * v)

        tau_b = basal_shear_stress(h_pos, speed, self.cos_slope, self.params)
        a_res = tau_b / (self.rho0 * np.maximum(h_pos, 1.0e-3))

        arresting = (a_res * dt) >= speed
        denom = 1.0 + np.where(speed > 1.0e-9, dt * a_res / np.maximum(speed, 1.0e-9), 0.0)
        hu_new = np.where(arresting, 0.0, hu_new / denom)
        hv_new = np.where(arresting, 0.0, hv_new / denom)
        hu_new = np.where(h_pos > H_DRY, hu_new, 0.0)
        hv_new = np.where(h_pos > H_DRY, hv_new, 0.0)

        # --- boundaries ---------------------------------------------------
        self.h = np.maximum(h_new, 0.0)
        self.hu = hu_new
        self.hv = hv_new
        self._apply_boundaries()

        self._advance_tracers(dt)

    # ------------------------------------------------------------------
    def _apply_boundaries(self) -> None:
        # Solid lateral walls (cross-slope).
        self.hu[:, 0] = 0.0
        self.hu[:, -1] = 0.0
        self.hv[0, :] = 0.0
        self.hv[-1, :] = 0.0

        if self.params.boundary is BoundaryCondition.WALL:
            g = self.grid
            x_wall = self.terrain.toe_x_m + self.params.barrier_distance_m
            col = int(np.clip((x_wall - g.x0) / g.dx, 0, g.nx - 2))
            # Zero the downstream velocity and remove material beyond the wall.
            self.hu[:, col] = np.minimum(self.hu[:, col], 0.0)
            self.hu[:, col + 1:] = 0.0
            self.hv[:, col + 1:] = 0.0

    # ------------------------------------------------------------------
    def _sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample of a cell-centred field at world coordinates."""
        g = self.grid
        fx = np.clip((x - g.x0) / g.dx - 0.5, 0.0, g.nx - 1.0001)
        fy = np.clip((y - g.y0) / g.dy - 0.5, 0.0, g.ny - 1.0001)
        i0 = np.floor(fx).astype(np.int64)
        j0 = np.floor(fy).astype(np.int64)
        i1 = np.minimum(i0 + 1, g.nx - 1)
        j1 = np.minimum(j0 + 1, g.ny - 1)
        tx = fx - i0
        ty = fy - j0
        return (
            field[j0, i0] * (1 - tx) * (1 - ty)
            + field[j0, i1] * tx * (1 - ty)
            + field[j1, i0] * (1 - tx) * ty
            + field[j1, i1] * tx * ty
        )

    def _advance_tracers(self, dt: float) -> None:
        h_pos = np.maximum(self.h, 0.0)
        u = np.where(h_pos > H_DRY, self.hu / np.maximum(h_pos, H_DRY), 0.0)
        v = np.where(h_pos > H_DRY, self.hv / np.maximum(h_pos, H_DRY), 0.0)

        h_loc = self._sample(h_pos, self.tracer_x, self.tracer_y)
        # A tracer is deposited once the local column has drained below the
        # dry threshold; this reproduces the final deposit footprint.
        self.tracer_active &= h_loc > H_DRY * 10.0

        u_loc = self._sample(u, self.tracer_x, self.tracer_y)
        v_loc = self._sample(v, self.tracer_x, self.tracer_y)

        self.tracer_x = np.where(self.tracer_active, self.tracer_x + dt * u_loc, self.tracer_x)
        self.tracer_y = np.where(self.tracer_active, self.tracer_y + dt * v_loc, self.tracer_y)

        g = self.grid
        self.tracer_x = np.clip(self.tracer_x, g.x0, g.x0 + (g.nx - 1) * g.dx)
        self.tracer_y = np.clip(self.tracer_y, g.y0, g.y0 + (g.ny - 1) * g.dy)

        h_loc = self._sample(h_pos, self.tracer_x, self.tracer_y)
        bed = np.asarray(self.terrain.height(self.tracer_x))
        self.tracer_z = np.where(
            self.tracer_active,
            bed + self.tracer_alpha * np.maximum(h_loc, 0.0),
            self.tracer_z,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def depth_field(self) -> np.ndarray:
        return np.maximum(self.h, 0.0)

    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        h_pos = np.maximum(self.h, H_DRY)
        wet = self.h > H_DRY
        u = np.where(wet, self.hu / h_pos, 0.0)
        v = np.where(wet, self.hv / h_pos, 0.0)
        return u, v

    def particle_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Tracer positions ``(N, 3)`` and velocities ``(N, 3)`` for rendering."""
        u, v = self.velocity_field()
        u_loc = self._sample(u, self.tracer_x, self.tracer_y)
        v_loc = self._sample(v, self.tracer_x, self.tracer_y)
        pos = np.stack([self.tracer_x, self.tracer_y, self.tracer_z], axis=1)
        vel = np.stack(
            [u_loc, v_loc, np.zeros_like(u_loc)], axis=1
        )
        return pos, vel

    def impact_force(self, depth: np.ndarray, vx: np.ndarray) -> float:
        g = self.grid
        x_monitor = self.terrain.toe_x_m + self.params.barrier_distance_m
        col = int(np.clip((x_monitor - g.x0) / g.dx, 0, g.nx - 1))
        h = depth[:, col]
        u = np.abs(vx[:, col])
        rho = self.rho0
        pressure = rho * h * u * u + 0.5 * rho * GRAVITY * h * h
        return float(np.sum(pressure) * g.dy)

    def summary(self, depth: np.ndarray) -> dict[str, float]:
        u, v = self.velocity_field()
        speed = np.sqrt(u * u + v * v)
        wet = self.h > H_DRY
        front = float(g_quantile(self.xs, depth, 0.995)) if np.any(wet) else 0.0
        ke = float(
            np.sum(0.5 * self.rho0 * np.maximum(self.h, 0.0) * speed * speed) * self.grid.dx * self.grid.dy
        )
        deposited = float(np.sum(~self.tracer_active) * self.params.initial_volume_m3 / max(self.tracer_x.size, 1))
        return {
            "max_speed": float(speed.max()) if speed.size else 0.0,
            "mean_speed": float(speed[wet].mean()) if np.any(wet) else 0.0,
            "front_x": front,
            "max_depth": float(depth.max()) if depth.size else 0.0,
            "kinetic_energy": ke,
            "deposited_volume": deposited,
        }


def g_quantile(xs: np.ndarray, depth: np.ndarray, q: float) -> float:
    """Approximate frontal position: the ``q`` quantile of wet-cell x-coordinates.

    Weighted by column volume so a thin spurious film at the leading edge does
    not dominate the reported front.
    """
    mass = depth.sum(axis=0)
    total = mass.sum()
    if total <= 1.0e-12:
        return 0.0
    cumulative = np.cumsum(mass) / total
    idx = int(np.searchsorted(cumulative, q))
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])
