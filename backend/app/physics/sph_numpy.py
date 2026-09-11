"""内置 NumPy SPH 求解器 / Built-in vectorised SPH solver.

这是系统的**保底求解器**：不依赖 PySPH、不依赖 GPU、不依赖编译器，
只要有 NumPy 就能跑，且完整实现了泥石流的非牛顿本构耦合。

Numerical scheme
----------------
* **核函数**    : 3D 三次样条核 (cubic spline), 支撑域 ``2h``
* **密度**      : 求和密度 ``rho_i = C_i * sum_j m_j W_ij``，其中 ``C_i`` 是
  在 **初始构型** 上标定的逐粒子修正因子。标定消除了初始点阵偏差
  （否则 2% 的密度误差会经 Tait 方程放大成 30 倍的虚假超压），
  同时完整保留了密度对压缩的响应 —— 这正是 WCSPH 产生压力的物理来源
* **状态方程**  : Tait 方程 ``p = B[(rho/rho0)^gamma - 1]``，``gamma = 7``，
  并截断 ``p >= 0`` 以消除自由面张力不稳定性
* **压力梯度**  : 对称形式 ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W_ij``
* **粘性**      : Morris 等 (1997) 拉普拉斯算子，粘度取本构模型的表观粘度
                  ``mu_app(gamma_dot)``；对角项半隐式积分, 允许远大于
                  显式粘性稳定限的时间步长
* **人工粘性**  : Monaghan 人工粘性, 抑制激波/粒子穿透
* **基底摩擦**  : Voellmy-Coulomb + Manning 组合, 逐粒子施加于床面层
* **邻域搜索**  : 均匀网格链表 + 排序 + 27 邻域, 全向量化

The apparent viscosity is evaluated from the SPH velocity-gradient tensor at
every particle, so the Bingham yield surface emerges from the constitutive law
rather than being imposed by hand.

Reference: Monaghan, J.J. (1992) *Smoothed Particle Hydrodynamics*,
Annu. Rev. Astron. Astrophys. 30, 543-574.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import GRAVITY
from ..schemas import BoundaryCondition, SimulationParams
from .rheology import apparent_viscosity, basal_shear_stress
from .terrain import Terrain, build_grid, build_terrain


# ---------------------------------------------------------------------------
# Cubic spline kernel (3D), support radius 2h
# ---------------------------------------------------------------------------
KERNEL_SIGMA_3D = 1.0 / math.pi


def kernel_values(q: np.ndarray, h: float, out_w: np.ndarray, out_dw: np.ndarray) -> None:
    """Evaluate ``W(q)`` and ``dW/dr(q)`` for ``q = r/h`` in place."""
    inv_h3 = 1.0 / (h * h * h)
    inv_h4 = inv_h3 / h

    inner = q < 1.0
    outer = ~inner

    # f(q) = 1 - 1.5 q^2 + 0.75 q^3 ;  f'(q) = -3q + 2.25 q^2
    qi = q[inner]
    out_w[inner] = (1.0 - 1.5 * qi * qi + 0.75 * qi * qi * qi) * KERNEL_SIGMA_3D * inv_h3
    out_dw[inner] = (-3.0 * qi + 2.25 * qi * qi) * KERNEL_SIGMA_3D * inv_h4

    # f(q) = 0.25 (2-q)^3 ;  f'(q) = -0.75 (2-q)^2
    qo = q[outer]
    two_minus = 2.0 - qo
    out_w[outer] = 0.25 * two_minus ** 3 * KERNEL_SIGMA_3D * inv_h3
    out_dw[outer] = -0.75 * two_minus * two_minus * KERNEL_SIGMA_3D * inv_h4


@dataclass
class SPHState:
    """Mutable particle state; kept separate so snapshots are cheap to copy."""

    pos: np.ndarray
    vel: np.ndarray
    active: np.ndarray = field(default=None)  # type: ignore[assignment]


class NumPySPHSolver:
    """Weakly-compressible SPH for non-Newtonian debris flow."""

    def __init__(self, params: SimulationParams, terrain: Terrain | None = None) -> None:
        self.params = params
        self.terrain = terrain if terrain is not None else build_terrain(params)
        self.grid = build_grid(self.terrain)

        self.rng = np.random.default_rng(params.seed)
        self.rho0 = float(params.density_kg_m3)
        self.time = 0.0
        self.step_count = 0

        self._initialise_particles()

        # --- numerical parameters ---------------------------------------
        # Smoothing length: 1.3x mean particle spacing gives ~57 neighbours.
        self.h = 1.3 * self.dx_mean
        self.support = 2.0 * self.h
        self.cell_size = self.support
        self.eta2 = (0.01 * self.h) ** 2

        # Sound speed: Mach ~0.1 keeps density fluctuations near 1%.
        self.v_scale = max(1.0, math.sqrt(2.0 * GRAVITY * max(self.terrain.crest_height_m, 1.0)))
        self.c0 = 10.0 * self.v_scale
        self.tait_b = self.rho0 * self.c0 * self.c0 / 7.0
        self.alpha_av = 0.15  # Monaghan artificial viscosity

        # Per-particle density renormalisation, calibrated on the initial
        # lattice. Without it the raw kernel sum carries an O(10%) lattice
        # bias which the Tait equation amplifies into a spurious overpressure.
        self.density_scale = 1.0
        self.w_ref = 1.0
        self._calibrate_density()
        # δ-SPH style density smoothing strength (0 disables).
        self.density_diffusion = 0.10

        self.dt = self._stable_dt(np.zeros_like(self.vel))

        # Cached per-step quantities (filled by ``step``).
        self.rho = np.full(self.n, self.rho0)
        self.pressure = np.zeros(self.n)
        self.gamma_dot = np.zeros(self.n)
        self.mu_app = np.full(self.n, params.viscosity_pa_s)
        self.depth = np.full(self.n, self.dx_mean)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialise_particles(self) -> None:
        p = self.params
        x0, x1, y0, y1 = self.terrain.release_bounds(p)
        thickness = self.terrain.release_thickness(p)

        volume_total = float(p.initial_volume_m3)
        target = int(p.particle_count)
        seed_spacing = (volume_total / max(target, 1)) ** (1.0 / 3.0)

        nx = max(1, int(round((x1 - x0) / seed_spacing)))
        ny = max(1, int(round((y1 - y0) / seed_spacing)))
        nz = max(1, int(round(thickness / seed_spacing)))

        xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        zs = (np.arange(nz) + 0.5) * thickness / nz

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        # Sit the block on the bed rather than at z = 0.
        bed = self.terrain.height(pos[:, 0])
        pos[:, 2] += bed

        # Break the perfect lattice so the initial state is not artificially
        # ordered (a lattice produces persistent spurious pressure modes).
        jitter = 0.06 * (volume_total / pos.shape[0]) ** (1.0 / 3.0)
        pos += self.rng.uniform(-jitter, jitter, size=pos.shape)

        self.n = pos.shape[0]
        self.volume_particle = volume_total / self.n
        self.mass_particle = self.rho0 * self.volume_particle
        self.dx_mean = self.volume_particle ** (1.0 / 3.0)

        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.active = np.ones(self.n, dtype=bool)
        self.start_x = pos[:, 0].copy()

    # ------------------------------------------------------------------
    # Diagnostics / state
    # ------------------------------------------------------------------
    @property
    def crest_reference_x(self) -> float:
        return float(np.min(self.start_x))

    def _stable_dt(self, accel: np.ndarray) -> float:
        speed = float(np.max(np.linalg.norm(self.vel, axis=1))) if self.n else 0.0
        dt_cfl = self.params.cfl_number * self.h / (self.c0 + speed)

        a_max = float(np.max(np.linalg.norm(accel, axis=1))) if accel.size else 0.0
        if a_max > 1.0e-9:
            dt_force = 0.25 * math.sqrt(self.h / a_max)
        else:
            dt_force = dt_cfl

        return float(max(1.0e-6, min(dt_cfl, dt_force)))

    # ------------------------------------------------------------------
    # Neighbour search
    # ------------------------------------------------------------------
    def _pairs_and_kernels(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Neighbour pairs plus kernel values.

        Returns ``(i, j, r_ij, r2, W, dW/dr, gradW)`` where ``gradW`` has shape
        ``(M, 3)`` and holds ``grad_i W_ij``.
        """
        i_idx, j_idx = self._build_pairs()
        if i_idx.size == 0:
            empty = np.empty(0)
            return (
                i_idx, j_idx, np.empty((0, 3)), empty, empty, empty, np.empty((0, 3)),
            )

        # Distance test before materialising the (M, 3) separation vector:
        # the cell stencil over-fetches by ~6x, and slicing a full vector array
        # for every candidate pair costs far more than three scalar subtractions.
        px, py, pz = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        dx = px[i_idx] - px[j_idx]
        dy = py[i_idx] - py[j_idx]
        dz = pz[i_idx] - pz[j_idx]
        r2 = dx * dx + dy * dy + dz * dz

        keep = (r2 > 1.0e-12) & (r2 < self.support * self.support)
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
        dx, dy, dz = dx[keep], dy[keep], dz[keep]
        r2 = r2[keep]

        if r2.size == 0:
            empty = np.empty(0)
            return (
                i_idx, j_idx, np.empty((0, 3)), empty, empty, empty, np.empty((0, 3)),
            )

        r = np.sqrt(r2)
        r_ij = np.stack([dx, dy, dz], axis=1)
        w = np.empty_like(r)
        dwdr = np.empty_like(r)
        kernel_values(r / self.h, self.h, w, dwdr)
        grad_w = (dwdr / r)[:, None] * r_ij
        return i_idx, j_idx, r_ij, r2, w, dwdr, grad_w

    def _calibrate_density(self) -> None:
        """Derive one global density scale from the initial lattice.

        The raw kernel sum ``sum_j W_ij`` under-resolves on a finite lattice
        (the cubic spline has a kink in its second derivative at ``q = 1``, so
        the quadrature carries an O(10%) bias at ``h = 1.3 dx``). Left alone,
        the Tait equation turns that bias into a spurious overpressure.

        Calibrating **one** scale factor from well-supported interior
        particles -- rather than a per-particle factor -- keeps the field's
        response to compression intact and makes free-surface handling fall
        out for free: a surface particle has fewer neighbours, so its density
        drops below ``rho0`` and ``p = max(p, 0)`` gives it atmospheric
        pressure. A per-particle correction would instead have to be clipped,
        and the clipped ones detonate when the flow later buries them.
        """
        i_idx, _j, _rij, _r2, w, _dw, _g = self._pairs_and_kernels()
        if i_idx.size == 0:
            self.density_scale = 1.0
            self.w_ref = 1.0
            return
        num = np.bincount(i_idx, weights=w, minlength=self.n)
        self.w_ref = float(np.percentile(num, 95.0))
        self.density_scale = self.rho0 / max(self.mass_particle * self.w_ref, 1.0e-12)

    def _build_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(i, j)`` directed neighbour index arrays."""
        pos = self.pos
        lo = pos.min(axis=0) - self.support
        span = (pos.max(axis=0) + self.support) - lo
        nc = np.maximum(1, np.ceil(span / self.cell_size).astype(np.int64))

        ci = ((pos - lo) / self.cell_size).astype(np.int64)
        np.clip(ci, 0, nc - 1, out=ci)

        lin = (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]
        order = np.argsort(lin, kind="stable")
        lin_sorted = lin[order]

        index = np.arange(self.n)
        i_parts: list[np.ndarray] = []
        j_parts: list[np.ndarray] = []

        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    tc = ci + (ox, oy, oz)
                    valid = (
                        (tc[:, 0] >= 0) & (tc[:, 0] < nc[0])
                        & (tc[:, 1] >= 0) & (tc[:, 1] < nc[1])
                        & (tc[:, 2] >= 0) & (tc[:, 2] < nc[2])
                    )
                    tlin = np.where(
                        valid,
                        (tc[:, 0] * nc[1] + tc[:, 1]) * nc[2] + tc[:, 2],
                        -1,
                    )
                    left = np.searchsorted(lin_sorted, tlin, side="left")
                    right = np.searchsorted(lin_sorted, tlin, side="right")
                    counts = right - left
                    total = int(counts.sum())
                    if total == 0:
                        continue
                    csum = np.cumsum(counts)
                    starts = np.repeat(left, counts)
                    within = np.arange(total) - np.repeat(csum - counts, counts)
                    i_parts.append(np.repeat(index, counts))
                    j_parts.append(order[starts + within])

        if not i_parts:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        return np.concatenate(i_parts), np.concatenate(j_parts)

    # ------------------------------------------------------------------
    # One time step
    # ------------------------------------------------------------------
    def step(self) -> float:
        """Advance the solution by one internal step; returns ``dt`` used."""
        dt = self._advance(self.dt)
        self.time += dt
        self.step_count += 1
        return dt

    def _advance(self, dt: float) -> float:
        pos, vel = self.pos, self.vel
        m_p = self.mass_particle

        i_idx, j_idx, r_ij, r2, w, dwdr, grad_w = self._pairs_and_kernels()
        has_pairs = r2.size > 0

        # --- summation density with the lattice-calibrated scale -----------
        if has_pairs:
            num = np.bincount(i_idx, weights=w, minlength=self.n)
            rho = self.mass_particle * num * self.density_scale

            # δ-SPH style density diffusion: a kernel-weighted average of the
            # density field. It removes the high-frequency noise a lattice
            # produces without touching a uniform field (where it is a no-op).
            volume_j = self.mass_particle / rho[j_idx]
            weight_sum = np.bincount(i_idx, weights=volume_j * w, minlength=self.n)
            density_sum = np.bincount(
                i_idx, weights=volume_j * rho[j_idx] * w, minlength=self.n
            )
            smoothed = density_sum / np.maximum(weight_sum, 1.0e-12)
            rho = (1.0 - self.density_diffusion) * rho + self.density_diffusion * smoothed
        else:
            rho = np.full(self.n, self.rho0)

        rho = np.clip(rho, 0.5 * self.rho0, 3.0 * self.rho0)
        self.rho = rho

        # --- Tait equation of state ---------------------------------------
        ratio = rho / self.rho0
        pressure = self.tait_b * (np.power(ratio, 7.0) - 1.0)
        # Clipping to zero removes tension: a particle whose kernel sum fell
        # below its calibrated value (free surface, spray) simply carries no
        # pressure instead of pulling its neighbours inward.
        pressure = np.maximum(pressure, 0.0)
        if has_pairs:
            # Isolated particles (strongly separated) must not carry pressure.
            isolated = num < 0.35 * self.w_ref
            pressure[isolated] = 0.0
        self.pressure = pressure

        # --- pressure acceleration ----------------------------------------
        accel = np.zeros_like(vel)
        accel[:, 2] -= GRAVITY

        if has_pairs:
            p_term = (
                self.pressure[i_idx] / (rho[i_idx] * rho[i_idx])
                + self.pressure[j_idx] / (rho[j_idx] * rho[j_idx])
            )
            coeff = -m_p * p_term
            for k in range(3):
                accel[:, k] += np.bincount(
                    i_idx, weights=coeff * grad_w[:, k], minlength=self.n
                )

            # --- artificial viscosity (Monaghan) --------------------------
            v_ij = vel[i_idx] - vel[j_idx]
            vr = np.einsum("ij,ij->i", v_ij, r_ij)
            mu_ij = self.h * vr / (r2 + self.eta2)
            approaching = vr < 0.0
            if np.any(approaching):
                rho_bar = 0.5 * (rho[i_idx] + rho[j_idx])
                pi_ij = np.where(
                    approaching,
                    -self.alpha_av * self.c0 * mu_ij / np.maximum(rho_bar, 1.0e-9),
                    0.0,
                )
                av_coeff = -m_p * pi_ij
                for k in range(3):
                    accel[:, k] += np.bincount(
                        i_idx, weights=av_coeff * grad_w[:, k], minlength=self.n
                    )

            # --- shear rate tensor:  L_ab = -sum_j V_j (v_j-v_i)_a dW/dx_b -
            vol_j = m_p / rho[j_idx]
            dv = vel[j_idx] - vel[i_idx]  # (M, 3)
            grad_components = [
                np.bincount(i_idx, weights=vol_j * dv[:, a] * grad_w[:, b], minlength=self.n)
                for a in range(3)
                for b in range(3)
            ]
            L = np.stack(grad_components, axis=1).reshape(self.n, 3, 3)
            L = -L
            D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
            gamma_dot = np.sqrt(np.maximum(2.0 * np.einsum("nij,nij->n", D, D), 0.0))
            self.gamma_dot = gamma_dot

            mu = apparent_viscosity(gamma_dot, self.params)
            self.mu_app = mu

            # --- viscous acceleration (Morris et al. 1997) ----------------
            # a_visc = 2 * sum_j m_j (mu_i+mu_j)/(rho_i rho_j) * (v_i-v_j)
            #          * (r . grad_i W) / (r^2 + eta^2)
            mu_pair = mu[i_idx] + mu[j_idx]
            # r_ij . grad_i W_ij = (dW/dr) * r
            r_dot_gradw = dwdr * np.sqrt(r2)
            c_ij = (
                2.0
                * m_p
                * mu_pair
                / (rho[i_idx] * rho[j_idx])
                * r_dot_gradw
                / (r2 + self.eta2)
            )
            # c_ij <= 0 for a diffusive kernel; define the positive coupling.
            coupling = np.maximum(-c_ij, 0.0)

            c_sum = np.bincount(i_idx, weights=coupling, minlength=self.n)
            coupling_vel = np.zeros_like(vel)
            for k in range(3):
                coupling_vel[:, k] = np.bincount(
                    i_idx, weights=coupling * vel[j_idx, k], minlength=self.n
                )
        else:
            self.gamma_dot = np.zeros(self.n)
            self.mu_app = np.full(self.n, self.params.viscosity_pa_s)
            c_sum = np.zeros(self.n)
            coupling_vel = np.zeros_like(vel)

        # --- bed interaction ---------------------------------------------
        bed = self.terrain.height(pos[:, 0])
        clearance = pos[:, 2] - bed
        radius = 0.5 * self.dx_mean
        touching = clearance < radius

        if np.any(touching):
            pos[touching, 2] = bed[touching] + radius
            normal = self.terrain.normal(pos[touching, 0])
            vt_normal = np.einsum("ij,ij->i", vel[touching], normal)
            # Inelastic normal contact: project the velocity onto the bed.
            vel[touching] -= np.where(
                vt_normal < 0.0, vt_normal, 0.0
            )[:, None] * normal

        # --- integrate momentum (semi-implicit viscosity diagonal) --------
        force = accel + coupling_vel
        vel_new = (vel + dt * force) / (1.0 + dt * c_sum[:, None])

        # --- basal friction (Voellmy-Coulomb + Manning) -------------------
        depth_column = self._column_depth()
        self.depth = depth_column
        speed = np.linalg.norm(vel_new, axis=1)
        cos_slope = self.terrain.cos_slope(pos[:, 0])
        tau_b = basal_shear_stress(depth_column, speed, cos_slope, self.params)
        decel = tau_b / (self.rho0 * np.maximum(depth_column, 1.0e-3))

        # Friction acts only inside the bed layer, and can arrest the material.
        bed_layer = radius + self.dx_mean
        near_bed = (pos[:, 2] - bed) < bed_layer
        if np.any(near_bed):
            scale = np.ones(self.n)
            scale[near_bed] = np.maximum(0.0, 1.0 - decel[near_bed] * dt / np.maximum(speed[near_bed], 1.0e-9))
            vel_new *= scale[:, None]

        # --- lateral / downstream confinement -----------------------------
        vel_new, pos = self._apply_domain_bounds(pos, vel_new)

        self.vel = vel_new
        self.pos = pos + dt * vel_new

        # The next step's dt is derived from the acceleration just computed.
        self.dt = self._stable_dt(accel)
        return dt

    # ------------------------------------------------------------------
    def _column_depth(self) -> np.ndarray:
        """Local flow depth per particle, sampled from the rasterised column."""
        grid = self.grid
        pos = self.pos
        ix = np.floor((pos[:, 0] - grid.x0) / grid.dx).astype(np.int64)
        iy = np.floor((pos[:, 1] - grid.y0) / grid.dy).astype(np.int64)
        ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        depth = np.full(self.n, 1.0e-3)
        if not np.any(ok):
            return depth
        cell = iy[ok] * grid.nx + ix[ok]
        counts = np.bincount(cell, minlength=grid.nx * grid.ny)
        cell_area = grid.dx * grid.dy
        height = counts * self.volume_particle / cell_area
        depth[ok] = height[cell]
        return np.maximum(depth, 1.0e-3)

    def _apply_domain_bounds(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confine particles to the computational flume."""
        t = self.terrain

        # Lateral walls (cross-slope): perfectly reflecting.
        below = pos[:, 1] < t.domain_y_min
        above = pos[:, 1] > t.domain_y_max
        vel[below, 1] = np.abs(vel[below, 1])
        vel[above, 1] = -np.abs(vel[above, 1])
        np.clip(pos[:, 1], t.domain_y_min, t.domain_y_max, out=pos[:, 1])

        # Downstream: either an open outflow or a rigid barrier.
        x_wall = None
        if self.params.boundary is BoundaryCondition.WALL:
            x_wall = t.toe_x_m + self.params.barrier_distance_m
        elif t.domain_x_max:
            x_wall = t.domain_x_max

        if x_wall is not None:
            beyond = pos[:, 0] > x_wall
            if np.any(beyond):
                if self.params.boundary is BoundaryCondition.WALL:
                    pos[beyond, 0] = x_wall
                    vel[beyond, 0] = -np.abs(vel[beyond, 0]) * 0.05
                else:
                    # Open boundary: clamp softly so the runout tail is kept
                    # visible instead of silently disappearing.
                    pos[beyond, 0] = x_wall
                    vel[beyond, 0] = np.minimum(vel[beyond, 0], 0.0)

        return vel, pos

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def depth_field(self) -> np.ndarray:
        """Column height ``h(x, y)`` on the raster grid, shape ``(ny, nx)``."""
        grid = self.grid
        pos = self.pos
        ix = np.floor((pos[:, 0] - grid.x0) / grid.dx).astype(np.int64)
        iy = np.floor((pos[:, 1] - grid.y0) / grid.dy).astype(np.int64)
        ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        counts = np.zeros(grid.nx * grid.ny)
        if np.any(ok):
            counts = np.bincount(
                iy[ok] * grid.nx + ix[ok], minlength=grid.nx * grid.ny
            ).astype(np.float64)
        return (counts * self.volume_particle / (grid.dx * grid.dy)).reshape(grid.ny, grid.nx)

    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        """Depth-averaged velocity on the raster grid, each ``(ny, nx)``."""
        grid = self.grid
        pos = self.pos
        ix = np.floor((pos[:, 0] - grid.x0) / grid.dx).astype(np.int64)
        iy = np.floor((pos[:, 1] - grid.y0) / grid.dy).astype(np.int64)
        ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        size = grid.nx * grid.ny
        vx = np.zeros(size)
        vy = np.zeros(size)
        if np.any(ok):
            cell = iy[ok] * grid.nx + ix[ok]
            counts = np.bincount(cell, minlength=size)
            vx = np.bincount(cell, weights=self.vel[ok, 0], minlength=size)
            vy = np.bincount(cell, weights=self.vel[ok, 1], minlength=size)
            nz = counts > 0
            vx[nz] /= counts[nz]
            vy[nz] /= counts[nz]
        return vx.reshape(grid.ny, grid.nx), vy.reshape(grid.ny, grid.nx)

    def particle_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Particle positions ``(N, 3)`` and velocities ``(N, 3)``."""
        return self.pos, self.vel

    def impact_force(self, depth: np.ndarray, vx: np.ndarray) -> float:
        """冲击力 (N) 作用于坡脚下游监测断面 —— 动压 + 静压积分。

        ``F = sum_y [ rho*h*v_x^2 + 0.5*rho*g*h^2 ] * dy``
        """
        grid = self.grid
        x_monitor = self.terrain.toe_x_m + self.params.barrier_distance_m
        col = int(np.clip((x_monitor - grid.x0) / grid.dx, 0, grid.nx - 1))
        h = depth[:, col]
        u = np.abs(vx[:, col])
        rho = self.rho0
        pressure = rho * h * u * u + 0.5 * rho * GRAVITY * h * h
        return float(np.sum(pressure) * grid.dy)

    def summary(self, depth: np.ndarray) -> dict[str, float]:
        speed = np.linalg.norm(self.vel, axis=1)
        moving = speed > 0.05
        front = float(np.percentile(self.pos[:, 0], 99.5)) if self.n else 0.0
        ke = float(np.sum(0.5 * self.mass_particle * speed * speed))
        deposited = float(np.sum(~moving) * self.volume_particle)
        return {
            "max_speed": float(speed.max()) if self.n else 0.0,
            "mean_speed": float(speed[moving].mean()) if np.any(moving) else 0.0,
            "front_x": front,
            "max_depth": float(depth.max()) if depth.size else 0.0,
            "kinetic_energy": ke,
            "deposited_volume": deposited,
        }
