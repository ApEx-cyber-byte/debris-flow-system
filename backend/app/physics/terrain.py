"""斜坡—坡脚—堆积区地形几何 / Slope–toe–runout bed geometry.

床面高度场采用「软斜坡」(softplus hinge) 解析表达式，使坡面到水平坡脚的
转折连续可导 —— 既避免数值上的速度脉冲，又保持解析梯度便于施加基底摩擦。

Coordinate system
-----------------
* ``x`` : 沿坡向下 (down-slope) 为正，坡顶 (crest) 位于 ``x = 0``
* ``y`` : 横向 (cross-slope)
* ``z`` : 竖直向上，坡顶床面高度为 ``L*sin(theta)``，坡脚远处趋于 0

Bed elevation
-------------
``z_b(x) = R*tan(theta) * softplus((x_toe - x)/R)``

其中 ``R`` 为坡脚转折半径，``x_toe = L*cos(theta)``。当 ``x << x_toe`` 时
``z_b ~ (x_toe - x)*tan(theta)``（严格线性坡面）；当 ``x >> x_toe`` 时
``z_b -> 0``（水平堆积区）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schemas import GridInfo, SimulationParams, TerrainInfo


def softplus(t: np.ndarray | float) -> np.ndarray:
    """Numerically stable ``ln(1 + exp(t))``."""
    arr = np.asarray(t, dtype=np.float64)
    return np.maximum(arr, 0.0) + np.log1p(np.exp(-np.abs(arr)))


def sigmoid(t: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function."""
    arr = np.asarray(t, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_t = np.exp(arr[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


@dataclass(frozen=True)
class Terrain:
    """Analytic bed surface plus the computational domain it spans."""

    slope_angle_deg: float
    slope_angle_rad: float
    slope_length_m: float
    toe_x_m: float
    crest_height_m: float
    turn_radius_m: float
    domain_x_min: float
    domain_x_max: float
    domain_y_min: float
    domain_y_max: float
    channel_width_m: float
    release_length_m: float

    # -- surface ---------------------------------------------------------
    def height(self, x: np.ndarray | float, y: np.ndarray | float | None = None) -> np.ndarray:
        """床面高程 ``z_b(x, y)`` (m)。对 y 无依赖（横断面水平）。"""
        xa = np.asarray(x, dtype=np.float64)
        t = (self.toe_x_m - xa) / self.turn_radius_m
        return self.turn_radius_m * math.tan(self.slope_angle_rad) * softplus(t)

    def gradient(self, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """床面梯度 ``(dz/dx, dz/dy)``。

        ``dz/dx = -tan(theta) * sigmoid((x_toe - x)/R)``，``dz/dy = 0``。
        """
        xa = np.asarray(x, dtype=np.float64)
        t = (self.toe_x_m - xa) / self.turn_radius_m
        dzdx = -math.tan(self.slope_angle_rad) * sigmoid(t)
        return dzdx, np.zeros_like(xa)

    def normal(self, x: np.ndarray | float) -> np.ndarray:
        """单位床面外法向 ``n = (-dz/dx, -dz/dy, 1)/|...|``，形状 ``(..., 3)``。"""
        dzdx, dzdy = self.gradient(x)
        raw = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        return raw / np.linalg.norm(raw, axis=-1, keepdims=True)

    def cos_slope(self, x: np.ndarray | float) -> np.ndarray:
        """床面法向竖直分量 ``n_z = cos(beta)``，用于库仑摩擦项。"""
        dzdx, dzdy = self.gradient(x)
        return 1.0 / np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)

    # -- release block ---------------------------------------------------
    def release_bounds(self, params: SimulationParams) -> tuple[float, float, float, float]:
        """释放区在 ``(x, y)`` 平面上的包围盒 ``(x0, x1, y0, y1)``。

        泥石流起动块体紧贴坡顶布置：``x0 = 0``（坡顶），沿坡向下延伸
        ``release_length_m``，横向居中且宽度为 ``channel_width_m``。
        """
        x0 = 0.0
        x1 = min(self.toe_x_m, x0 + params.release_length_m)
        half = 0.5 * self.channel_width_m
        return x0, x1, -half, half

    def release_thickness(self, params: SimulationParams) -> float:
        """释放体厚度 h0 (m)：由体积 / 释放区水平投影面积得到。"""
        if params.release_height_m > 0.0:
            return float(params.release_height_m)
        x0, x1, y0, y1 = self.release_bounds(params)
        area = max(1.0e-6, (x1 - x0) * (y1 - y0))
        return float(params.initial_volume_m3 / area)

    # -- serialisation ---------------------------------------------------
    def to_info(self) -> TerrainInfo:
        return TerrainInfo(
            slope_angle_deg=self.slope_angle_deg,
            slope_length_m=self.slope_length_m,
            toe_x_m=self.toe_x_m,
            crest_height_m=self.crest_height_m,
            turn_radius_m=self.turn_radius_m,
            domain_x_min=self.domain_x_min,
            domain_x_max=self.domain_x_max,
            domain_y_min=self.domain_y_min,
            domain_y_max=self.domain_y_max,
        )


def build_terrain(params: SimulationParams) -> Terrain:
    """从用户参数构造地形与计算域。

    堆积区长度取 ``2.5 * 坡面水平投影``，横向范围取释放区宽度的 4 倍，
    保证扩散后的堆积体不会过早触达边界。

    坡脚转折半径 ``R`` 取坡面水平投影的 8%（下限 0.4 m）。这个取值很关键：
    softplus 铰链在 ``x`` 远小于坡脚时按 ``sigmoid((x_toe-x)/R)`` 衰减，
    ``R`` 过大会把**整条坡面**都削圆 —— 例如 ``R = 0.25*x_toe`` 时，
    坡面中段的坡度只剩理论值的 88.6%，直接削弱驱动力。取 8% 后，
    释放区内的坡度保持在 ``tan(theta)`` 的 99.8% 以上，同时坡脚仍有
    约 ±2 m 的连续过渡，避免转折处的数值脉冲。
    """
    theta = params.slope_angle_rad
    toe_x = params.toe_x_m
    crest_h = params.crest_height_m

    # A pure vertical drop (theta = 90 deg) is out of spec (<= 60 deg), so
    # tan(theta) stays finite; guard only against numerical dust.
    turn_radius = max(0.4, 0.08 * max(toe_x, 1.0))

    runout_length = max(8.0, 2.5 * max(toe_x, 1.0))
    margin_x = 0.15 * max(toe_x, 1.0)
    margin_y = 2.0 * params.channel_width_m

    return Terrain(
        slope_angle_deg=params.slope_angle_deg,
        slope_angle_rad=theta,
        slope_length_m=params.slope_length_m,
        toe_x_m=toe_x,
        crest_height_m=crest_h,
        turn_radius_m=turn_radius,
        domain_x_min=-margin_x,
        domain_x_max=toe_x + runout_length,
        domain_y_min=-0.5 * params.channel_width_m - margin_y,
        domain_y_max=0.5 * params.channel_width_m + margin_y,
        channel_width_m=params.channel_width_m,
        release_length_m=params.release_length_m,
    )


def build_grid(terrain: Terrain, target_cells: int = 220) -> GridInfo:
    """深度/速度栅格元数据。``dx = dy``，格点数控制在 ``target_cells`` 量级。"""
    lx = terrain.domain_x_max - terrain.domain_x_min
    ly = terrain.domain_y_max - terrain.domain_y_min
    cell = max(lx, ly) / float(target_cells)
    nx = max(8, int(math.ceil(lx / cell)))
    ny = max(8, int(math.ceil(ly / cell)))
    return GridInfo(
        nx=nx,
        ny=ny,
        x0=terrain.domain_x_min,
        y0=terrain.domain_y_min,
        dx=lx / nx,
        dy=ly / ny,
    )


def domain_height_field(terrain: Terrain, grid: GridInfo) -> np.ndarray:
    """Terrain elevation sampled on the raster grid, shape ``(ny, nx)``."""
    xs = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    return np.asarray(terrain.height(xs), dtype=np.float64)[None, :].repeat(grid.ny, axis=0)
