"""物理启发式代理模型 / Physics-informed fast surrogate.

用途
----
用户拖动参数滑块时，AI 接口要能**立刻**给出量级正确的预测（跑距、峰值流速、
到达时间、冲击力、影响范围），而不是等几十秒的完整数值模拟。

方法
----
把泥石流的质量中心当作一个沿程一维质点，用与完整求解器**完全相同的
非牛顿本构与基底阻力公式**积分其动量方程：

.. math::

    \\frac{dv}{dt} = g\\sin\\theta - \\mu_{bed}\\,g\\cos\\theta
                     - \\frac{\\tau_y}{\\rho h}
                     - \\frac{g n^2 v^2}{h^{4/3}}

坡面上积分到坡脚得到 ``v_toe``；随后在水平堆积区以同样阻力减速到零，
积分距离即为跑距 ``R``。这比纯经验回归更可靠，也比完整模拟快 4~5 个数量级
（毫秒级），适合作为 UI 的实时预判。

.. note::
   这是**代理模型的第一版基线**。``app/ai/models.py`` 给出了 CNN-LSTM 与
   DR-LSTM 的网络定义与权重加载路径；一旦用数值模拟批量生成的数据训练完成，
   把 ``AIPredictor`` 的默认模型切换为 ``cnn-lstm-v1`` 即可获得学习型代理，
   接口与返回结构保持不变。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import GRAVITY
from ..schemas import SimulationParams
from ..physics.terrain import build_terrain


@dataclass
class SurrogatePrediction:
    runout_distance_m: float
    max_velocity_m_s: float
    arrival_time_s: float
    max_flow_depth_m: float
    peak_impact_force_kn: float
    affected_area_m2: float
    confidence: float


# ---------------------------------------------------------------------------
# Calibrated coefficients
# ---------------------------------------------------------------------------
# A one-dimensional centre-of-mass model cannot represent the vertical
# structure of the flow: the fast, thin front outruns the centre of mass, and
# a Bingham fluid carries a plug core that the depth-averaged resistance
# over-penalises. Two transparent coefficients absorb that structural gap.
# Both were fitted by `scripts/calibrate_surrogate.py` against full
# depth-integrated simulations on 8 parameter sets; see
# `scripts/validate_surrogate.py` for the resulting error report.
#
# DEPTH_FACTOR > 1 thickens the effective runout-phase depth (less resistance);
# SPREAD_COEFF < 0.5 weakens the lateral-thinning term.
#
# Fitted values (see scripts/calibrate_surrogate.py, 8 parameter sets,
# 12 s depth-integrated reference runs):
#     跑距   平均绝对误差 15.1%
#     峰值流速 平均绝对误差 21.8%   (代理系统性偏高约 +20%)
DEPTH_FACTOR: float = 1.10
SPREAD_COEFF: float = 0.25


def _toe_depth(params: SimulationParams, h0: float) -> float:
    """Flow depth at the slope toe, after the release has elongated.

    This is the single most important correction over a naive model: a debris
    pulse **stretches** on its way down. A block released over
    ``release_length`` spreads out over most of the slope, so the depth at the
    toe is far smaller than ``h0`` -- and because both the yield resistance
    ``tau_y/(rho*h)`` and the Manning resistance ``g n^2 v^2 / h^(4/3)`` scale
    inversely with depth, ignoring this makes the predicted runout far too
    long. Volume conservation gives, with little lateral spreading on the
    slope::

        h_toe = V / (W * L_eff),   L_eff = release_length + slope_length
    """
    length_eff = params.release_length_m + params.slope_length_m
    width = max(params.channel_width_m, 0.5)
    h_toe = DEPTH_FACTOR * params.initial_volume_m3 / (length_eff * width)
    # Never let the estimate wander outside the physically sensible band.
    return float(min(max(h_toe, 0.05), h0))


def _integrate_slope(
    params: SimulationParams, h0: float, dt: float = 2.0e-3
) -> tuple[float, float]:
    """Integrate the centre-of-mass momentum balance down the slope.

    The depth thins linearly from ``h0`` at the crest to ``h_toe`` at the toe,
    so both the driving force and the depth-dependent resistance evolve the
    way they do in the full solver.

    Returns ``(v_toe, t_toe)``: velocity and travel time at the slope toe.
    """
    theta = params.slope_angle_rad
    rho = params.density_kg_m3
    mu_bed = params.basal_friction_coeff
    tau_y = params.yield_stress_pa
    n = params.manning_n

    h_toe = _toe_depth(params, h0)
    target = params.slope_length_m

    v = 0.0
    s = 0.0
    t = 0.0

    while s < target and t < 600.0:
        progress = min(1.0, s / max(target, 1.0e-6))
        h = max(h0 + (h_toe - h0) * progress, 0.05)

        manning = GRAVITY * n * n * v * v / (h ** (4.0 / 3.0))
        decel = mu_bed * GRAVITY * math.cos(theta) + tau_y / (rho * h) + manning
        accel = GRAVITY * math.sin(theta) - decel

        v = max(0.0, v + accel * dt)
        s += v * dt
        t += dt

    return v, t


def _integrate_runout(
    params: SimulationParams, v0: float, h0: float, dt: float = 2.0e-3
) -> tuple[float, float]:
    """Decelerate on the horizontal runout plane.

    Starts from the elongated toe depth, which then thins further as the flow
    spreads laterally.

    Returns ``(runout_distance, stopping_time)`` measured from the toe.
    """
    rho = params.density_kg_m3
    mu_bed = params.basal_friction_coeff
    tau_y = params.yield_stress_pa
    n = params.manning_n

    v = v0
    s = 0.0
    t = 0.0
    h_start = max(_toe_depth(params, h0), 0.05)

    while v > 1.0e-3 and t < 600.0:
        spreading = 1.0 + SPREAD_COEFF * s / max(params.channel_width_m, 1.0)
        h_eff = max(h_start / spreading, 0.03)
        manning = GRAVITY * n * n * v * v / (h_eff ** (4.0 / 3.0))
        decel = mu_bed * GRAVITY + tau_y / (rho * h_eff) + manning
        v = max(0.0, v - decel * dt)
        s += v * dt
        t += dt

    return s, t


def predict(params: SimulationParams) -> SurrogatePrediction:
    """Fast physics-based prediction for the given parameters."""
    terrain = build_terrain(params)
    h0 = terrain.release_thickness(params)

    v_toe, t_toe = _integrate_slope(params, h0)
    runout, t_runout = _integrate_runout(params, v_toe, h0)

    # --- arrival time at the monitoring section ---------------------------
    barrier = params.barrier_distance_m
    if runout >= barrier and v_toe > 1.0e-6:
        # Linear deceleration phase: solve  v0*t - a*t^2/2 = barrier
        a_eff = v_toe * v_toe / (2.0 * max(runout, 1.0e-6))
        discriminant = max(v_toe * v_toe - 2.0 * a_eff * barrier, 0.0)
        t_barrier = t_toe + (v_toe - math.sqrt(discriminant)) / max(a_eff, 1.0e-9)
    else:
        t_barrier = t_toe + t_runout

    # --- depth: the flow spreads laterally as it runs out -----------------
    horizontal_reach = terrain.toe_x_m + runout
    width_final = params.channel_width_m + 2.0 * runout * math.tan(math.radians(18.0))
    area = max(horizontal_reach * width_final, 1.0e-6)
    mean_depth = params.initial_volume_m3 / area
    max_depth = max(mean_depth * 2.0, h0 * 0.35)

    # --- impact: dynamic + hydrostatic pressure (same as the solver) ------
    rho = params.density_kg_m3
    v_impact = v_toe if runout >= barrier else 0.0
    h_impact = min(max_depth, h0)
    peak_force = (
        rho * h_impact * v_impact * v_impact + 0.5 * rho * GRAVITY * h_impact * h_impact
    ) * params.channel_width_m

    # --- confidence: the surrogate degrades in strongly 3-D regimes -------
    confidence = 0.82
    if params.slope_angle_deg > 50.0:
        confidence -= 0.12
    if params.slope_angle_deg < 5.0:
        confidence -= 0.18
    if params.volume_concentration > 0.65:
        confidence -= 0.06
    if runout < 1.0:
        confidence -= 0.15
    confidence = float(min(max(confidence, 0.05), 0.95))

    return SurrogatePrediction(
        runout_distance_m=round(runout, 3),
        max_velocity_m_s=round(v_toe, 3),
        arrival_time_s=round(t_barrier, 3),
        max_flow_depth_m=round(max_depth, 3),
        peak_impact_force_kn=round(peak_force / 1000.0, 3),
        affected_area_m2=round(area, 3),
        confidence=confidence,
    )
