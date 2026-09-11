"""非牛顿本构模型 / Non-Newtonian constitutive relations for debris flow.

泥石流是典型的非牛顿流体：低剪切速率下表现出屈服应力与高表观粘度，
高剪切速率下剪切稀化。本模块提供四种本构关系，全部以「表观粘度」
``mu_app(gamma_dot)`` 的形式暴露，供 SPH 动量方程使用。

Implemented models
------------------
* ``Bingham``           -- 宾汉体，用 Papanastasiou 指数正则化消除 tau_y/gamma_dot 奇异
* ``Cross``             -- Cross 四参数模型，覆盖全剪切速率区间
* ``Herschel-Bulkley``  -- 屈服 + 幂律，剪切稀化/增稠均可
* ``PowerLaw``          -- 纯幂律（无屈服应力）

References
----------
* Bingham (1922); Papanastasiou, T.C. (1987) *Flows of materials with yield*,
  J. Rheol. 31(5), 385-404.  -- 正则化方法
* Cross, M.M. (1965) *Rheology of non-Newtonian fluids*, J. Colloid Sci. 20.
* Herschel & Bulkley (1926).
* TITAN2D 系列深度积分泥石流模型中的 Voellmy-Coulomb 基底阻力公式参考：
  https://github.com/TITAN2D/titan2d (GPL-3.0) —— 仅公式参考，未复制任何代码
"""
from __future__ import annotations

import numpy as np

from ..config import GRAVITY, SOLID_DENSITY_DEFAULT, WATER_DENSITY
from ..schemas import RheologyModel, SimulationParams

# Regularisation floor for the shear rate, in 1/s. Keeps mu_app finite as
# gamma_dot -> 0 without altering behaviour at engineering shear rates.
GAMMA_DOT_FLOOR: float = 1.0e-3

# Papanastasiou stress-growth exponent (s). Larger -> sharper yield surface,
# smaller -> smoother / more stable. With GAMMA_DOT_FLOOR = 1e-3 the exponential
# still evaluates as exp(-1), so there is no underflow.
PAPANASTASIOU_M: float = 1000.0


def mixture_density(
    volume_concentration: float,
    solid_density: float = SOLID_DENSITY_DEFAULT,
    fluid_density: float = WATER_DENSITY,
) -> float:
    """体积加权混合密度 ``rho = Cv*rho_s + (1-Cv)*rho_f`` (kg/m^3)."""
    cv = float(np.clip(volume_concentration, 0.0, 1.0))
    return cv * solid_density + (1.0 - cv) * fluid_density


def apparent_viscosity(
    gamma_dot: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """表观粘度 ``mu_app`` (Pa*s) —— 本构模型的核心。

    Parameters
    ----------
    gamma_dot : ndarray
        剪切速率幅值 ``|gamma_dot| = sqrt(2 * D:D)`` (1/s)，逐粒子。
    params : SimulationParams
        含屈服应力、粘性系数与本构选型。

    Returns
    -------
    ndarray
        与 ``gamma_dot`` 同形状的表观粘度，已施加下限正则化与上限截断，
        保证 SPH 时间积分的数值稳定性。
    """
    g = np.abs(np.asarray(gamma_dot, dtype=np.float64))
    g_reg = np.maximum(g, GAMMA_DOT_FLOOR)

    mu_p = float(params.viscosity_pa_s)
    tau_y = float(params.yield_stress_pa)
    model = params.rheology

    if model is RheologyModel.BINGHAM:
        # tau = tau_y*(1 - exp(-m*gamma_dot)) + mu_p*gamma_dot
        # =>  mu_app = mu_p + tau_y*(1 - exp(-m*gamma_dot)) / gamma_dot
        # gamma_dot -> 0 时极限为 mu_p + tau_y*m（由上限截断兜底）
        yield_term = tau_y * (1.0 - np.exp(-PAPANASTASIOU_M * g_reg)) / g_reg
        mu_app = mu_p + yield_term

    elif model is RheologyModel.CROSS:
        # mu_app = mu_inf + (mu_0 - mu_inf) / (1 + (lambda*gamma_dot)^n)
        mu_inf = float(params.cross_mu_inf_pa_s)
        mu_0 = mu_p  # 低剪切极限粘度取用户给定粘性系数
        lam = float(params.cross_lambda_s)
        n = float(params.power_law_index)
        mu_app = mu_inf + (mu_0 - mu_inf) / (1.0 + np.power(lam * g_reg, n))

    elif model is RheologyModel.HERSCHEL_BULKLEY:
        # tau = tau_y + K*gamma_dot^n  =>  mu_app = tau_y/gamma_dot + K*gamma_dot^(n-1)
        n = float(params.power_law_index)
        k = float(params.viscosity_pa_s)
        mu_app = tau_y / g_reg + k * np.power(g_reg, n - 1.0)

    elif model is RheologyModel.POWER_LAW:
        n = float(params.power_law_index)
        k = float(params.viscosity_pa_s)
        mu_app = k * np.power(g_reg, n - 1.0)

    else:  # pragma: no cover - exhaustive enum, defensive only
        raise ValueError(f"unsupported rheology model: {model!r}")

    mu_max = float(params.max_apparent_viscosity_pa_s)
    return np.clip(mu_app, 1.0e-6, mu_max)


def basal_shear_stress(
    depth: np.ndarray,
    speed: np.ndarray,
    cos_slope: np.ndarray | float,
    params: SimulationParams,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """基底剪应力 ``tau_b`` (Pa) —— Voellmy-Coulomb + Manning 组合。

    ``tau_b = tau_y + mu_bed*rho*g*h*cos(beta) + rho*g*n^2*|v|^2/h^(1/3)``

    第一项是屈服应力（Bingham 塞流最终停止的物理来源），第二项是库仑摩擦，
    第三项是 Manning 湍流阻力。除以 ``rho*h`` 即得基底减速加速度。

    Parameters
    ----------
    depth : ndarray
        局部泥深 h (m)，逐粒子。
    speed : ndarray
        切向速度幅值 |v_t| (m/s)。
    cos_slope : ndarray | float
        局地床面法向的竖直分量 ``n_z = cos(beta)``。在平整坡面上等于 cos(theta)，
        在坡脚过渡段自动连续变化。
    """
    h = np.maximum(np.asarray(depth, dtype=np.float64), 1.0e-3)
    v = np.abs(np.asarray(speed, dtype=np.float64))
    rho = float(params.density_kg_m3)

    yield_term = float(params.yield_stress_pa)
    coulomb = float(params.basal_friction_coeff) * rho * gravity * h * np.asarray(cos_slope)
    manning = rho * gravity * (float(params.manning_n) ** 2) * v * v / np.power(h, 1.0 / 3.0)
    turbulent = float(params.turbulence_coeff) * rho * v * v

    return yield_term + coulomb + manning + turbulent


def describe(params: SimulationParams) -> dict[str, object]:
    """Human-readable summary of the active constitutive law (served to the UI)."""
    probe = np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0])
    mu = apparent_viscosity(probe, params)
    return {
        "model": params.rheology.value,
        "model_label": {
            RheologyModel.BINGHAM: "宾汉体 Bingham (Papanastasiou 正则化)",
            RheologyModel.CROSS: "Cross 四参数模型",
            RheologyModel.HERSCHEL_BULKLEY: "Herschel-Bulkley 屈服幂律",
            RheologyModel.POWER_LAW: "幂律 Power-Law",
        }[params.rheology],
        "yield_stress_pa": params.yield_stress_pa,
        "viscosity_pa_s": params.viscosity_pa_s,
        "mixture_density_kg_m3": params.density_kg_m3,
        "derived_mixture_density_kg_m3": mixture_density(params.volume_concentration),
        "shear_rate_probe": probe.tolist(),
        "apparent_viscosity_probe": mu.tolist(),
    }
