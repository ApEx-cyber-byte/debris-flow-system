"""代理模型精度校验 / Surrogate accuracy validation.

对若干组参数同时运行**完整数值模拟**与 **AI 代理预测**，对比跑距、峰值流速
与峰值冲击力，输出相对误差。用于给文档提供**诚实的**精度数字，而不是
只声称「毫秒级、量级正确」。

用法::

    cd backend
    python scripts/validate_surrogate.py                 # 6 组参数，默认时长
    python scripts/validate_surrogate.py --duration 8    # 更快
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

from app.ai.physics_surrogate import predict  # noqa: E402
from app.physics.engine import SimulationEngine  # noqa: E402
from app.schemas import SimulationParams, SolverBackend  # noqa: E402

CASES: list[tuple[str, dict]] = [
    ("默认（30°/20m/150m³）", {}),
    ("缓坡 15°", {"slope_angle_deg": 15.0}),
    ("陡坡 45°", {"slope_angle_deg": 45.0}),
    ("高屈服应力 2000 Pa", {"yield_stress_pa": 2000.0}),
    ("低屈服应力 100 Pa", {"yield_stress_pa": 100.0}),
    ("粗糙床面 n=0.08", {"manning_n": 0.08}),
    ("大体积 400 m³", {"initial_volume_m3": 400.0}),
    ("长坡 40 m", {"slope_length_m": 40.0}),
]


def run_simulation(params: SimulationParams) -> dict:
    engine = SimulationEngine(params)
    return engine.run(on_frame=lambda _frame: None)


def relative_error(predicted: float, actual: float) -> float:
    if abs(actual) < 1.0e-9:
        return float("nan")
    return (predicted - actual) / actual * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 AI 代理模型精度")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--particles", type=int, default=1500)
    args = parser.parse_args()

    print(
        f"代理模型 vs 完整数值模拟（浅水方程后端，时长 {args.duration:.0f} s，"
        f"{args.particles} 粒子）\n"
    )
    header = (
        f"{'工况':<24}{'跑距(m) 模拟/代理':>26}{'误差':>9}"
        f"{'峰速(m/s) 模拟/代理':>28}{'误差':>9}"
    )
    print(header)
    print("-" * len(header))

    runout_errors: list[float] = []
    speed_errors: list[float] = []

    for name, overrides in CASES:
        params = SimulationParams(
            **{**overrides, "duration_s": args.duration,
               "particle_count": args.particles,
               "backend": SolverBackend.SHALLOW_WATER}
        )
        started = time.perf_counter()
        metrics = run_simulation(params)
        wall = time.perf_counter() - started

        started = time.perf_counter()
        surrogate = predict(params)
        surrogate_wall = time.perf_counter() - started

        sim_runout = metrics["runout_distance_m"]
        sim_speed = metrics["max_speed"]
        e_runout = relative_error(surrogate.runout_distance_m, sim_runout)
        e_speed = relative_error(surrogate.max_velocity_m_s, sim_speed)
        runout_errors.append(e_runout)
        speed_errors.append(e_speed)

        print(
            f"{name:<24}"
            f"{sim_runout:>12.2f} /{surrogate.runout_distance_m:>12.2f}"
            f"{e_runout:>8.1f}%"
            f"{sim_speed:>14.2f} /{surrogate.max_velocity_m_s:>12.2f}"
            f"{e_speed:>8.1f}%"
            f"    (模拟 {wall:.1f}s / 代理 {surrogate_wall * 1000:.1f}ms)"
        )

    def summarise(label: str, errors: list[float]) -> None:
        finite = [e for e in errors if e == e]
        mae = statistics.fmean(abs(e) for e in finite)
        bias = statistics.fmean(finite)
        print(f"  {label}: 平均绝对误差 {mae:.1f}%  平均偏差 {bias:+.1f}%")

    print("\n汇总（n = %d）" % len(CASES))
    summarise("跑距", runout_errors)
    summarise("峰值流速", speed_errors)

    print(
        "\n说明：代理模型是用与完整求解器相同的本构与阻力公式做的**一维质点**\n"
        "      能量/动量积分，秒级以内返回，用于参数拖动时的实时预判；\n"
        "      它不解析二维扩散，因此跑距存在系统性偏差，最终结论仍应以数值模拟为准。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
