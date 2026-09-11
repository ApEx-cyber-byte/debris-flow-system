"""代理模型系数标定 / Calibrate the surrogate's structural coefficients.

一维质点模型无法表达流动的垂向结构与「前锋快于质心」这一事实，
因此用两个**透明的**结构系数吸收这部分系统性偏差：

* ``DEPTH_FACTOR``   —— 堆积区有效泥深放大系数
* ``SPREAD_COEFF``   —— 横向展宽减薄速率

做法：先用完整数值模拟缓存 8 组工况的参考解（**这一步很慢**，结果缓存到
``data/calibration_reference.json``），再对系数做网格搜索——因为代理模型
单次求解只要约 1 ms，几万组搜索是秒级的。

用法::

    cd backend
    python scripts/calibrate_surrogate.py            # 用缓存（没有则先跑模拟）
    python scripts/calibrate_surrogate.py --refresh  # 强制重跑模拟

标定完成后，把输出的两个系数写回 ``app/ai/physics_surrogate.py``。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

from app.ai import physics_surrogate as surrogate  # noqa: E402
from app.physics.engine import SimulationEngine  # noqa: E402
from app.schemas import SimulationParams, SolverBackend  # noqa: E402
from scripts.validate_surrogate import CASES  # noqa: E402

CACHE = ROOT / "data" / "calibration_reference.json"


def build_reference(duration: float, particles: int, refresh: bool) -> list[dict]:
    if CACHE.exists() and not refresh:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"使用缓存参考解 {CACHE}（{len(cached)} 组工况）")
        return cached

    print(f"运行 {len(CASES)} 组完整数值模拟（时长 {duration:.0f} s）...")
    reference: list[dict] = []
    for name, overrides in CASES:
        params = SimulationParams(
            **{**overrides, "duration_s": duration,
               "particle_count": particles,
               "backend": SolverBackend.SHALLOW_WATER}
        )
        started = time.perf_counter()
        metrics = SimulationEngine(params).run(on_frame=lambda _f: None)
        wall = time.perf_counter() - started
        record = {
            "name": name,
            "params": params.model_dump(mode="json"),
            "runout_distance_m": metrics["runout_distance_m"],
            "max_speed": metrics["max_speed"],
            "max_depth": metrics["max_depth"],
            "max_impact_force_kn": metrics["max_impact_force_kn"],
        }
        reference.append(record)
        print(
            f"  {name:<24} 跑距 {record['runout_distance_m']:6.2f} m  "
            f"峰速 {record['max_speed']:5.2f} m/s   ({wall:.1f}s)"
        )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"参考解已缓存到 {CACHE}")
    return reference


def score(reference: list[dict], depth_factor: float, spread_coeff: float) -> tuple[float, float, float]:
    surrogate.DEPTH_FACTOR = depth_factor
    surrogate.SPREAD_COEFF = spread_coeff

    runout_errors: list[float] = []
    speed_errors: list[float] = []
    for record in reference:
        params = SimulationParams(**record["params"])
        prediction = surrogate.predict(params)
        runout_errors.append(
            (prediction.runout_distance_m - record["runout_distance_m"])
            / max(record["runout_distance_m"], 1e-9)
        )
        speed_errors.append(
            (prediction.max_velocity_m_s - record["max_speed"])
            / max(record["max_speed"], 1e-9)
        )

    mae_runout = statistics.fmean(abs(e) for e in runout_errors)
    mae_speed = statistics.fmean(abs(e) for e in speed_errors)
    # Runout dominates a hazard assessment, so it carries the larger weight;
    # the velocity term also penalises a systematic bias, not just spread.
    combined = 0.65 * mae_runout + 0.35 * mae_speed
    return combined, mae_runout, mae_speed


def main() -> int:
    parser = argparse.ArgumentParser(description="标定代理模型系数")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--particles", type=int, default=1500)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    reference = build_reference(args.duration, args.particles, args.refresh)

    print("\n网格搜索 DEPTH_FACTOR × SPREAD_COEFF ...")
    best: tuple[float, float, float, float, float] | None = None
    for depth_factor in [0.8 + 0.1 * i for i in range(21)]:          # 0.8 .. 2.8
        for spread_coeff in [0.05 + 0.05 * i for i in range(21)]:     # 0.05 .. 1.05
            combined, mae_runout, mae_speed = score(reference, depth_factor, spread_coeff)
            if best is None or combined < best[0]:
                best = (combined, depth_factor, spread_coeff, mae_runout, mae_speed)

    assert best is not None
    _combined, depth_factor, spread_coeff, mae_runout, mae_speed = best
    print(
        f"\n最优: DEPTH_FACTOR = {depth_factor:.2f}   "
        f"SPREAD_COEFF = {spread_coeff:.2f}"
    )
    print(f"  跑距平均绝对误差 {mae_runout * 100:.1f}%")
    print(f"  峰值流速平均绝对误差 {mae_speed * 100:.1f}%")

    # Per-case detail at the optimum, so the residual bias stays visible.
    score(reference, depth_factor, spread_coeff)
    print("\n  逐工况残差:")
    for record in reference:
        params = SimulationParams(**record["params"])
        prediction = surrogate.predict(params)
        print(
            f"    {record['name']:<24}"
            f"跑距 {record['runout_distance_m']:6.2f} -> "
            f"{prediction.runout_distance_m:6.2f} m   "
            f"峰速 {record['max_speed']:5.2f} -> {prediction.max_velocity_m_s:5.2f} m/s"
        )

    print(
        f"\n请把 app/ai/physics_surrogate.py 中的两个常量更新为:\n"
        f"    DEPTH_FACTOR: float = {depth_factor:.2f}\n"
        f"    SPREAD_COEFF: float = {spread_coeff:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
