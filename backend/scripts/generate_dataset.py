"""样本生成脚本 / Training-data generator for the AI surrogate.

用内置的 CPU 求解器对参数空间做拉丁超立方采样，批量产生
「参数 -> 运动特征」样本，供 ``train_surrogate.py`` 训练 CNN-LSTM / DR-LSTM。

用法::

    python scripts/generate_dataset.py --samples 300 --out data/dataset.npz
    python scripts/generate_dataset.py --samples 20 --duration 4 --quick   # 冒烟测试

样本量与时长的取舍：每个样本都要跑一次完整的数值模拟，300 个样本
在 CPU 浅水方程求解器上大约需要 20–40 分钟。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

from app.ai.models import FEATURE_NAMES, TARGET_NAMES, encode_features  # noqa: E402
from app.physics.engine import SimulationEngine  # noqa: E402
from app.schemas import SimulationParams, SolverBackend  # noqa: E402


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified sampling: one jittered point per stratum in each dimension."""
    cut = np.linspace(0.0, 1.0, n + 1)
    samples = np.empty((n, dims))
    for d in range(dims):
        samples[:, d] = rng.uniform(cut[:-1], cut[1:])
    for d in range(dims):
        rng.shuffle(samples[:, d])
    return samples


def sample_params(unit: np.ndarray) -> SimulationParams:
    """Map a unit-hypercube point onto the physical parameter box."""
    u = unit
    return SimulationParams(
        slope_angle_deg=float(5.0 + u[0] * 50.0),
        slope_length_m=float(8.0 + u[1] * 42.0),
        manning_n=float(0.01 + u[2] * 0.09),
        volume_concentration=float(0.25 + u[3] * 0.4),
        yield_stress_pa=float(50.0 + u[4] * 3000.0),
        viscosity_pa_s=float(5.0 + u[5] * 250.0),
        density_kg_m3=float(1600.0 + u[6] * 900.0),
        initial_volume_m3=float(30.0 + u[7] * 500.0),
        channel_width_m=float(4.0 + u[8] * 12.0),
        release_length_m=float(4.0 + u[9] * 10.0),
        basal_friction_coeff=float(0.02 + u[10] * 0.35),
        rheology=["bingham", "cross", "herschel_bulkley", "power_law"][
            int(u[11] * 3.999)
        ],
        backend=SolverBackend.SHALLOW_WATER,
        particle_count=800,
        duration_s=12.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="生成代理模型训练样本")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "dataset.npz")
    parser.add_argument("--duration", type=float, default=None, help="覆盖模拟时长 (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="极小样本冒烟测试")
    args = parser.parse_args()

    if args.quick:
        args.samples = min(args.samples, 4)
        args.duration = args.duration or 4.0

    rng = np.random.default_rng(args.seed)
    units = latin_hypercube(args.samples, len(FEATURE_NAMES), rng)

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    failures = 0
    started = time.perf_counter()

    for index, unit in enumerate(units):
        try:
            params = sample_params(unit)
            if args.duration is not None:
                params = params.model_copy(update={"duration_s": args.duration})

            engine = SimulationEngine(params)

            def on_frame(_frame) -> None:  # noqa: ANN001 - keep memory flat
                pass

            metrics = engine.run(on_frame=on_frame)

            features.append(encode_features(params))
            targets.append(
                np.array(
                    [
                        metrics.get("runout_distance_m", 0.0),
                        metrics.get("max_speed", 0.0),
                        params.duration_s,
                        metrics.get("max_depth", 0.0),
                        metrics.get("max_impact_force_kn", 0.0),
                        # Affected area estimated from the runout footprint.
                        max(1.0, metrics.get("runout_distance_m", 0.0))
                        * params.channel_width_m,
                    ],
                    dtype=np.float32,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad sample must not stop the run
            failures += 1
            print(f"  样本 {index} 失败: {type(error).__name__}: {error}")

        if (index + 1) % 10 == 0 or index + 1 == args.samples:
            elapsed = time.perf_counter() - started
            rate = (index + 1) / max(elapsed, 1e-6)
            remaining = (args.samples - index - 1) / max(rate, 1e-6)
            print(
                f"  {index + 1}/{args.samples} 已完成  "
                f"用时 {elapsed:.0f}s  预计剩余 {remaining:.0f}s  失败 {failures}"
            )

    if not features:
        print("没有产生任何有效样本", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.stack(features),
        targets=np.stack(targets),
        feature_names=np.array(FEATURE_NAMES),
        target_names=np.array(TARGET_NAMES),
    )
    print(f"\n已写入 {args.out}（{len(features)} 个样本，失败 {failures} 个）")
    print("下一步: python scripts/train_surrogate.py --data " f"{args.out} --model cnn-lstm-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
