"""代理模型训练脚本 / Surrogate-model training entry point.

用法::

    # 1) 先用数值求解器批量生成样本（这一步很慢，建议后台跑）
    python scripts/generate_dataset.py --samples 400 --out data/dataset.npz

    # 2) 训练 CNN-LSTM 或 DR-LSTM
    python scripts/train_surrogate.py --data data/dataset.npz --model cnn-lstm-v1

    # 3) 权重会自动写入 backend/weights/，
    #    后端重启后 /api/ai/models 即会把它标为 available

本脚本**不是**第一版的运行必需项：没有权重时，
``app.ai.predictor`` 会自动使用物理启发式代理，接口行为完全一致。

参考项目
--------
* https://github.com/lizhixu123/DL-for-landslide-propagation-early-warning
  （CNN-LSTM / DR-LSTM 用于滑坡—泥石流传播预警）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.console import use_utf8_console  # noqa: E402

use_utf8_console()

from app.ai.models import (  # noqa: E402
    FEATURE_NAMES,
    TARGET_NAMES,
    build_cnn_lstm,
    build_dr_lstm,
)

WEIGHTS_DIR = ROOT / "weights"

TARGET_SCALES = np.array(
    [
        50.0,   # runout_distance_m
        15.0,   # max_velocity_m_s
        20.0,   # arrival_time_s
        5.0,    # max_flow_depth_m
        2000.0,  # peak_impact_force_kn
        20000.0,  # affected_area_m2
    ],
    dtype=np.float32,
)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise SystemExit(
            f"特征维度不匹配：数据集 {features.shape[1]}，当前定义 {len(FEATURE_NAMES)}"
        )
    if targets.shape[1] != len(TARGET_NAMES):
        raise SystemExit(
            f"目标维度不匹配：数据集 {targets.shape[1]}，当前定义 {len(TARGET_NAMES)}"
        )
    return features, targets


def standardise(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_mean, f_std = features.mean(axis=0), features.std(axis=0) + 1e-8
    t_mean, t_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8
    return (
        (features - f_mean) / f_std,
        (targets - t_mean) / t_std,
        f_mean,
        f_std,
    ), (t_mean, t_std)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练泥石流代理模型")
    parser.add_argument("--data", type=Path, required=True, help="dataset.npz 路径")
    parser.add_argument(
        "--model",
        choices=["cnn-lstm-v1", "dr-lstm-v1"],
        default="cnn-lstm-v1",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(
            "需要 PyTorch 才能训练：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    features, targets = load_dataset(args.data)
    print(f"数据集：{features.shape[0]} 个样本，{features.shape[1]} 个特征")

    (f_scaled, t_scaled, _f_mean, _f_std), (t_mean, t_std) = standardise(features, targets)

    n_val = max(1, int(len(f_scaled) * args.val_split))
    permutation = np.random.permutation(len(f_scaled))
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = build_cnn_lstm() if args.model == "cnn-lstm-v1" else build_dr_lstm()
    if network is None:
        print("网络构建失败（PyTorch 不可用）", file=sys.stderr)
        return 1
    network = network.to(device)

    train_x = torch.from_numpy(f_scaled[train_idx]).to(device)
    train_y = torch.from_numpy(t_scaled[train_idx]).to(device)
    val_x = torch.from_numpy(f_scaled[val_idx]).to(device)
    val_y = torch.from_numpy(t_scaled[val_idx]).to(device)

    optimiser = torch.optim.Adam(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=25, factor=0.5)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = WEIGHTS_DIR / ("cnn_lstm_v1.pt" if args.model == "cnn-lstm-v1" else "dr_lstm_v1.pt")

    for epoch in range(1, args.epochs + 1):
        network.train()
        indices = torch.randperm(len(train_x), device=device)
        epoch_loss = 0.0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            optimiser.zero_grad()
            loss = criterion(network(train_x[batch]), train_y[batch])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss) * len(batch)
        epoch_loss /= max(len(indices), 1)

        network.eval()
        with torch.no_grad():
            val_loss = float(criterion(network(val_x), val_y))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(network.state_dict(), checkpoint)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d}  train={epoch_loss:.5f}  val={val_loss:.5f}"
                f"  best={best_val:.5f}"
            )

    # Persist the normalisation constants next to the weights so serving
    # cannot drift from training.
    np.savez(
        WEIGHTS_DIR / checkpoint.stem.replace("_v1", "_norm"),
        feature_mean=_f_mean if "_f_mean" in dir() else features.mean(axis=0),
        target_mean=t_mean,
        target_std=t_std,
    )

    print(f"\n训练完成，最佳验证损失 {best_val:.5f}")
    print(f"权重已写入 {checkpoint}")
    print("重启后端后 /api/ai/models 会将其标记为 available。")
    _ = TARGET_SCALES
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
