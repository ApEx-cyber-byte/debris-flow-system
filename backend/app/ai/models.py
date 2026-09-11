"""PyTorch 代理模型定义 / PyTorch surrogate network definitions.

预留接口。两个网络按需求文档给定：

* ``CNNLSTMSurrogate``  —— CNN 提取坡面/地形空间特征，LSTM 捕捉时序演化，
  用于「参数 -> 时空响应场」的映射
* ``DRLSTMSurrogate``   —— Dual-Resolution LSTM：粗细双分辨率分支，
  粗分支给全局跑距、细分支给局部流深，兼顾精度与推理速度

参考项目：https://github.com/lizhixu123/DL-for-landslide-propagation-early-warning

.. note::
   这两个网络在**没有训练数据与权重文件**时不会被启用。
   :class:`app.ai.predictor.AIPredictor` 会检测权重是否存在，
   不存在时自动使用物理启发式代理（:mod:`app.ai.physics_surrogate`）。
   训练脚本见 ``backend/scripts/train_surrogate.py``。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# The input feature vector layout shared by both networks and by the training
# pipeline. Keeping it in one place prevents train/serve skew.
FEATURE_NAMES: tuple[str, ...] = (
    "slope_angle_deg",
    "slope_length_m",
    "manning_n",
    "volume_concentration",
    "yield_stress_pa",
    "viscosity_pa_s",
    "density_kg_m3",
    "initial_volume_m3",
    "release_thickness_m",
    "channel_width_m",
    "basal_friction_coeff",
    "rheology_code",
)

TARGET_NAMES: tuple[str, ...] = (
    "runout_distance_m",
    "max_velocity_m_s",
    "arrival_time_s",
    "max_flow_depth_m",
    "peak_impact_force_kn",
    "affected_area_m2",
)


def encode_features(params: Any) -> np.ndarray:
    """Turn :class:`SimulationParams` into the canonical feature vector."""
    from ..physics.terrain import build_terrain

    terrain = build_terrain(params)
    rheology_code = {
        "bingham": 0.0,
        "cross": 1.0,
        "herschel_bulkley": 2.0,
        "power_law": 3.0,
    }[params.rheology.value]

    values = [
        params.slope_angle_deg,
        params.slope_length_m,
        params.manning_n,
        params.volume_concentration,
        params.yield_stress_pa,
        params.viscosity_pa_s,
        params.density_kg_m3,
        params.initial_volume_m3,
        terrain.release_thickness(params),
        params.channel_width_m,
        params.basal_friction_coeff,
        rheology_code,
    ]
    return np.asarray(values, dtype=np.float32)


def _torch():
    """Import torch lazily so the backend runs fine without it installed."""
    try:
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        return torch, nn
    except ImportError:  # pragma: no cover - depends on the environment
        return None, None


def build_cnn_lstm(input_dim: int = len(FEATURE_NAMES), output_dim: int = len(TARGET_NAMES)):
    """Construct a CNN-LSTM surrogate, or ``None`` when torch is unavailable.

    The parameter vector is reshaped into a small 2-D "image" (a 1 x N map)
    so the convolutional encoder learns local interactions between adjacent
    physical parameters (e.g. slope angle with slope length), while the LSTM
    head models the temporal propagation of the front.
    """
    torch, nn = _torch()
    if torch is None or nn is None:
        return None

    class CNNLSTMSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_dim)
            )

        def forward(self, x):  # pragma: no cover - requires torch
            # x: (batch, input_dim)
            h = x.unsqueeze(1)                 # (batch, 1, input_dim)
            h = self.encoder(h)                # (batch, 32, 8)
            h = h.transpose(1, 2)              # (batch, 8, 32) -> sequence
            out, _ = self.lstm(h)
            return self.head(out[:, -1, :])

    return CNNLSTMSurrogate()


def build_dr_lstm(input_dim: int = len(FEATURE_NAMES), output_dim: int = len(TARGET_NAMES)):
    """Construct a dual-resolution LSTM surrogate, or ``None`` without torch."""
    torch, nn = _torch()
    if torch is None or nn is None:
        return None

    class DRLSTMSurrogate(nn.Module):
        """Coarse branch -> global runout; fine branch -> local depth field."""

        def __init__(self) -> None:
            super().__init__()
            self.coarse = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
            )
            self.fine = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
            )
            self.coarse_lstm = nn.LSTM(64, 48, num_layers=1, batch_first=True)
            self.fine_lstm = nn.LSTM(128, 48, num_layers=1, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, output_dim)
            )

        def forward(self, x):  # pragma: no cover - requires torch
            coarse = self.coarse(x).unsqueeze(1).repeat(1, 4, 1)
            fine = self.fine(x).unsqueeze(1).repeat(1, 16, 1)
            c_out, _ = self.coarse_lstm(coarse)
            f_out, _ = self.fine_lstm(fine)
            merged = torch.cat([c_out[:, -1, :], f_out[:, -1, :]], dim=-1)
            return self.fusion(merged)

    return DRLSTMSurrogate()
