"""AI 预测门面 / AI prediction facade.

对外只暴露一个 ``AIPredictor``，内部按下面的顺序选模型：

1. 若存在已训练的 PyTorch 权重（``backend/weights/*.pt``）→ 用 CNN-LSTM / DR-LSTM
2. 否则 → 用物理启发式代理（毫秒级，量级正确）

返回结构统一为 :class:`~app.schemas.AIPredictResponse`，
因此前端不需要知道背后是学习型模型还是物理代理。
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..physics.backends import probe
from ..schemas import AIPredictRequest, AIPredictResponse, SimulationParams
from . import models as model_defs
from .physics_surrogate import SurrogatePrediction, predict as physics_predict

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "heuristic-v1": {
        "label": "物理启发式代理 (一维质点能量/动量平衡)",
        "kind": "physics",
        "description": "与完整求解器共用同一套非牛顿本构与基底阻力公式",
    },
    "cnn-lstm-v1": {
        "label": "CNN-LSTM 代理模型",
        "kind": "torch",
        "checkpoint": "cnn_lstm_v1.pt",
        "description": "CNN 提取参数空间特征, LSTM 建模前锋时序演化",
    },
    "dr-lstm-v1": {
        "label": "DR-LSTM 双分辨率代理模型",
        "kind": "torch",
        "checkpoint": "dr_lstm_v1.pt",
        "description": "粗分支给出全局跑距, 细分支给出局部流深",
    },
}


class AIPredictor:
    """Thread-safe facade over the available surrogate models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, object] = {}
        self._loaded: dict[str, bool] = {}

    # ------------------------------------------------------------------
    def available_models(self) -> list[str]:
        names = ["heuristic-v1"]
        for name, spec in MODEL_REGISTRY.items():
            if spec["kind"] != "torch":
                continue
            weights = WEIGHTS_DIR / str(spec["checkpoint"])
            if weights.exists():
                names.append(name)
        return names

    @property
    def torch_available(self) -> bool:
        return probe().torch

    # ------------------------------------------------------------------
    def _load_torch(self, name: str):  # pragma: no cover - needs weights + torch
        if name in self._cache:
            return self._cache[name]

        spec = MODEL_REGISTRY.get(name)
        if spec is None or spec["kind"] != "torch":
            return None

        torch, _ = model_defs._torch()  # noqa: SLF001 - intentional reuse
        if torch is None:
            return None

        weights = WEIGHTS_DIR / str(spec["checkpoint"])
        if not weights.exists():
            return None

        builder = (
            model_defs.build_cnn_lstm if name == "cnn-lstm-v1" else model_defs.build_dr_lstm
        )
        network = builder()
        if network is None:
            return None
        network.load_state_dict(torch.load(weights, map_location="cpu"))
        network.eval()
        self._cache[name] = network
        return network

    # ------------------------------------------------------------------
    def predict(self, request: AIPredictRequest) -> AIPredictResponse:
        params: SimulationParams = request.params
        name = request.model_name
        notes: list[str] = []

        if name not in MODEL_REGISTRY:
            notes.append(f"未知模型 {name!r}，已回退到 heuristic-v1")
            name = "heuristic-v1"

        network = self._load_torch(name) if MODEL_REGISTRY[name]["kind"] == "torch" else None
        trained = network is not None

        if network is not None:  # pragma: no cover - requires a trained checkpoint
            import torch  # noqa: PLC0415

            with self._lock, torch.no_grad():
                features = model_defs.encode_features(params)[None, :]
                output = network(torch.from_numpy(features)).numpy()[0]
            values = dict(zip(model_defs.TARGET_NAMES, output.tolist(), strict=False))
            prediction = SurrogatePrediction(
                runout_distance_m=float(values["runout_distance_m"]),
                max_velocity_m_s=float(values["max_velocity_m_s"]),
                arrival_time_s=float(values["arrival_time_s"]),
                max_flow_depth_m=float(values["max_flow_depth_m"]),
                peak_impact_force_kn=float(values["peak_impact_force_kn"]),
                affected_area_m2=float(values["affected_area_m2"]),
                confidence=0.9,
            )
            notes.append(f"已加载训练权重 {MODEL_REGISTRY[name]['checkpoint']}")
        else:
            prediction = physics_predict(params)
            if MODEL_REGISTRY[name]["kind"] == "torch":
                reason = (
                    "PyTorch 未安装" if not self.torch_available
                    else f"未找到权重文件 {MODEL_REGISTRY[name]['checkpoint']}"
                )
                notes.append(f"{name} 不可用（{reason}），已回退到物理启发式代理")
                name = "heuristic-v1"

        notes.append(
            "学习型代理模型接口已预留：运行 backend/scripts/train_surrogate.py "
            "批量生成数值模拟样本并训练，权重放入 backend/weights/ 后自动启用。"
        )

        return AIPredictResponse(
            model_name=name,
            available_models=self.available_models(),
            is_surrogate_trained=trained,
            runout_distance_m=prediction.runout_distance_m,
            max_velocity_m_s=prediction.max_velocity_m_s,
            arrival_time_s=prediction.arrival_time_s,
            max_flow_depth_m=prediction.max_flow_depth_m,
            peak_impact_force_kn=prediction.peak_impact_force_kn,
            affected_area_m2=prediction.affected_area_m2,
            confidence=prediction.confidence,
            notes=notes,
        )


predictor = AIPredictor()
