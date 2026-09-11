"""AI 预测路由 / AI surrogate prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..ai.models import FEATURE_NAMES, TARGET_NAMES, encode_features
from ..ai.predictor import MODEL_REGISTRY, predictor
from ..schemas import AIPredictRequest, AIPredictResponse, SimulationParams

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/models", summary="可用代理模型列表")
def list_models() -> dict:
    return {
        "models": [
            {
                "name": name,
                "label": spec["label"],
                "kind": spec["kind"],
                "description": spec["description"],
                "available": name in predictor.available_models(),
            }
            for name, spec in MODEL_REGISTRY.items()
        ],
        "default": "heuristic-v1",
        "torch_available": predictor.torch_available,
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
    }


@router.post("/predict", response_model=AIPredictResponse, summary="快速预测泥石流运动特征")
def predict(request: AIPredictRequest) -> AIPredictResponse:
    try:
        return predictor.predict(request)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 500 with a clear message
        raise HTTPException(status_code=500, detail=f"AI 预测失败: {exc}") from exc


@router.post("/features", summary="调试：参数 -> 特征向量")
def features(params: SimulationParams) -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "features": encode_features(params).tolist(),
    }
