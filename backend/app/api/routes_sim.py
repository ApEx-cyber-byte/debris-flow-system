"""REST 路由：仿真任务 / Simulation run lifecycle."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..physics.backends import BackendUnavailable, environment_report
from ..physics.rheology import describe
from ..schemas import (
    BackendInfo,
    GridInfo,
    RunResult,
    RunStatus,
    SimulationCreated,
    SimulationParams,
    SimulationSummary,
    TerrainInfo,
    TimeSeriesPoint,
)
from ..services.task_manager import TaskManager, task_manager

router = APIRouter(prefix="/api", tags=["simulation"])


def _record_or_404(run_id: str, manager: TaskManager = task_manager):
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"仿真任务 {run_id} 不存在",
        )
    return record


# ---------------------------------------------------------------------------
# System introspection
# ---------------------------------------------------------------------------
@router.get("/system/backends", summary="求解后端与可选依赖探测结果")
def get_backends() -> dict:
    return environment_report()


@router.get("/system/defaults", response_model=SimulationParams, summary="默认参数")
def get_defaults() -> SimulationParams:
    return SimulationParams()


@router.post("/system/rheology", summary="非牛顿本构探针曲线")
def post_rheology(params: SimulationParams) -> dict:
    return describe(params)


# ---------------------------------------------------------------------------
# Simulation lifecycle
# ---------------------------------------------------------------------------
@router.post(
    "/simulations",
    response_model=SimulationCreated,
    status_code=status.HTTP_201_CREATED,
    summary="创建并启动一次仿真",
)
def create_simulation(params: SimulationParams) -> SimulationCreated:
    try:
        record = task_manager.create(params)
    except BackendUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    task_manager.start(record.run_id)

    engine = record.engine
    assert engine is not None  # guaranteed by TaskManager.create
    return SimulationCreated(
        run_id=record.run_id,
        status=record.status,
        ws_url=f"/ws/simulations/{record.run_id}",
        terrain=TerrainInfo.model_validate(engine.terrain.to_info()),
        grid=GridInfo.model_validate(engine.solver.grid),
        backend=BackendInfo.model_validate(record.backend),
    )


@router.get("/simulations", response_model=list[SimulationSummary], summary="任务列表")
def list_simulations() -> list[SimulationSummary]:
    return [record.summary() for record in task_manager.list()]


@router.get("/simulations/{run_id}", response_model=SimulationSummary, summary="任务状态")
def get_simulation(run_id: str) -> SimulationSummary:
    return _record_or_404(run_id).summary()


@router.post("/simulations/{run_id}/pause", response_model=SimulationSummary)
def pause_simulation(run_id: str) -> SimulationSummary:
    record = task_manager.pause(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"仿真任务 {run_id} 不存在")
    return record.summary()


@router.post("/simulations/{run_id}/resume", response_model=SimulationSummary)
def resume_simulation(run_id: str) -> SimulationSummary:
    record = task_manager.resume(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"仿真任务 {run_id} 不存在")
    return record.summary()


@router.post("/simulations/{run_id}/stop", response_model=SimulationSummary)
def stop_simulation(run_id: str) -> SimulationSummary:
    record = task_manager.stop(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"仿真任务 {run_id} 不存在")
    return record.summary()


@router.delete("/simulations/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_simulation(run_id: str) -> None:
    _record_or_404(run_id)
    task_manager.stop(run_id)


@router.get("/simulations/{run_id}/result", response_model=RunResult, summary="完整结果")
def get_result(run_id: str) -> RunResult:
    record = _record_or_404(run_id)
    engine = record.engine
    if engine is None:
        raise HTTPException(status_code=409, detail="任务尚未初始化求解器")

    return RunResult(
        run_id=record.run_id,
        status=record.status,
        params=record.params,
        backend=record.backend or BackendInfo(
            requested=record.params.backend.value, active="unknown", accelerated=False
        ),
        terrain=engine.terrain.to_info(),
        grid=engine.solver.grid,
        time_series=[TimeSeriesPoint(**point) for point in engine.time_series],
        metrics=dict(record.metrics),
        wall_time_s=float(record.metrics.get("wall_time_s", 0.0)),
    )


@router.get("/simulations/{run_id}/depth-profile", summary="流深沿程剖面")
def get_depth_profile(run_id: str) -> dict:
    record = _record_or_404(run_id)
    engine = record.engine
    if engine is None:
        raise HTTPException(status_code=409, detail="任务尚未初始化求解器")
    return {
        "run_id": run_id,
        "toe_x_m": engine.terrain.toe_x_m,
        "profile": engine.depth_profile(),
    }


@router.get("/simulations/{run_id}/status", response_model=RunStatus, summary="任务状态枚举")
def get_status(run_id: str) -> RunStatus:
    return _record_or_404(run_id).status
