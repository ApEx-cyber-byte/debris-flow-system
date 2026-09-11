"""数据导出路由 / CSV & JSON export.

支持四种导出对象：
* ``time_series``  —— 时间序列（流速 / 流深 / 冲击力 / 动能 / 堆积量）
* ``particles``    —— 指定关键帧的粒子位置与速度快照
* ``depth_profile``—— 沿程流深剖面
* ``full``         —— 完整 JSON 记录（参数 + 全部时间序列 + 诊断指标）
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..schemas import ExportFormat, ExportKind
from ..services.task_manager import task_manager

router = APIRouter(prefix="/api/simulations", tags=["export"])


def _engine_or_404(run_id: str):
    record = task_manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"仿真任务 {run_id} 不存在")
    if record.engine is None:
        raise HTTPException(status_code=409, detail="任务尚未初始化求解器")
    return record, record.engine


def _attachment(filename: str) -> dict[str, str]:
    # RFC 5987 encoding keeps Chinese filenames intact across browsers.
    from urllib.parse import quote

    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}


@router.get("/{run_id}/export", summary="导出仿真数据 (CSV / JSON)")
def export_run(
    run_id: str,
    kind: ExportKind = Query(ExportKind.TIME_SERIES, description="导出对象"),
    fmt: ExportFormat = Query(ExportFormat.CSV, description="文件格式"),
    frame_index: int | None = Query(None, ge=0, description="粒子快照的关键帧索引"),
) -> Response:
    record, engine = _engine_or_404(run_id)

    if kind is ExportKind.TIME_SERIES:
        rows = engine.time_series
        if not rows:
            raise HTTPException(status_code=409, detail="尚无可用数据，请等待仿真产生关键帧")
        if fmt is ExportFormat.CSV:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            return Response(
                content=buffer.getvalue().encode("utf-8-sig"),
                media_type="text/csv; charset=utf-8",
                headers=_attachment(f"debris_flow_{run_id}_time_series.csv"),
            )
        return Response(
            content=json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
            media_type="application/json",
            headers=_attachment(f"debris_flow_{run_id}_time_series.json"),
        )

    if kind is ExportKind.DEPTH_PROFILE:
        profile = engine.depth_profile()
        if not profile:
            raise HTTPException(status_code=409, detail="尚无可用剖面数据")
        if fmt is ExportFormat.CSV:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=["x", "depth"])
            writer.writeheader()
            writer.writerows(profile)
            return Response(
                content=buffer.getvalue().encode("utf-8-sig"),
                media_type="text/csv; charset=utf-8",
                headers=_attachment(f"debris_flow_{run_id}_depth_profile.csv"),
            )
        return Response(
            content=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
            media_type="application/json",
            headers=_attachment(f"debris_flow_{run_id}_depth_profile.json"),
        )

    if kind is ExportKind.PARTICLES:
        if not engine.frames:
            raise HTTPException(status_code=409, detail="尚无可用粒子数据")
        index = len(engine.frames) - 1 if frame_index is None else frame_index
        if index >= len(engine.frames):
            raise HTTPException(
                status_code=416,
                detail=f"关键帧索引 {index} 超出范围 (共 {len(engine.frames)} 帧)",
            )
        frame = engine.frames[index]
        pos = np.asarray(frame.particle_positions, dtype=np.float64)
        vel = np.asarray(frame.particle_velocities, dtype=np.float64)
        speed = np.linalg.norm(vel, axis=1)

        if fmt is ExportFormat.CSV:
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["id", "x", "y", "z", "vx", "vy", "vz", "speed"])
            for i in range(pos.shape[0]):
                writer.writerow(
                    [i, *[f"{v:.6f}" for v in pos[i]], *[f"{v:.6f}" for v in vel[i]], f"{speed[i]:.6f}"]
                )
            return Response(
                content=buffer.getvalue().encode("utf-8-sig"),
                media_type="text/csv; charset=utf-8",
                headers=_attachment(f"debris_flow_{run_id}_particles_frame{index}.csv"),
            )

        payload = {
            "run_id": run_id,
            "frame_index": index,
            "time_s": frame.time,
            "step": frame.step,
            "particle_stride": frame.particle_stride,
            "n_particles": int(pos.shape[0]),
            "particles": [
                {
                    "x": float(pos[i, 0]), "y": float(pos[i, 1]), "z": float(pos[i, 2]),
                    "vx": float(vel[i, 0]), "vy": float(vel[i, 1]), "vz": float(vel[i, 2]),
                    "speed": float(speed[i]),
                }
                for i in range(pos.shape[0])
            ],
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            media_type="application/json",
            headers=_attachment(f"debris_flow_{run_id}_particles_frame{index}.json"),
        )

    # FULL
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": record.status.value,
        "params": record.params.model_dump(mode="json"),
        "backend": (record.backend.model_dump(mode="json") if record.backend else None),
        "terrain": engine.terrain.to_info().model_dump(mode="json"),
        "grid": engine.solver.grid.model_dump(mode="json"),
        "metrics": record.metrics,
        "time_series": engine.time_series,
        "depth_profile": engine.depth_profile(),
        "n_frames_retained": len(engine.frames),
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        media_type="application/json",
        headers=_attachment(f"debris_flow_{run_id}_full.json"),
    )
