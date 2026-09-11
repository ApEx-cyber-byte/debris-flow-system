"""WebSocket 关键帧推送 / keyframe streaming endpoint.

协议
----
服务端 → 客户端：

* **二进制帧** ``meta``   —— 握手：参数、地形、栅格、后端、初始床面高程
* **二进制帧** ``frame``  —— 关键帧：粒子位置/速度 + 流深栅格 + 流速栅格 + 诊断量
* **文本消息** ``completed`` / ``error`` —— 结束事件

客户端 → 服务端（文本 JSON）：

* ``{"type": "stride", "value": 3}`` —— 改变粒子降采样步长（减轻前端压力）
* ``{"type": "pause"}`` / ``{"type": "resume"}`` / ``{"type": "stop"}``
* ``{"type": "ping"}`` —— 心跳

队列有界：客户端消费不过来时丢弃最旧的帧，而不是拖慢物理求解。
"""
from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..schemas import RunStatus
from ..services.frames import encode_json_message
from ..services.task_manager import task_manager

router = APIRouter(tags=["stream"])

TERMINAL_STATES = {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.CANCELLED}


@router.websocket("/ws/simulations/{run_id}")
async def stream_simulation(websocket: WebSocket, run_id: str) -> None:
    record = task_manager.get(run_id)
    if record is None:
        await websocket.close(code=4404, reason=f"仿真任务 {run_id} 不存在")
        return

    await websocket.accept()
    queue = record.bus.subscribe()

    async def pump_client_messages() -> None:
        """Apply client control messages until the socket closes."""
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = message.get("type")
            if kind == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif kind == "pause":
                task_manager.pause(run_id)
            elif kind == "resume":
                task_manager.resume(run_id)
            elif kind == "stop":
                task_manager.stop(run_id)

    consumer = asyncio.create_task(pump_client_messages())
    try:
        # Replay the handshake and the latest frame so a client that connects
        # mid-run (or reconnects) paints a complete scene immediately instead
        # of waiting for the next keyframe — or getting nothing at all if the
        # run already finished.
        if record.bus.meta is not None:
            await websocket.send_bytes(record.bus.meta)

        if record.bus.latest is not None:
            frame = record.bus.latest
            grid = record.engine.solver.grid if record.engine else None
            await websocket.send_bytes(encode_frame_bytes(run_id, frame, grid))

        # A run that already ended still deserves a clean, immediate answer.
        if record.status in TERMINAL_STATES:
            await websocket.send_text(
                encode_json_message(
                    {
                        "type": "completed",
                        "run_id": run_id,
                        "status": record.status.value,
                        "metrics": record.metrics,
                        "n_frames": record.frames_seen,
                    }
                )
            )
            return

        while True:
            message = await queue.get()
            if message is None:  # end-of-stream sentinel
                break
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        record.bus.unsubscribe(queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()


def encode_frame_bytes(run_id: str, frame, grid) -> bytes:
    """Re-encode a retained keyframe for replay to a late-joining client."""
    from ..services.frames import encode_message

    return encode_message(
        {
            "type": "frame",
            "run_id": run_id,
            "index": frame.index,
            "step": frame.step,
            "time": frame.time,
            "dt": frame.dt,
            "n_particles_total": frame.n_particles_total,
            "particle_stride": frame.particle_stride,
            "replay": True,
            "grid": (grid.model_dump(mode="json") if grid is not None else None),
            "stats": {k: float(v) for k, v in frame.stats.items()},
        },
        frame.to_arrays(),
    )
