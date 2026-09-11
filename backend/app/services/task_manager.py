"""任务调度与帧广播 / Run scheduling and frame fan-out.

职责
----
* 为每次仿真建立 :class:`RunRecord`（参数、状态、进度、指标体系）
* 在后台线程中用 :class:`SimulationEngine` 推进物理
* 通过 :class:`FrameBus` 把关键帧广播给所有订阅的 WebSocket 客户端
* 可选后端（PySPH / SynxFlow）**在运行时**失败时自动回退到 CPU 求解器

线程模型
--------
物理求解在 ``threading`` 工作线程中运行（NumPy 会释放 GIL，所以不会
阻塞事件循环）；帧通过 ``loop.call_soon_threadsafe`` 投递回 asyncio
事件循环，再由订阅者的队列消费。队列有界，慢客户端会丢旧帧而不是
拖慢求解。
"""
from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import settings
from ..physics.engine import KeyFrame, SimulationEngine
from ..schemas import (
    BackendInfo,
    RunStatus,
    SimulationParams,
    SimulationSummary,
)
from .frames import encode_json_message, encode_message

SUBSCRIBER_QUEUE_SIZE = 8


class FrameBus:
    """Thread-safe fan-out of keyframes to asyncio consumers.

    Messages are either ``bytes`` (a binary keyframe) or ``str`` (a JSON
    control event). The distinction matters on the wire: the stream endpoint
    forwards ``bytes`` with ``send_bytes`` and ``str`` with ``send_text``, so a
    control message must never be encoded to bytes before publishing or the
    client cannot tell it apart from a keyframe.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[bytes | str | None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.latest: KeyFrame | None = None
        # The handshake is published once, before any client can be listening,
        # so it is retained and replayed to every new subscriber.
        self.meta: bytes | None = None
        self.lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[bytes | str | None]:
        queue: asyncio.Queue[bytes | str | None] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_SIZE
        )
        with self.lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes | str | None]) -> None:
        with self.lock:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        with self.lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    def publish_meta(self, message: bytes) -> None:
        """Publish the run handshake and retain it for late subscribers."""
        self.meta = message
        self.publish(message)

    def publish(self, message: bytes | str | None, frame: KeyFrame | None = None) -> None:
        """Publish from any thread. ``None`` is the end-of-stream sentinel."""
        if frame is not None:
            self.latest = frame
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._fan_out, message)

    def _fan_out(self, message: bytes | str | None) -> None:
        with self.lock:
            targets = list(self._subscribers)
        for queue in targets:
            if queue.full():
                try:
                    queue.get_nowait()  # drop the oldest frame for slow clients
                except asyncio.QueueEmpty:  # pragma: no cover - race only
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:  # pragma: no cover - raced with another put
                pass


@dataclass
class RunRecord:
    """Everything the API needs to know about one run."""

    run_id: str
    params: SimulationParams
    status: RunStatus = RunStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    message: str = ""
    error: str | None = None

    engine: SimulationEngine | None = None
    backend: BackendInfo | None = None
    bus: FrameBus = field(default_factory=FrameBus)
    thread: threading.Thread | None = None

    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)

    metrics: dict[str, Any] = field(default_factory=dict)
    frames_seen: int = 0

    @property
    def simulated_time(self) -> float:
        return float(self.engine.solver.time) if self.engine is not None else 0.0

    @property
    def progress(self) -> float:
        return float(self.engine.progress) if self.engine is not None else 0.0

    def summary(self) -> SimulationSummary:
        return SimulationSummary(
            run_id=self.run_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            progress=self.progress,
            simulated_time_s=self.simulated_time,
            wall_time_s=float(self.metrics.get("wall_time_s", 0.0)),
            n_particles=int(
                self.params.particle_count if self.engine is None
                else getattr(self.engine.solver, "n", self.params.particle_count)
            ),
            n_frames=self.frames_seen,
            message=self.message or (self.error or ""),
            params=self.params,
            backend=self.backend,
        )


class TaskManager:
    """Registry + scheduler for simulation runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the serving event loop so worker threads can publish frames.

        Every ``FrameBus`` holds its own loop reference — publishing is a
        silent no-op until that bus has one — so binding the manager must also
        reach the buses of runs that already exist.
        """
        self._loop = loop
        with self._lock:
            existing = list(self._runs.values())
        for record in existing:
            record.bus.bind_loop(loop)

    # ------------------------------------------------------------------
    def create(self, params: SimulationParams) -> RunRecord:
        """Build the engine eagerly so the UI gets terrain/grid/backend up front.

        Raises :class:`BackendUnavailable` when the caller explicitly asked for
        an optional backend that is not installed here.
        """
        run_id = uuid.uuid4().hex[:12]
        engine = SimulationEngine(params)
        record = RunRecord(
            run_id=run_id, params=params, engine=engine, backend=engine.backend
        )
        # Inherit the serving loop, otherwise this run's frames go nowhere.
        if self._loop is not None:
            record.bus.bind_loop(self._loop)
        with self._lock:
            self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    # ------------------------------------------------------------------
    def start(self, run_id: str) -> RunRecord:
        record = self._runs[run_id]
        if record.status is RunStatus.RUNNING:
            return record

        record.status = RunStatus.RUNNING
        record.started_at = time.time()
        record.stop_event.clear()
        record.pause_event.clear()

        thread = threading.Thread(
            target=self._run_worker, args=(record,), name=f"dfs-run-{run_id}", daemon=True
        )
        record.thread = thread
        thread.start()
        return record

    # ------------------------------------------------------------------
    def _run_worker(self, record: RunRecord) -> None:
        try:
            self._execute(record)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            record.error = f"{type(exc).__name__}: {exc}"
            record.message = record.error
            record.status = RunStatus.FAILED
            record.finished_at = time.time()
            traceback.print_exc()
            # Control events stay ``str`` so the socket sends them as TEXT
            # frames; a keyframe is the only thing that goes out as binary.
            record.bus.publish(
                encode_json_message(
                    {"type": "error", "run_id": record.run_id, "message": record.error}
                )
            )
            record.bus.publish(None)

    # ------------------------------------------------------------------
    def _execute(self, record: RunRecord) -> None:
        params = record.params

        engine = record.engine
        if engine is None:  # defensive: a record created outside ``create()``
            engine = SimulationEngine(params)
            record.engine = engine
            record.backend = engine.backend

        grid = engine.solver.grid
        terrain = engine.terrain

        # --- handshake: terrain + grid + backend --------------------------
        record.bus.publish_meta(
            encode_message(
                {
                    "type": "meta",
                    "run_id": record.run_id,
                    "params": params.model_dump(mode="json"),
                    "terrain": terrain.to_info().model_dump(mode="json"),
                    "grid": grid.model_dump(mode="json"),
                    "backend": engine.backend.model_dump(mode="json"),
                    "n_particles": int(getattr(engine.solver, "n", params.particle_count)),
                    "duration_s": params.duration_s,
                    "frame_stride": params.frame_stride,
                },
                [("bed", np.asarray(terrain.height(
                    grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
                ), dtype=np.float32)[None, :].repeat(grid.ny, axis=0))],
            )
        )

        def on_frame(frame: KeyFrame) -> None:
            record.frames_seen += 1
            record.message = (
                f"t={frame.time:.2f}s / {params.duration_s:.2f}s "
                f"(step {frame.step})"
            )
            record.bus.publish(
                encode_message(
                    {
                        "type": "frame",
                        "run_id": record.run_id,
                        "index": frame.index,
                        "step": frame.step,
                        "time": frame.time,
                        "dt": frame.dt,
                        "n_particles_total": frame.n_particles_total,
                        "particle_stride": frame.particle_stride,
                        "grid": grid.model_dump(mode="json"),
                        "stats": {k: float(v) for k, v in frame.stats.items()},
                    },
                    frame.to_arrays(),
                ),
                frame=frame,
            )

        try:
            metrics = engine.run(
                on_frame=on_frame,
                should_stop=record.stop_event.is_set,
                should_pause=record.pause_event.is_set,
            )
        except Exception as exc:  # noqa: BLE001
            # An optional accelerated backend may fail mid-run (PySPH/SynxFlow
            # are not installed by default and are not covered by CI here).
            # Fall back to the built-in CPU solver instead of failing the run.
            if engine.backend.accelerated:
                fallback = params.model_copy(update={"backend": "shallow_water"})
                record.message = (
                    f"{engine.backend.active} 运行失败({type(exc).__name__})，"
                    "已自动回退到 CPU 浅水方程求解器"
                )
                engine = SimulationEngine(fallback)
                record.engine = engine
                record.backend = engine.backend
                engine.backend.notes.append(record.message)
                metrics = engine.run(
                    on_frame=on_frame,
                    should_stop=record.stop_event.is_set,
                    should_pause=record.pause_event.is_set,
                )
            else:
                raise

        record.metrics = metrics
        record.finished_at = time.time()
        record.status = (
            RunStatus.CANCELLED if record.stop_event.is_set() else RunStatus.FINISHED
        )
        record.bus.publish(
            encode_json_message(
                {
                    "type": "completed",
                    "run_id": record.run_id,
                    "status": record.status.value,
                    "metrics": metrics,
                    "n_frames": record.frames_seen,
                }
            )
        )
        record.bus.publish(None)

    # ------------------------------------------------------------------
    def stop(self, run_id: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record is None:
            return None
        record.stop_event.set()
        record.pause_event.clear()
        return record

    def pause(self, run_id: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record is None:
            return None
        record.pause_event.set()
        record.status = RunStatus.PAUSED
        return record

    def resume(self, run_id: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record is None:
            return None
        record.pause_event.clear()
        record.status = RunStatus.RUNNING
        return record

    def shutdown(self) -> None:
        for record in self._runs.values():
            record.stop_event.set()


task_manager = TaskManager()
