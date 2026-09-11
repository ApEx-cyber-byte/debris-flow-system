"""仿真引擎 / Simulation engine.

把「求解器」与「任务调度」解耦：:class:`SimulationEngine` 只负责
推进时间、按 ``frame_stride`` 抽取关键帧、计算诊断量；它不知道
自己跑的是 NumPy SPH 还是 GPU 浅水方程，也不知道帧要发去哪里。

Frame production contract (mirrors ``frontend/src/types/simulation.ts``)
-----------------------------------------------------------------------
* 每 ``frame_stride`` 个时间步产出一帧（规格要求默认 10）。
* 每帧包含：粒子位置 / 速度（可降采样至 ``max_frame_particles``）、
  深度栅格、速度矢量栅格、以及一组标量诊断量。
* 所有数组以 ``float32`` 传出，跨 WebSocket 用二进制帧传输。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..config import settings
from ..schemas import SimulationParams
from .backends import create_solver
from .terrain import Terrain, build_terrain


@dataclass
class KeyFrame:
    """One pushed keyframe (metal + arrays), backend-agnostic."""

    index: int
    step: int
    time: float
    dt: float
    particle_positions: np.ndarray  # (M, 3) float32
    particle_velocities: np.ndarray  # (M, 3) float32
    particle_stride: int
    n_particles_total: int
    depth: np.ndarray  # (ny, nx) float32
    velocity_x: np.ndarray  # (ny, nx) float32
    velocity_y: np.ndarray  # (ny, nx) float32
    stats: dict[str, float] = field(default_factory=dict)

    def to_arrays(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("pos", self.particle_positions.astype(np.float32, copy=False)),
            ("vel", self.particle_velocities.astype(np.float32, copy=False)),
            ("depth", self.depth.astype(np.float32, copy=False)),
            ("vx", self.velocity_x.astype(np.float32, copy=False)),
            ("vy", self.velocity_y.astype(np.float32, copy=False)),
        ]


class SimulationEngine:
    """Owns one solver instance and turns its state into keyframes."""

    def __init__(self, params: SimulationParams) -> None:
        self.params = params
        self.terrain: Terrain = build_terrain(params)
        self.solver, self.backend = create_solver(params)

        # Fall back to the built-in CPU solvers if an optional backend fails
        # at construction time (e.g. a partially-installed PySPH).
        self.grid = getattr(self.solver, "grid", None)

        self.frame_index = 0
        self.max_frame_particles = max(
            settings.min_particles, min(settings.max_particles, settings.max_frame_particles)
        )
        self._stride = max(1, int(params.particle_count) // self.max_frame_particles)

        self.time_series: list[dict[str, float]] = []
        self.frames: list[KeyFrame] = []

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(self.params.duration_s)

    @property
    def progress(self) -> float:
        return float(min(1.0, self.solver.time / max(self.duration, 1.0e-9)))

    @property
    def finished(self) -> bool:
        return self.solver.time >= self.duration - 1.0e-12

    # ------------------------------------------------------------------
    def _sample_particles(self) -> tuple[np.ndarray, np.ndarray, int]:
        pos, vel = self.solver.particle_state()
        stride = self._stride
        if stride > 1 and pos.shape[0] > self.max_frame_particles:
            pos = pos[::stride]
            vel = vel[::stride]
        else:
            stride = 1
        return (
            np.ascontiguousarray(pos, dtype=np.float32),
            np.ascontiguousarray(vel, dtype=np.float32),
            stride,
        )

    # ------------------------------------------------------------------
    def capture_frame(self, dt: float) -> KeyFrame:
        depth = np.asarray(self.solver.depth_field(), dtype=np.float64)
        vx, vy = self.solver.velocity_field()
        stats = self.solver.summary(depth)
        stats["impact_force"] = self.solver.impact_force(depth, vx)

        pos, vel, stride = self._sample_particles()
        total = int(
            getattr(self.solver, "n", None)
            or getattr(self.solver, "tracer_x", np.empty(0)).size
            or self.params.particle_count
        )

        frame = KeyFrame(
            index=self.frame_index,
            step=self.solver.step_count,
            time=float(self.solver.time),
            dt=float(dt),
            particle_positions=pos,
            particle_velocities=vel,
            particle_stride=stride,
            n_particles_total=total,
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            velocity_x=np.ascontiguousarray(vx, dtype=np.float32),
            velocity_y=np.ascontiguousarray(vy, dtype=np.float32),
            stats=stats,
        )
        self.frame_index += 1
        return frame

    # ------------------------------------------------------------------
    def record(self, frame: KeyFrame) -> None:
        """Append the frame to the in-memory history used for export."""
        if len(self.frames) < settings.max_retained_frames:
            self.frames.append(frame)
        self.time_series.append(
            {
                "t": frame.time,
                "max_speed": frame.stats.get("max_speed", 0.0),
                "mean_speed": frame.stats.get("mean_speed", 0.0),
                "front_x": frame.stats.get("front_x", 0.0),
                "max_depth": frame.stats.get("max_depth", 0.0),
                "kinetic_energy": frame.stats.get("kinetic_energy", 0.0),
                "impact_force_kn": frame.stats.get("impact_force", 0.0) / 1000.0,
                "deposited_volume": frame.stats.get("deposited_volume", 0.0),
            }
        )

    # ------------------------------------------------------------------
    def run(
        self,
        on_frame: Callable[[KeyFrame], None],
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
        on_step: Callable[[float], None] | None = None,
    ) -> dict[str, object]:
        """Integrate to completion, emitting a keyframe every ``frame_stride`` steps.

        Returns a metrics dictionary describing the completed run.
        """
        started = time.perf_counter()
        stride = max(1, int(self.params.frame_stride))
        peak: dict[str, float] = {
            "max_speed": 0.0,
            "max_depth": 0.0,
            "max_impact_force_kn": 0.0,
            "max_kinetic_energy": 0.0,
            "max_front_x_m": 0.0,
            "runout_distance_m": 0.0,
        }

        while not self.finished:
            if should_stop is not None and should_stop():
                break
            if should_pause is not None:
                while should_pause() and not (should_stop is not None and should_stop()):
                    time.sleep(0.05)

            dt = self.solver.step()

            if self.solver.step_count % stride == 0 or self.finished:
                frame = self.capture_frame(dt)
                self.record(frame)
                self._update_peaks(peak, frame)
                on_frame(frame)

            if on_step is not None:
                on_step(self.solver.time)

        wall = time.perf_counter() - started
        # 跑距按「坡脚下游的水平距离」定义，与 AI 代理模型的口径一致；
        # 绝对前锋坐标另存为 max_front_x_m，避免两者混用。
        peak["runout_distance_m"] = max(
            0.0, peak["max_front_x_m"] - self.terrain.toe_x_m
        )
        return {
            **peak,
            "wall_time_s": wall,
            "simulated_time_s": float(self.solver.time),
            "steps": int(self.solver.step_count),
            "frames": int(self.frame_index),
            "speedup_ratio": float(self.solver.time / wall) if wall > 0 else 0.0,
            "n_particles": self.solver.n if hasattr(self.solver, "n") else int(
                self.params.particle_count
            ),
            "backend": self.backend.active,
            "toe_x_m": self.terrain.toe_x_m,
        }

    @staticmethod
    def _update_peaks(peak: dict[str, float], frame: KeyFrame) -> None:
        peak["max_speed"] = max(peak["max_speed"], frame.stats.get("max_speed", 0.0))
        peak["max_depth"] = max(peak["max_depth"], frame.stats.get("max_depth", 0.0))
        peak["max_impact_force_kn"] = max(
            peak["max_impact_force_kn"], frame.stats.get("impact_force", 0.0) / 1000.0
        )
        peak["max_kinetic_energy"] = max(
            peak["max_kinetic_energy"], frame.stats.get("kinetic_energy", 0.0)
        )
        peak["max_front_x_m"] = max(peak["max_front_x_m"], frame.stats.get("front_x", 0.0))

    # ------------------------------------------------------------------
    def depth_profile(self) -> list[dict[str, float]]:
        """Depth profile along the flow centreline from the final frame."""
        if not self.frames:
            return []
        frame = self.frames[-1]
        grid = self.solver.grid
        centre = frame.depth.shape[0] // 2
        row = frame.depth[centre, :]
        xs = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
        return [
            {"x": float(xs[i]), "depth": float(row[i])}
            for i in range(grid.nx)
            if row[i] > 1.0e-4
        ]


def iter_backend_notes() -> Iterator[str]:
    from .backends import probe

    yield from probe().notes
