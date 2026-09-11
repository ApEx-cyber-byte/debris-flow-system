"""Global runtime configuration for the debris-flow analysis backend.

All values can be overridden through environment variables prefixed with
``DFS_`` (e.g. ``DFS_MAX_PARTICLES=20000``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------
GRAVITY: float = 9.80665  # m/s^2, standard gravity
WATER_DENSITY: float = 1000.0  # kg/m^3
SOLID_DENSITY_DEFAULT: float = 2650.0  # kg/m^3, quartz-rich sediment


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Process-wide settings. Frozen so a run cannot mutate shared state."""

    host: str = field(default_factory=lambda: _env_str("DFS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("DFS_PORT", 8000))

    # --- domain limits (guard-rails, mirrored by the frontend) ---
    min_particles: int = 200
    max_particles: int = field(default_factory=lambda: _env_int("DFS_MAX_PARTICLES", 20000))
    default_particles: int = 4000

    max_duration_s: float = field(default_factory=lambda: _env_float("DFS_MAX_DURATION", 120.0))
    max_concurrent_runs: int = field(default_factory=lambda: _env_int("DFS_MAX_RUNS", 2))

    # --- streaming ---
    # The specification requires one pushed keyframe every 10 time steps.
    default_frame_stride: int = 10
    # Hard cap on particles serialised per frame (frontend downsamples further).
    max_frame_particles: int = field(default_factory=lambda: _env_int("DFS_FRAME_PARTICLES", 6000))
    # Frames retained in memory for the live charts / export.
    max_retained_frames: int = field(default_factory=lambda: _env_int("DFS_KEEP_FRAMES", 3000))

    # --- storage ---
    runs_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("DFS_RUNS_DIR", str(Path(__file__).resolve().parent.parent / "runs"))
        )
    )

    # --- solver selection ---
    # "auto" probes PySPH / SynxFlow / PyTorch and silently falls back.
    default_backend: str = field(default_factory=lambda: _env_str("DFS_BACKEND", "auto"))

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
