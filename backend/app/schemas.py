"""Pydantic schemas — the REST/WebSocket contract shared with the frontend.

The TypeScript mirror of this file lives at
``frontend/src/types/simulation.ts``. Any change here MUST be mirrored there.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import settings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class RheologyModel(str, Enum):
    """Non-Newtonian constitutive laws supported by the solver."""

    BINGHAM = "bingham"
    CROSS = "cross"
    HERSCHEL_BULKLEY = "herschel_bulkley"
    POWER_LAW = "power_law"


class SolverBackend(str, Enum):
    """Which numerical engine should advance the simulation."""

    AUTO = "auto"          # probe PySPH -> SynxFlow -> NumPy SPH
    NUMPY_SPH = "numpy_sph"  # built-in vectorised SPH (always available)
    PYSPH = "pysph"        # https://github.com/sweeneyde/pysph  (optional)
    SYNXFLOW = "synxflow"  # https://github.com/SynxFlow/SynxFlow (optional, CUDA)
    SHALLOW_WATER = "shallow_water"  # built-in CPU depth-integrated solver


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BoundaryCondition(str, Enum):
    """Bed treatment at the slope toe / runout plane."""

    OPEN = "open"      # material may leave the domain
    WALL = "wall"      # a vertical barrier stops the flow


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
class SimulationParams(BaseModel):
    """User-facing physical parameters.

    Every default reproduces the reference case: a 4 m thick debris column
    released on a 30 deg slope of 20 m, running out onto a flat plane.
    """

    model_config = {"extra": "forbid"}

    # --- slope geometry -----------------------------------------------------
    slope_angle_deg: float = Field(
        30.0, ge=0.0, le=60.0,
        description="坡度角 (deg)。0=水平, 60=陡坡。规格要求 0–60 可调。",
    )
    slope_length_m: float = Field(
        20.0, gt=0.5, le=200.0, description="坡长 (m), 沿坡面量测。"
    )

    # --- rheology -----------------------------------------------------------
    rheology: RheologyModel = Field(
        RheologyModel.BINGHAM, description="非牛顿本构模型"
    )
    density_kg_m3: float = Field(
        2200.0, ge=800.0, le=3200.0, description="泥石流混合体密度 (kg/m^3)"
    )
    volume_concentration: float = Field(
        0.45, ge=0.05, le=0.75, description="体积浓度 Cv (固体体积分数)"
    )
    yield_stress_pa: float = Field(
        500.0, ge=0.0, le=20000.0, description="屈服应力 tau_y (Pa), Bingham/HB 用"
    )
    viscosity_pa_s: float = Field(
        60.0, ge=0.1, le=10000.0, description="粘性系数 mu (Pa*s)"
    )
    manning_n: float = Field(
        0.04, ge=0.0, le=0.30, description="Manning 粗糙度系数 n (s/m^(1/3))"
    )

    # --- extended constitutive parameters ----------------------------------
    power_law_index: float = Field(
        0.6, gt=0.05, le=2.0,
        description="流性指数 n。Cross: 幂指数; HB/Power-law: 剪切稀化指数。",
    )
    cross_lambda_s: float = Field(
        2.0, gt=1e-4, le=1000.0,
        description="Cross 模型时间常数 lambda (s)。仅 CROSS 模型使用。",
    )
    cross_mu_inf_pa_s: float = Field(
        5.0, ge=0.0, le=10000.0,
        description="Cross 模型无穷剪切粘度 mu_inf (Pa*s)。",
    )
    max_apparent_viscosity_pa_s: float = Field(
        20000.0, ge=1.0, le=1.0e6,
        description="表观粘度上限 (Pa*s), 用于正则化屈服面奇异性。",
    )
    basal_friction_coeff: float = Field(
        0.12, ge=0.0, le=1.0,
        description="基底库仑摩擦系数 mu_bed (Voellmy/Coulomb 项)。",
    )
    turbulence_coeff: float = Field(
        0.0, ge=0.0, le=1.0,
        description="湍流/弥散附加阻力系数 (0=关闭)。",
    )

    # --- release geometry ---------------------------------------------------
    initial_volume_m3: float = Field(
        150.0, gt=0.1, le=100000.0, description="初始释放体积 (m^3)"
    )
    release_length_m: float = Field(
        8.0, gt=0.1, le=100.0, description="释放区沿坡长度 (m)"
    )
    channel_width_m: float = Field(
        8.0, gt=0.5, le=100.0, description="释放区宽度 (m)"
    )
    release_height_m: float = Field(
        0.0, ge=0.0, le=50.0,
        description="初始泥石流厚度 (m)。0 = 由体积与释放区面积自动推算。"
    )

    # --- time integration ---------------------------------------------------
    duration_s: float = Field(
        12.0, gt=0.05, le=120.0, description="模拟时长 (s)"
    )
    particle_count: int = Field(
        4000, ge=200, le=20000, description="SPH 粒子数量"
    )

    # --- solver / streaming -------------------------------------------------
    backend: SolverBackend = Field(
        SolverBackend.AUTO, description="求解后端; auto 自动探测并回退"
    )
    frame_stride: int = Field(
        10, ge=1, le=200, description="每 N 个时间步推送一帧 (规格要求默认 10)"
    )
    cfl_number: float = Field(0.25, gt=0.01, le=0.8, description="CFL 数")
    seed: int = Field(42, ge=0, le=2**31 - 1, description="随机种子")

    # --- instrumentation ----------------------------------------------------
    barrier_distance_m: float = Field(
        3.0, ge=0.0, le=200.0,
        description="冲击力监测断面距坡脚的水平距离 (m)。",
    )
    boundary: BoundaryCondition = Field(
        BoundaryCondition.OPEN, description="坡脚边界条件"
    )

    # ------------------------------------------------------------------
    @field_validator("particle_count")
    @classmethod
    def _clamp_particles(cls, v: int) -> int:
        return max(settings.min_particles, min(settings.max_particles, v))

    @field_validator("duration_s")
    @classmethod
    def _clamp_duration(cls, v: float) -> float:
        return min(settings.max_duration_s, v)

    @model_validator(mode="after")
    def _check_release_fits_slope(self) -> "SimulationParams":
        if self.release_length_m > self.slope_length_m:
            raise ValueError(
                "release_length_m 不能超过 slope_length_m "
                f"({self.release_length_m} > {self.slope_length_m})"
            )
        return self

    # ------------------------------------------------------------------
    @property
    def slope_angle_rad(self) -> float:
        import math

        return math.radians(self.slope_angle_deg)

    @property
    def toe_x_m(self) -> float:
        """Horizontal distance from the crest to the slope toe."""
        import math

        return self.slope_length_m * math.cos(self.slope_angle_rad)

    @property
    def crest_height_m(self) -> float:
        """Vertical drop of the slope."""
        import math

        return self.slope_length_m * math.sin(self.slope_angle_rad)


# ---------------------------------------------------------------------------
# Runtime descriptors
# ---------------------------------------------------------------------------
class TerrainInfo(BaseModel):
    """Bed geometry handed to the frontend for mesh construction."""

    slope_angle_deg: float
    slope_length_m: float
    toe_x_m: float
    crest_height_m: float
    turn_radius_m: float
    domain_x_min: float
    domain_x_max: float
    domain_y_min: float
    domain_y_max: float


class GridInfo(BaseModel):
    """Metadata describing the depth / velocity raster streamed in frames."""

    nx: int
    ny: int
    x0: float
    y0: float
    dx: float
    dy: float


class BackendInfo(BaseModel):
    """Which engine is actually running, and why."""

    requested: str
    active: str
    accelerated: bool
    notes: list[str] = Field(default_factory=list)


class SimulationSummary(BaseModel):
    """Lightweight run record used by list endpoints."""

    run_id: str
    status: RunStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    progress: float = 0.0
    simulated_time_s: float = 0.0
    wall_time_s: float = 0.0
    n_particles: int = 0
    n_frames: int = 0
    message: str = ""
    params: SimulationParams | None = None
    backend: BackendInfo | None = None


class SimulationCreated(BaseModel):
    run_id: str
    status: RunStatus
    ws_url: str
    terrain: TerrainInfo
    grid: GridInfo
    backend: BackendInfo


class TimeSeriesPoint(BaseModel):
    """One row of the diagnostic time series."""

    t: float
    max_speed: float
    mean_speed: float
    front_x: float
    max_depth: float
    kinetic_energy: float
    impact_force_kn: float
    deposited_volume: float


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    params: SimulationParams
    backend: BackendInfo
    terrain: TerrainInfo
    grid: GridInfo
    time_series: list[TimeSeriesPoint]
    metrics: dict[str, Any]
    wall_time_s: float


# ---------------------------------------------------------------------------
# AI surrogate
# ---------------------------------------------------------------------------
class AIPredictRequest(BaseModel):
    params: SimulationParams
    model_name: str = Field("heuristic-v1", description="代理模型名称")


class AIPredictResponse(BaseModel):
    model_name: str
    available_models: list[str]
    is_surrogate_trained: bool
    runout_distance_m: float
    max_velocity_m_s: float
    arrival_time_s: float
    max_flow_depth_m: float
    peak_impact_force_kn: float
    affected_area_m2: float
    confidence: float
    notes: list[str] = Field(default_factory=list)


class ExportFormat(str, Enum):
    CSV = "csv"
    JSON = "json"


class ExportKind(str, Enum):
    TIME_SERIES = "time_series"
    PARTICLES = "particles"
    DEPTH_PROFILE = "depth_profile"
    FULL = "full"


class ExportRequest(BaseModel):
    kind: ExportKind = ExportKind.TIME_SERIES
    fmt: ExportFormat = ExportFormat.CSV
    frame_index: int | None = Field(
        None, ge=0, description="PARTICLES 导出所需的关键帧索引; 默认最后一帧"
    )
