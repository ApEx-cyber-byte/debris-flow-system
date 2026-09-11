"""后端测试 / Backend test-suite.

运行::

    cd backend
    .venv\\Scripts\\python -m pytest tests -q

覆盖范围：
* 非牛顿本构的量纲与单调性
* 地形几何与解析梯度
* 两个 CPU 求解器的质量守恒与稳定性
* WebSocket 二进制帧编解码
* REST / WebSocket 端到端（可选用例）
"""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from app.config import GRAVITY
from app.physics.rheology import (
    apparent_viscosity,
    basal_shear_stress,
    mixture_density,
)
from app.physics.shallow_water import ShallowWaterSolver
from app.physics.sph_numpy import NumPySPHSolver
from app.physics.terrain import build_grid, build_terrain
from app.schemas import RheologyModel, SimulationParams, SolverBackend
from app.services.frames import decode_message, encode_message


def frame_roundtrip() -> tuple[dict, dict[str, np.ndarray]]:
    """Encode a representative keyframe and decode it again."""
    positions = np.arange(300, dtype=np.float32).reshape(100, 3)
    depth = np.linspace(0.0, 2.0, 16 * 8).reshape(8, 16).astype(np.float32)
    message = encode_message(
        {
            "type": "frame",
            "run_id": "test",
            "index": 1,
            "grid": {"nx": 16, "ny": 8, "x0": 0.0, "y0": 0.0, "dx": 0.5, "dy": 0.5},
            "stats": {"max_speed": 1.5},
        },
        [("pos", positions), ("vel", positions * 0.5), ("depth", depth)],
    )
    return decode_message(message)


# ---------------------------------------------------------------------------
# Rheology
# ---------------------------------------------------------------------------
class TestRheology:
    def test_bingham_has_yield_stress(self) -> None:
        """At vanishing shear rate the Bingham fluid must appear very viscous."""
        params = SimulationParams(rheology=RheologyModel.BINGHAM, yield_stress_pa=1000.0)
        mu_low = apparent_viscosity(np.array([1.0e-3]), params)[0]
        mu_high = apparent_viscosity(np.array([100.0]), params)[0]
        assert mu_low > mu_high * 50, (mu_low, mu_high)

    def test_bingham_recovers_plastic_viscosity_at_high_shear(self) -> None:
        params = SimulationParams(
            rheology=RheologyModel.BINGHAM, yield_stress_pa=500.0, viscosity_pa_s=80.0
        )
        mu = apparent_viscosity(np.array([1.0e5]), params)[0]
        assert abs(mu - 80.0) < 1.0, mu

    def test_power_law_shear_thinning(self) -> None:
        params = SimulationParams(
            rheology=RheologyModel.POWER_LAW, viscosity_pa_s=50.0, power_law_index=0.5
        )
        mu = apparent_viscosity(np.array([0.1, 1.0, 10.0]), params)
        assert mu[0] > mu[1] > mu[2], mu

    def test_cross_model_bounds(self) -> None:
        params = SimulationParams(
            rheology=RheologyModel.CROSS,
            viscosity_pa_s=200.0,
            cross_mu_inf_pa_s=10.0,
            cross_lambda_s=2.0,
            power_law_index=1.0,
        )
        mu = apparent_viscosity(np.array([1.0e-6, 1.0e6]), params)
        assert mu[0] == pytest.approx(200.0, rel=0.05)
        assert mu[1] == pytest.approx(10.0, rel=0.05)

    def test_viscosity_is_always_finite_and_capped(self) -> None:
        for model in RheologyModel:
            params = SimulationParams(rheology=model, max_apparent_viscosity_pa_s=5000.0)
            mu = apparent_viscosity(np.array([0.0, 1.0e-8, 1.0, 1.0e8]), params)
            assert np.all(np.isfinite(mu))
            assert np.all(mu <= 5000.0 + 1e-9)
            assert np.all(mu > 0.0)

    def test_basal_stress_grows_with_roughness_and_depth(self) -> None:
        params = SimulationParams(manning_n=0.05, yield_stress_pa=400.0)
        shallow = basal_shear_stress(np.array([0.5]), np.array([5.0]), 0.87, params)
        deep = basal_shear_stress(np.array([2.0]), np.array([5.0]), 0.87, params)
        assert deep[0] > shallow[0]  # more overburden -> more Coulomb friction

    def test_mixture_density_brackets_endpoints(self) -> None:
        assert mixture_density(1.0) == pytest.approx(2650.0)
        assert mixture_density(0.0) == pytest.approx(1000.0)
        assert 1000.0 < mixture_density(0.45) < 2650.0


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
class TestTerrain:
    def test_crest_height_matches_slope_geometry(self) -> None:
        params = SimulationParams(slope_angle_deg=30.0, slope_length_m=20.0)
        terrain = build_terrain(params)
        # softplus introduces a ~0.2% offset from the ideal triangle
        assert terrain.height(0.0) == pytest.approx(10.0, rel=0.01)
        assert terrain.height(terrain.toe_x_m * 3) < 0.05

    def test_gradient_is_negative_on_the_slope(self) -> None:
        params = SimulationParams(slope_angle_deg=35.0, slope_length_m=20.0)
        terrain = build_terrain(params)
        dzdx, _ = terrain.gradient(np.array([1.0]))
        assert dzdx[0] == pytest.approx(-math.tan(math.radians(35.0)), rel=0.02)

    def test_gradient_is_flat_on_the_runout(self) -> None:
        params = SimulationParams(slope_angle_deg=35.0, slope_length_m=20.0)
        terrain = build_terrain(params)
        dzdx, _ = terrain.gradient(np.array([terrain.toe_x_m * 4]))
        assert abs(float(dzdx[0])) < 1e-3

    def test_normal_is_unit_and_tilted_on_slope(self) -> None:
        params = SimulationParams(slope_angle_deg=40.0, slope_length_m=20.0)
        terrain = build_terrain(params)
        n = terrain.normal(np.array([1.0]))[0]
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert n[2] == pytest.approx(math.cos(math.radians(40.0)), rel=0.02)

    def test_horizontal_slope_has_zero_crest(self) -> None:
        params = SimulationParams(slope_angle_deg=0.0, slope_length_m=20.0)
        terrain = build_terrain(params)
        assert terrain.crest_height_m == pytest.approx(0.0, abs=1e-9)
        assert float(np.max(np.abs(terrain.height(np.linspace(0, 30, 50))))) < 1e-9

    def test_grid_covers_the_domain(self) -> None:
        params = SimulationParams()
        terrain = build_terrain(params)
        grid = build_grid(terrain)
        assert grid.x0 + grid.nx * grid.dx >= terrain.domain_x_max - 1e-9
        assert grid.y0 + grid.ny * grid.dy >= terrain.domain_y_max - 1e-9


# ---------------------------------------------------------------------------
# Shallow-water solver
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shallow_water_run():
    """Run the depth-integrated solver to arrest once, sharing the result."""
    params = SimulationParams(
        duration_s=9.0, particle_count=800, backend=SolverBackend.SHALLOW_WATER
    )
    solver = ShallowWaterSolver(params)
    peak = 0.0
    while solver.time < params.duration_s:
        solver.step()
        if solver.step_count % 50 == 0:
            vx, vy = solver.velocity_field()
            peak = max(peak, float(np.sqrt(vx * vx + vy * vy).max()))
    return params, solver, peak


class TestShallowWater:
    def test_conserves_mass(self, shallow_water_run) -> None:
        params, solver, _peak = shallow_water_run
        volume = float(solver.h.sum() * solver.grid.dx * solver.grid.dy)
        assert volume == pytest.approx(params.initial_volume_m3, rel=1e-6)

    def test_flow_accelerates_then_arrests(self, shallow_water_run) -> None:
        """The release must run out and then genuinely stop (yield stress)."""
        _params, solver, peak = shallow_water_run
        depth = solver.depth_field()
        vx, vy = solver.velocity_field()
        final = float(np.sqrt(vx * vx + vy * vy).max())

        assert peak > 2.0, f"flow never accelerated (peak={peak})"
        assert final < 0.2 * peak, f"flow never arrested (peak={peak}, final={final})"
        assert depth.max() > 0.05

    def test_no_nan(self, shallow_water_run) -> None:
        _params, solver, _peak = shallow_water_run
        assert np.all(np.isfinite(solver.h))
        assert np.all(np.isfinite(solver.hu))
        assert np.all(np.isfinite(solver.hv))

    def test_front_advances_down_slope(self) -> None:
        params = SimulationParams(
            duration_s=2.0, particle_count=500, backend=SolverBackend.SHALLOW_WATER
        )
        solver = ShallowWaterSolver(params)
        start_front = solver.summary(solver.depth_field())["front_x"]
        for _ in range(400):
            solver.step()
        end_front = solver.summary(solver.depth_field())["front_x"]
        assert end_front > start_front

    def test_time_step_is_adaptive_and_bounded(self) -> None:
        params = SimulationParams(backend=SolverBackend.SHALLOW_WATER)
        solver = ShallowWaterSolver(params)
        for _ in range(50):
            dt = solver.step()
            assert 0 < dt <= 0.05


# ---------------------------------------------------------------------------
# SPH solver
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sph_solver():
    params = SimulationParams(
        duration_s=0.4, particle_count=600, backend=SolverBackend.NUMPY_SPH
    )
    instance = NumPySPHSolver(params)
    for _ in range(120):
        instance.step()
    return instance


class TestNumPySPH:
    def test_particles_stay_finite(self, sph_solver) -> None:
        assert np.all(np.isfinite(sph_solver.pos))
        assert np.all(np.isfinite(sph_solver.vel))
        assert np.all(np.isfinite(sph_solver.rho))

    def test_density_responds_to_compression(self, sph_solver) -> None:
        """Regression guard: the density field must not be pinned to rho0.

        An earlier formulation normalised density per-particle, which made it
        identically rho0 and left the pressure field at zero everywhere.
        """
        assert sph_solver.pressure.max() > 0.0
        assert sph_solver.rho.max() > sph_solver.rho0

    def test_gravity_makes_the_flow_move(self, sph_solver) -> None:
        speed = np.linalg.norm(sph_solver.vel, axis=1)
        assert speed.max() > 0.3

    def test_particles_never_sink_through_the_bed(self, sph_solver) -> None:
        bed = sph_solver.terrain.height(sph_solver.pos[:, 0])
        assert np.all(sph_solver.pos[:, 2] >= bed - 1e-6)

    def test_density_stays_within_physical_bounds(self, sph_solver) -> None:
        assert sph_solver.rho.min() >= 0.5 * sph_solver.rho0 - 1e-9
        assert sph_solver.rho.max() <= 3.0 * sph_solver.rho0 + 1e-9


# ---------------------------------------------------------------------------
# Frame codec
# ---------------------------------------------------------------------------
class TestFrameCodec:
    def test_binary_roundtrip(self) -> None:
        header, arrays = frame_roundtrip()
        assert header["type"] == "frame"
        assert arrays["pos"].size == 3 * 100
        assert arrays["depth"].size == 16 * 8
        np.testing.assert_allclose(arrays["pos"][:3], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(arrays["depth"][0], 0.0, atol=1e-6)

    def test_header_is_4_byte_aligned(self) -> None:
        """The client builds a Float32Array view over the payload."""
        message = encode_message({"type": "frame", "note": "中文标签"}, [("a", np.zeros(3))])
        (header_length,) = struct.unpack_from("<I", message, 0)
        assert (4 + header_length) % 4 == 0


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
class TestParams:
    def test_slope_angle_bounds(self) -> None:
        with pytest.raises(ValueError):
            SimulationParams(slope_angle_deg=61.0)
        with pytest.raises(ValueError):
            SimulationParams(slope_angle_deg=-1.0)

    def test_release_must_fit_on_the_slope(self) -> None:
        with pytest.raises(ValueError):
            SimulationParams(slope_length_m=10.0, release_length_m=20.0)

    def test_defaults_satisfy_the_specification(self) -> None:
        params = SimulationParams()
        assert 0 <= params.slope_angle_deg <= 60
        assert params.frame_stride == 10  # 每 10 个时间步推送一帧

    def test_derived_geometry(self) -> None:
        params = SimulationParams(slope_angle_deg=30.0, slope_length_m=20.0)
        assert params.toe_x_m == pytest.approx(20.0 * math.cos(math.radians(30.0)))
        assert params.crest_height_m == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# AI surrogate
# ---------------------------------------------------------------------------
class TestSurrogate:
    def test_prediction_is_finite_and_signed(self) -> None:
        from app.ai.physics_surrogate import predict

        result = predict(SimulationParams())
        assert result.runout_distance_m >= 0
        assert result.max_velocity_m_s > 0
        assert result.arrival_time_s > 0
        assert 0 < result.confidence <= 1

    def test_steeper_slopes_run_further(self) -> None:
        from app.ai.physics_surrogate import predict

        gentle = predict(SimulationParams(slope_angle_deg=15.0))
        steep = predict(SimulationParams(slope_angle_deg=45.0))
        assert steep.runout_distance_m > gentle.runout_distance_m
        assert steep.max_velocity_m_s > gentle.max_velocity_m_s

    def test_higher_yield_stress_shortens_runout(self) -> None:
        from app.ai.physics_surrogate import predict

        weak = predict(SimulationParams(yield_stress_pa=100.0))
        strong = predict(SimulationParams(yield_stress_pa=5000.0))
        assert strong.runout_distance_m < weak.runout_distance_m
