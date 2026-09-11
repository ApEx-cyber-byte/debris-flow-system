/**
 * TypeScript mirror of `backend/app/schemas.py`.
 *
 * IMPORTANT: this file is the frontend half of the API contract. Whenever the
 * Pydantic models change, this file must change with them — the two are
 * validated against each other by `backend/tests/test_contract.py`.
 */

export type RheologyModel = 'bingham' | 'cross' | 'herschel_bulkley' | 'power_law'

export type SolverBackend = 'auto' | 'numpy_sph' | 'pysph' | 'synxflow' | 'shallow_water'

export type RunStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'finished'
  | 'failed'
  | 'cancelled'

export type BoundaryCondition = 'open' | 'wall'

export interface SimulationParams {
  // slope geometry
  slope_angle_deg: number
  slope_length_m: number
  // rheology
  rheology: RheologyModel
  density_kg_m3: number
  volume_concentration: number
  yield_stress_pa: number
  viscosity_pa_s: number
  manning_n: number
  // extended constitutive parameters
  power_law_index: number
  cross_lambda_s: number
  cross_mu_inf_pa_s: number
  max_apparent_viscosity_pa_s: number
  basal_friction_coeff: number
  turbulence_coeff: number
  // release geometry
  initial_volume_m3: number
  release_length_m: number
  channel_width_m: number
  release_height_m: number
  // time integration
  duration_s: number
  particle_count: number
  // solver / streaming
  backend: SolverBackend
  frame_stride: number
  cfl_number: number
  seed: number
  // instrumentation
  barrier_distance_m: number
  boundary: BoundaryCondition
}

export interface TerrainInfo {
  slope_angle_deg: number
  slope_length_m: number
  toe_x_m: number
  crest_height_m: number
  turn_radius_m: number
  domain_x_min: number
  domain_x_max: number
  domain_y_min: number
  domain_y_max: number
}

export interface GridInfo {
  nx: number
  ny: number
  x0: number
  y0: number
  dx: number
  dy: number
}

export interface BackendInfo {
  requested: string
  active: string
  accelerated: boolean
  notes: string[]
}

export interface SimulationCreated {
  run_id: string
  status: RunStatus
  ws_url: string
  terrain: TerrainInfo
  grid: GridInfo
  backend: BackendInfo
}

export interface SimulationSummary {
  run_id: string
  status: RunStatus
  created_at: number
  started_at: number | null
  finished_at: number | null
  progress: number
  simulated_time_s: number
  wall_time_s: number
  n_particles: number
  n_frames: number
  message: string
  params: SimulationParams | null
  backend: BackendInfo | null
}

export interface TimeSeriesPoint {
  t: number
  max_speed: number
  mean_speed: number
  front_x: number
  max_depth: number
  kinetic_energy: number
  impact_force_kn: number
  deposited_volume: number
}

/** Full result payload from `GET /api/simulations/{id}/result`. */
export interface RunResult {
  run_id: string
  status: RunStatus
  params: SimulationParams
  backend: BackendInfo
  terrain: TerrainInfo
  grid: GridInfo
  time_series: TimeSeriesPoint[]
  metrics: Record<string, number>
  wall_time_s: number
}

/** Decoded keyframe header (see backend/app/services/frames.py). */
export interface FrameHeader {
  type: 'meta' | 'frame' | 'completed' | 'error'
  run_id: string
  index?: number
  step?: number
  time?: number
  dt?: number
  n_particles_total?: number
  particle_stride?: number
  grid?: GridInfo
  terrain?: TerrainInfo
  stats?: Record<string, number>
  params?: SimulationParams
  backend?: BackendInfo
  duration_s?: number
  frame_stride?: number
  n_particles?: number
}

/** One decoded keyframe with typed arrays ready for Three.js. */
export interface DecodedFrame {
  header: FrameHeader
  positions: Float32Array
  velocities: Float32Array
  depth: Float32Array
  velocityX: Float32Array
  velocityY: Float32Array
  receivedAt: number
}

export interface AIPredictResponse {
  model_name: string
  available_models: string[]
  is_surrogate_trained: boolean
  runout_distance_m: number
  max_velocity_m_s: number
  arrival_time_s: number
  max_flow_depth_m: number
  peak_impact_force_kn: number
  affected_area_m2: number
  confidence: number
  notes: string[]
}

export interface BackendReport {
  pysph: { available: boolean; url: string }
  synxflow: { available: boolean; cuda: boolean; url: string }
  torch: { available: boolean; url: string }
  default_backend: string
  labels: Record<string, string>
  notes: string[]
  cpu_count: number
}

export interface RheologyProbe {
  model: string
  model_label: string
  yield_stress_pa: number
  viscosity_pa_s: number
  mixture_density_kg_m3: number
  derived_mixture_density_kg_m3: number
  shear_rate_probe: number[]
  apparent_viscosity_probe: number[]
}

export type ExportKind = 'time_series' | 'particles' | 'depth_profile' | 'full'
export type ExportFormat = 'csv' | 'json'
