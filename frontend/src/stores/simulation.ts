/**
 * Simulation store — owns the parameter set, the run lifecycle and the live
 * keyframe stream.
 *
 * Performance note: frames arrive at tens of hertz. Routing them through Vue's
 * reactivity would re-render the whole tree on every push, so decoded frames
 * go straight to plain callbacks and only *summary* values (counters, the
 * latest statistics, the time series) are reactive. The Three.js viewport
 * subscribes with `onFrame()`; the charts read the throttled reactive state.
 */
import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import { api, ApiError } from '@/api/client'
import { SimulationStream, downsample } from '@/api/ws'
import type {
  AIPredictResponse,
  BackendInfo,
  BackendReport,
  DecodedFrame,
  GridInfo,
  RheologyModel,
  RunStatus,
  SimulationParams,
  TerrainInfo,
  TimeSeriesPoint,
} from '@/types/simulation'

export type UiStatus = RunStatus | 'idle'

const DEFAULT_PARAMS: SimulationParams = {
  slope_angle_deg: 30,
  slope_length_m: 20,
  rheology: 'bingham',
  density_kg_m3: 2200,
  volume_concentration: 0.45,
  yield_stress_pa: 500,
  viscosity_pa_s: 60,
  manning_n: 0.04,
  power_law_index: 0.6,
  cross_lambda_s: 2.0,
  cross_mu_inf_pa_s: 5.0,
  max_apparent_viscosity_pa_s: 20000,
  basal_friction_coeff: 0.12,
  turbulence_coeff: 0.0,
  initial_volume_m3: 150,
  release_length_m: 8,
  channel_width_m: 8,
  release_height_m: 0,
  duration_s: 12,
  particle_count: 4000,
  backend: 'auto',
  frame_stride: 10,
  cfl_number: 0.25,
  seed: 42,
  barrier_distance_m: 3.0,
  boundary: 'open',
}

export const RHEOLOGY_LABELS: Record<RheologyModel, string> = {
  bingham: '宾汉体 Bingham（Papanastasiou 正则化）',
  cross: 'Cross 四参数模型',
  herschel_bulkley: 'Herschel-Bulkley 屈服幂律',
  power_law: '幂律 Power-Law（无屈服应力）',
}

export const BACKEND_LABELS: Record<string, string> = {
  auto: '自动（探测 PySPH / SynxFlow，否则 CPU）',
  shallow_water: 'CPU 深度积分浅水方程（SynxFlow 的 CPU 替代，实时）',
  numpy_sph: '内置 NumPy SPH（完整非牛顿本构，精度高、较慢）',
  pysph: 'PySPH（需安装 PySPH）',
  synxflow: 'SynxFlow（需安装 SynxFlow + CUDA GPU）',
}

/** Backends that run far slower than real time, with a UI warning. */
export const SLOW_BACKENDS = ['numpy_sph', 'pysph']

export const useSimulationStore = defineStore('simulation', () => {
  const params = ref<SimulationParams>({ ...DEFAULT_PARAMS })
  const backendReport = ref<BackendReport | null>(null)
  const status = ref<UiStatus>('idle')
  const runId = ref<string | null>(null)
  const terrain = ref<TerrainInfo | null>(null)
  const grid = ref<GridInfo | null>(null)
  const backend = ref<BackendInfo | null>(null)
  const bedRaster = shallowRef<Float32Array | null>(null)

  const timeSeries = ref<TimeSeriesPoint[]>([])
  const frameCount = ref(0)
  const simulatedTime = ref(0)
  const progress = ref(0)
  const stats = ref<Record<string, number>>({})
  const messages = ref<string[]>([])
  const errorMessage = ref<string | null>(null)

  const aiPrediction = ref<AIPredictResponse | null>(null)
  const aiPending = ref(false)

  const maxFrameParticles = ref(6000)

  // --- live stream plumbing (non-reactive) --------------------------------
  const frameSubscribers = new Set<(frame: DecodedFrame) => void>()
  let stream: SimulationStream | null = null
  let pendingFrame: DecodedFrame | null = null
  let rafHandle = 0

  function onFrame(callback: (frame: DecodedFrame) => void): () => void {
    frameSubscribers.add(callback)
    return () => frameSubscribers.delete(callback)
  }

  function log(message: string): void {
    const stamp = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    messages.value = [`[${stamp}] ${message}`, ...messages.value].slice(0, 200)
  }

  // Frames are coalesced to one per animation frame: the solver can outrun the
  // display, and rendering stale intermediate frames is wasted work.
  function scheduleFrame(frame: DecodedFrame): void {
    pendingFrame = frame
    if (rafHandle) return
    rafHandle = requestAnimationFrame(() => {
      rafHandle = 0
      const frameToDraw = pendingFrame
      pendingFrame = null
      if (!frameToDraw) return
      frameSubscribers.forEach((callback) => {
        try {
          callback(frameToDraw)
        } catch (error) {
          console.error('[simulation] frame subscriber failed', error)
        }
      })
    })
  }

  // ------------------------------------------------------------------
  const isRunning = computed(
    () => status.value === 'running' || status.value === 'pending' || status.value === 'paused',
  )
  const isSlowBackend = computed(() => {
    const active = backend.value?.active ?? params.value.backend
    return SLOW_BACKENDS.includes(active)
  })

  /** Rough wall-clock estimate so a slow SPH run is not a surprise. */
  const estimatedWallSeconds = computed(() => {
    const active = backend.value?.active ?? params.value.backend
    if (active === 'numpy_sph') return params.value.duration_s * 55
    if (active === 'pysph') return params.value.duration_s * 1.5
    if (active === 'synxflow') return params.value.duration_s * 0.2
    return params.value.duration_s * 1.0
  })

  // ------------------------------------------------------------------
  async function loadBackends(): Promise<void> {
    try {
      backendReport.value = await api.backends()
      const missing = backendReport.value.notes
      missing.forEach((note) => log(note))
    } catch (error) {
      log(`无法获取后端信息: ${describeError(error)}`)
    }
  }

  async function loadDefaults(): Promise<void> {
    try {
      params.value = { ...(await api.defaults()) }
    } catch (error) {
      log(`使用前端内置默认参数（后端默认值获取失败: ${describeError(error)}）`)
    }
  }

  async function start(): Promise<void> {
    if (isRunning.value) {
      log('已有仿真在运行，请先停止')
      return
    }
    errorMessage.value = null
    timeSeries.value = []
    frameCount.value = 0
    simulatedTime.value = 0
    progress.value = 0
    stats.value = {}
    bedRaster.value = null

    try {
      status.value = 'pending'
      const created = await api.createSimulation(params.value)
      runId.value = created.run_id
      terrain.value = created.terrain
      grid.value = created.grid
      backend.value = created.backend
      log(`任务 ${created.run_id} 已创建，求解后端 = ${created.backend.active}`)
      created.backend.notes.forEach((note) => log(note))
      attachStream(created.run_id)
    } catch (error) {
      status.value = 'failed'
      errorMessage.value = describeError(error)
      log(`创建仿真失败: ${errorMessage.value}`)
    }
  }

  function attachStream(id: string): void {
    stream?.close()
    stream = new SimulationStream(id, {
      onOpen: () => {
        status.value = 'running'
        log('WebSocket 已连接，开始接收关键帧')
      },
      onMeta: (header, bed) => {
        if (header.terrain) terrain.value = header.terrain
        if (header.grid) grid.value = header.grid
        if (header.backend) backend.value = header.backend
        if (bed && bed.length) bedRaster.value = bed
        log(`握手完成：${header.n_particles ?? 0} 个粒子，栅格 ${header.grid?.nx}×${header.grid?.ny}`)
      },
      onFrame: (frame) => {
        const reduced = downsample(frame, maxFrameParticles.value)
        frameCount.value += 1
        simulatedTime.value = frame.header.time ?? 0
        const total = params.value.duration_s
        progress.value = total > 0 ? Math.min(1, simulatedTime.value / total) : 0
        stats.value = frame.header.stats ?? {}
        appendTimeSeries(frame)
        scheduleFrame(reduced)
      },
      onCompleted: (header) => {
        status.value = 'finished'
        progress.value = 1
        const metrics = (header as unknown as { metrics?: Record<string, number> }).metrics
        if (metrics) {
          log(
            `仿真完成：峰值流速 ${(metrics.max_speed ?? 0).toFixed(2)} m/s，` +
              `跑距 ${(metrics.runout_distance_m ?? 0).toFixed(2)} m，` +
              `峰值冲击力 ${(metrics.max_impact_force_kn ?? 0).toFixed(1)} kN，` +
              `用时 ${(metrics.wall_time_s ?? 0).toFixed(1)} s`,
          )
        } else {
          log('仿真完成')
        }
      },
      onError: (message) => {
        status.value = 'failed'
        errorMessage.value = message
        log(`仿真错误: ${message}`)
      },
      onClose: () => {
        if (status.value === 'running') status.value = 'finished'
      },
    })
    stream.connect()
  }

  function appendTimeSeries(frame: DecodedFrame): void {
    const s = frame.header.stats
    if (!s) return
    timeSeries.value.push({
      t: frame.header.time ?? 0,
      max_speed: s.max_speed ?? 0,
      mean_speed: s.mean_speed ?? 0,
      front_x: s.front_x ?? 0,
      max_depth: s.max_depth ?? 0,
      kinetic_energy: s.kinetic_energy ?? 0,
      impact_force_kn: (s.impact_force ?? 0) / 1000,
      deposited_volume: s.deposited_volume ?? 0,
    })
  }

  async function pause(): Promise<void> {
    if (!runId.value) return
    stream?.pause()
    status.value = 'paused'
    log('已暂停')
  }

  async function resume(): Promise<void> {
    if (!runId.value) return
    stream?.resume()
    status.value = 'running'
    log('已继续')
  }

  async function stop(): Promise<void> {
    if (!runId.value) return
    stream?.stop()
    status.value = 'cancelled'
    log('已请求停止')
  }

  async function predict(): Promise<void> {
    aiPending.value = true
    try {
      aiPrediction.value = await api.aiPredict(params.value)
      log(
        `AI 预测（${aiPrediction.value.model_name}）：跑距 ` +
          `${aiPrediction.value.runout_distance_m.toFixed(1)} m，` +
          `峰值流速 ${aiPrediction.value.max_velocity_m_s.toFixed(1)} m/s，` +
          `置信度 ${(aiPrediction.value.confidence * 100).toFixed(0)}%`,
      )
    } catch (error) {
      log(`AI 预测失败: ${describeError(error)}`)
    } finally {
      aiPending.value = false
    }
  }

  function resetParams(): void {
    params.value = { ...DEFAULT_PARAMS }
    log('参数已恢复默认值')
  }

  function applyPreset(name: 'gentle' | 'moderate' | 'catastrophic'): void {
    const presets: Record<string, Partial<SimulationParams>> = {
      gentle: {
        slope_angle_deg: 15,
        manning_n: 0.06,
        yield_stress_pa: 1200,
        viscosity_pa_s: 120,
        initial_volume_m3: 80,
      },
      moderate: {
        slope_angle_deg: 30,
        manning_n: 0.04,
        yield_stress_pa: 500,
        viscosity_pa_s: 60,
        initial_volume_m3: 150,
      },
      catastrophic: {
        slope_angle_deg: 45,
        manning_n: 0.025,
        yield_stress_pa: 200,
        viscosity_pa_s: 25,
        initial_volume_m3: 400,
      },
    }
    params.value = { ...params.value, ...presets[name] }
    log(`已应用预设：${{ gentle: '缓坡低速', moderate: '典型泥石流', catastrophic: '陡坡高速' }[name]}`)
  }

  function dispose(): void {
    stream?.close()
    stream = null
    frameSubscribers.clear()
    if (rafHandle) cancelAnimationFrame(rafHandle)
  }

  return {
    // state
    params,
    backendReport,
    status,
    runId,
    terrain,
    grid,
    backend,
    bedRaster,
    timeSeries,
    frameCount,
    simulatedTime,
    progress,
    stats,
    messages,
    errorMessage,
    aiPrediction,
    aiPending,
    maxFrameParticles,
    // derived
    isRunning,
    isSlowBackend,
    estimatedWallSeconds,
    // actions
    onFrame,
    loadBackends,
    loadDefaults,
    start,
    pause,
    resume,
    stop,
    predict,
    resetParams,
    applyPreset,
    dispose,
    log,
  }
})

function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return String(error)
}
