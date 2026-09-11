/**
 * REST client for the FastAPI backend.
 *
 * All requests go through `request()` so error handling, timeouts and the
 * base URL stay in one place. The dev server proxies `/api` to :8000.
 */
import type {
  AIPredictResponse,
  BackendReport,
  ExportFormat,
  ExportKind,
  RheologyProbe,
  RunResult,
  RunStatus,
  SimulationCreated,
  SimulationParams,
  SimulationSummary,
} from '@/types/simulation'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    })

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const body = await response.json()
        if (body?.detail) {
          detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
        }
      } catch {
        /* non-JSON error body — keep the status line */
      }
      throw new ApiError(detail, response.status)
    }

    if (response.status === 204) return undefined as T
    return (await response.json()) as T
  } finally {
    window.clearTimeout(timer)
  }
}

// ---------------------------------------------------------------------------
// Simulation lifecycle
// ---------------------------------------------------------------------------
export const api = {
  health: () => request<{ status: string; runs: number }>('/health'),

  backends: () => request<BackendReport>('/api/system/backends'),

  defaults: () => request<SimulationParams>('/api/system/defaults'),

  rheologyProbe: (params: SimulationParams) =>
    request<RheologyProbe>('/api/system/rheology', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  /** Create and immediately start a run. */
  createSimulation: (params: SimulationParams) =>
    request<SimulationCreated>('/api/simulations', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  listSimulations: () => request<SimulationSummary[]>('/api/simulations'),

  getSimulation: (runId: string) => request<SimulationSummary>(`/api/simulations/${runId}`),

  pause: (runId: string) =>
    request<SimulationSummary>(`/api/simulations/${runId}/pause`, { method: 'POST' }),

  resume: (runId: string) =>
    request<SimulationSummary>(`/api/simulations/${runId}/resume`, { method: 'POST' }),

  stop: (runId: string) =>
    request<SimulationSummary>(`/api/simulations/${runId}/stop`, { method: 'POST' }),

  timeSeries: (runId: string) =>
    request<{ run_id: string; toe_x_m: number; profile: { x: number; depth: number }[] }>(
      `/api/simulations/${runId}/depth-profile`,
    ),

  // -------------------------------------------------------------------------
  // AI surrogate
  // -------------------------------------------------------------------------
  aiModels: () =>
    request<{
      models: { name: string; label: string; kind: string; description: string; available: boolean }[]
      default: string
      torch_available: boolean
    }>('/api/ai/models'),

  aiPredict: (params: SimulationParams, modelName = 'heuristic-v1') =>
    request<AIPredictResponse>('/api/ai/predict', {
      method: 'POST',
      body: JSON.stringify({ params, model_name: modelName }),
    }),

  // -------------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------------
  exportUrl: (runId: string, kind: ExportKind, fmt: ExportFormat, frameIndex?: number) => {
    const query = new URLSearchParams({ kind, fmt })
    if (frameIndex !== undefined) query.set('frame_index', String(frameIndex))
    return `${BASE}/api/simulations/${runId}/export?${query.toString()}`
  },

  /** Trigger a browser download without navigating away from the SPA. */
  download: async (
    runId: string,
    kind: ExportKind,
    fmt: ExportFormat,
    frameIndex?: number,
  ): Promise<void> => {
    const response = await fetch(api.exportUrl(runId, kind, fmt, frameIndex))
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const body = await response.json()
        if (body?.detail) detail = String(body.detail)
      } catch {
        /* keep status line */
      }
      throw new ApiError(detail, response.status)
    }

    const blob = await response.blob()
    const disposition = response.headers.get('Content-Disposition') ?? ''
    const match = /filename\*=UTF-8''([^;]+)/i.exec(disposition)
    const filename = match
      ? decodeURIComponent(match[1])
      : `debris_flow_${runId}_${kind}.${fmt}`

    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  },

  status: (runId: string) => request<RunStatus>(`/api/simulations/${runId}/status`),

  result: (runId: string) => request<RunResult>(`/api/simulations/${runId}/result`),
}
