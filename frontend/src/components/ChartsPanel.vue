<script setup lang="ts">
/**
 * 曲线面板 / Charts panel.
 *
 * Hosts the three charts the specification requires:
 *   ① 流速时间曲线   ② 流深距离剖面   ③ 冲击力时间曲线
 *
 * Charts are re-optioned on a throttle (4 Hz) rather than per frame: the
 * solver can emit far more keyframes than a human can read, and re-laying out
 * three ECharts instances at 50 Hz would dominate the frame budget.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

import {
  depthProfileOption,
  impactForceOption,
  velocityChartOption,
} from '@/charts/options'
import { useSimulationStore } from '@/stores/simulation'
import type { DecodedFrame } from '@/types/simulation'

const store = useSimulationStore()

const velocityEl = ref<HTMLDivElement | null>(null)
const depthEl = ref<HTMLDivElement | null>(null)
const impactEl = ref<HTMLDivElement | null>(null)

let velocityChart: echarts.ECharts | null = null
let depthChart: echarts.ECharts | null = null
let impactChart: echarts.ECharts | null = null

let unsubscribe: (() => void) | null = null
let latestProfile: { x: number; depth: number }[] = []
let refreshHandle = 0
let lastRefresh = 0
const THROTTLE_MS = 250

const peakSpeed = ref(0)
const peakImpact = ref(0)
const latestMaxDepth = ref(0)
const latestFront = ref(0)

function initCharts(): void {
  if (velocityEl.value) velocityChart = echarts.init(velocityEl.value, undefined, { renderer: 'canvas' })
  if (depthEl.value) depthChart = echarts.init(depthEl.value, undefined, { renderer: 'canvas' })
  if (impactEl.value) impactChart = echarts.init(impactEl.value, undefined, { renderer: 'canvas' })
}

function renderAll(): void {
  const series = store.timeSeries
  const toeX = store.terrain?.toe_x_m ?? store.params.slope_length_m * Math.cos((store.params.slope_angle_deg * Math.PI) / 180)

  velocityChart?.setOption(
    velocityChartOption(series, { max_speed: peakSpeed.value }),
    true,
  )
  impactChart?.setOption(impactForceOption(series), true)
  depthChart?.setOption(depthProfileOption(latestProfile, toeX), true)
}

function scheduleRender(): void {
  const now = performance.now()
  if (now - lastRefresh < THROTTLE_MS) {
    if (refreshHandle) return
    refreshHandle = window.setTimeout(() => {
      refreshHandle = 0
      lastRefresh = performance.now()
      renderAll()
    }, THROTTLE_MS - (now - lastRefresh))
    return
  }
  lastRefresh = now
  renderAll()
}

/** Extract the centreline depth profile from a keyframe. */
function profileFromFrame(frame: DecodedFrame): void {
  const grid = store.grid
  if (!grid) return
  const row = Math.floor(grid.ny / 2)
  const offset = row * grid.nx
  const profile: { x: number; depth: number }[] = []
  for (let i = 0; i < grid.nx; i += 1) {
    const depth = frame.depth[offset + i] ?? 0
    if (depth > 1e-4) {
      profile.push({ x: grid.x0 + (i + 0.5) * grid.dx, depth })
    }
  }
  latestProfile = profile
}

const handleResize = (): void => {
  velocityChart?.resize()
  depthChart?.resize()
  impactChart?.resize()
}

onMounted(() => {
  initCharts()
  renderAll()

  unsubscribe = store.onFrame((frame) => {
    const stats = frame.header.stats
    if (stats) {
      peakSpeed.value = Math.max(peakSpeed.value, stats.max_speed ?? 0)
      peakImpact.value = Math.max(peakImpact.value, (stats.impact_force ?? 0) / 1000)
      latestMaxDepth.value = stats.max_depth ?? 0
      latestFront.value = stats.front_x ?? 0
    }
    profileFromFrame(frame)
    scheduleRender()
  })

  window.addEventListener('resize', handleResize)
})

watch(() => store.runId, () => {
  peakSpeed.value = 0
  peakImpact.value = 0
  latestProfile = []
  renderAll()
})

onBeforeUnmount(() => {
  unsubscribe?.()
  window.removeEventListener('resize', handleResize)
  if (refreshHandle) window.clearTimeout(refreshHandle)
  velocityChart?.dispose()
  depthChart?.dispose()
  impactChart?.dispose()
  velocityChart = depthChart = impactChart = null
})
</script>

<template>
  <div class="charts-panel">
    <div class="charts-summary">
      <div class="summary-item">
        <span class="dfs-stat-label">峰值流速</span>
        <span class="dfs-stat-value dfs-mono">{{ peakSpeed.toFixed(2) }} <em>m/s</em></span>
      </div>
      <div class="summary-item">
        <span class="dfs-stat-label">峰值冲击力</span>
        <span class="dfs-stat-value dfs-mono">{{ peakImpact.toFixed(1) }} <em>kN</em></span>
      </div>
      <div class="summary-item">
        <span class="dfs-stat-label">当前最大泥深</span>
        <span class="dfs-stat-value dfs-mono">{{ latestMaxDepth.toFixed(2) }} <em>m</em></span>
      </div>
      <div class="summary-item">
        <span class="dfs-stat-label">前锋位置</span>
        <span class="dfs-stat-value dfs-mono">{{ latestFront.toFixed(1) }} <em>m</em></span>
      </div>
    </div>

    <div class="chart-block">
      <div class="chart-title">① 流速时间曲线</div>
      <div ref="velocityEl" class="chart-canvas" />
    </div>

    <div class="chart-block">
      <div class="chart-title">② 流深距离剖面（沿程中线）</div>
      <div ref="depthEl" class="chart-canvas" />
    </div>

    <div class="chart-block">
      <div class="chart-title">③ 冲击力时间曲线</div>
      <div ref="impactEl" class="chart-canvas" />
    </div>

    <div v-if="!store.timeSeries.length" class="charts-empty dfs-hint">
      运行仿真后，这里会实时绘制流速、流深与冲击力曲线。
    </div>
  </div>
</template>

<style scoped>
.charts-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px 12px;
  overflow-y: auto;
}

.charts-summary {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--dfs-panel-2);
  border: 1px solid var(--dfs-border);
}

.summary-item em {
  font-style: normal;
  font-size: 10px;
  color: var(--dfs-text-dim);
  margin-left: 2px;
}

.chart-block {
  border: 1px solid var(--dfs-border);
  border-radius: 8px;
  background: var(--dfs-panel-2);
  padding: 6px 8px 2px;
}

.chart-title {
  font-size: 12px;
  color: var(--dfs-text-dim);
  padding: 2px 2px 0;
}

.chart-canvas {
  width: 100%;
  height: 190px;
}

.charts-empty {
  padding: 6px 4px;
}
</style>
