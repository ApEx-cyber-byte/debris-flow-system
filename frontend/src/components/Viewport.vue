<script setup lang="ts">
/**
 * 三维视口 / 3D viewport.
 *
 * Owns the `DebrisFlowScene` lifecycle and subscribes to the store's frame
 * stream. Frames bypass Vue's reactivity entirely (see the store docs) — this
 * component only forwards them into the Three.js buffers.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import { DebrisFlowScene } from '@/three/SceneManager'
import { buildTerrain } from '@/three/terrain'
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()
const host = ref<HTMLDivElement | null>(null)

let scene: DebrisFlowScene | null = null
let unsubscribe: (() => void) | null = null

const options = ref({
  particleSize: 0.32,
  showParticles: true,
  showDepth: true,
  showVectors: false,
  showWireframe: false,
  colorBySpeed: true,
})

onMounted(() => {
  if (!host.value) return
  scene = new DebrisFlowScene(
    host.value,
    store.params.slope_angle_deg,
    store.params.slope_length_m,
    store.params.channel_width_m,
  )

  unsubscribe = store.onFrame((frame) => scene?.updateFrame(frame))

  // Adopt geometry the backend streamed, so the heat-map raster, the terrain
  // mesh and the particles all share one coordinate frame.
  watch(
    () => [store.terrain, store.grid, store.bedRaster] as const,
    ([terrain, grid, bed]) => {
      if (scene && terrain && grid) {
        scene.applyTerrainInfo(terrain, grid, bed ?? null)
      }
    },
    { immediate: true },
  )

  // Rebuild the terrain live while the slope slider moves, before any run.
  watch(
    () => [store.params.slope_angle_deg, store.params.slope_length_m, store.params.channel_width_m],
    () => {
      if (!scene || store.terrain) return // the backend geometry wins once a run starts
      const { slope_angle_deg, slope_length_m, channel_width_m } = store.params
      scene.setTerrain(buildTerrain(slope_angle_deg, slope_length_m, channel_width_m), null)
    },
  )
})

watch(
  options,
  (value) => {
    scene?.setOptions(value)
  },
  { deep: true },
)

watch(
  () => store.runId,
  () => scene?.resetSpeedRange(),
)

function resetView(): void {
  scene?.resetView()
}

function sideView(): void {
  scene?.setSideView()
}

function topView(): void {
  scene?.setTopView()
}

function screenshot(): void {
  if (!scene) return
  const url = scene.captureScreenshot()
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `debris_flow_scene_${stamp}.png`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  ElMessage.success('场景截图已保存为 PNG')
}

onBeforeUnmount(() => {
  unsubscribe?.()
  scene?.dispose()
  scene = null
})

defineExpose({ screenshot })
</script>

<template>
  <div class="viewport">
    <div ref="host" class="viewport-canvas" />

    <div class="viewport-toolbar">
      <el-button-group size="small">
        <el-button @click="resetView">复位视角</el-button>
        <el-button @click="sideView">侧视</el-button>
        <el-button @click="topView">俯视</el-button>
      </el-button-group>
      <el-button size="small" type="primary" plain @click="screenshot">保存截图</el-button>
    </div>

    <div class="viewport-legend">
      <div class="legend-title">流速 (m/s)</div>
      <div class="legend-bar" />
      <div class="legend-scale">
        <span>0</span><span>峰值</span>
      </div>
    </div>

    <div class="viewport-layers">
      <el-checkbox v-model="options.showParticles" size="small" label="SPH 粒子" />
      <el-checkbox v-model="options.showDepth" size="small" label="流深热力图" />
      <el-checkbox v-model="options.showVectors" size="small" label="流速矢量场" />
      <el-checkbox v-model="options.showWireframe" size="small" label="地形网格" />
      <el-checkbox v-model="options.colorBySpeed" size="small" label="按流速着色" />
      <div class="layer-slider">
        <span class="dfs-hint">粒子尺寸</span>
        <el-slider v-model="options.particleSize" :min="0.06" :max="1.2" :step="0.02" size="small" />
      </div>
    </div>

    <div v-if="!store.runId" class="viewport-empty">
      <div class="empty-title">尚未开始仿真</div>
      <div class="dfs-hint">在左侧调整参数后点击「开始仿真」；可旋转、缩放、平移视角。</div>
    </div>
  </div>
</template>

<style scoped>
.viewport {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #070b16;
}

.viewport-canvas {
  position: absolute;
  inset: 0;
}

.viewport-toolbar {
  position: absolute;
  top: 12px;
  right: 12px;
  display: flex;
  gap: 8px;
  z-index: 5;
}

.viewport-legend {
  position: absolute;
  left: 14px;
  bottom: 16px;
  width: 180px;
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(9, 14, 26, 0.78);
  border: 1px solid var(--dfs-border);
  z-index: 5;
}

.legend-title {
  font-size: 11px;
  color: var(--dfs-text-dim);
  margin-bottom: 6px;
}

.legend-bar {
  height: 10px;
  border-radius: 3px;
  background: linear-gradient(
    90deg,
    #2659d9 0%,
    #1ac0d9 25%,
    #4dd95a 50%,
    #f7cc26 75%,
    #f22619 100%
  );
}

.legend-scale {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--dfs-text-dim);
  margin-top: 2px;
}

.viewport-layers {
  position: absolute;
  left: 14px;
  top: 12px;
  z-index: 5;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(9, 14, 26, 0.78);
  border: 1px solid var(--dfs-border);
}

.layer-slider {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 170px;
}

.layer-slider :deep(.el-slider) {
  flex: 1;
}

.viewport-empty {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
  z-index: 4;
}

.empty-title {
  font-size: 16px;
  color: var(--dfs-text);
  margin-bottom: 6px;
}

:deep(.el-checkbox__label) {
  color: var(--dfs-text-dim) !important;
  font-size: 12px !important;
}
</style>
