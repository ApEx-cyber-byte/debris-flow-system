<script setup lang="ts">
/**
 * 参数面板 / Parameter panel.
 *
 * Every physical parameter named in the specification is exposed here with its
 * valid range mirrored from the backend's Pydantic constraints, so the UI can
 * never submit a value the API would reject.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { api } from '@/api/client'
import {
  BACKEND_LABELS,
  RHEOLOGY_LABELS,
  useSimulationStore,
} from '@/stores/simulation'
import type { RheologyModel, SolverBackend } from '@/types/simulation'

const store = useSimulationStore()
const activeSections = ref<string[]>(['slope', 'rheology'])

const rheologyProbe = ref<{ shear_rate_probe: number[]; apparent_viscosity_probe: number[]; model_label: string } | null>(null)
const probePending = ref(false)

const params = computed(() => store.params)

const rheologyOptions = (Object.keys(RHEOLOGY_LABELS) as RheologyModel[]).map((value) => ({
  value,
  label: RHEOLOGY_LABELS[value],
}))

const backendOptions = (Object.keys(BACKEND_LABELS) as SolverBackend[]).map((value) => ({
  value,
  label: BACKEND_LABELS[value],
}))

const showsYieldStress = computed(() =>
  ['bingham', 'herschel_bulkley'].includes(params.value.rheology),
)
const showsCrossParams = computed(() => params.value.rheology === 'cross')
const showsPowerLawIndex = computed(() =>
  ['cross', 'herschel_bulkley', 'power_law'].includes(params.value.rheology),
)

/** Debounced constitutive-law probe used for the viscosity curve. */
let probeTimer: number | undefined
async function refreshProbe(): Promise<void> {
  probePending.value = true
  try {
    rheologyProbe.value = await api.rheologyProbe(params.value)
  } catch {
    rheologyProbe.value = null
  } finally {
    probePending.value = false
  }
}

watch(
  () => [params.value.rheology, params.value.yield_stress_pa, params.value.viscosity_pa_s,
         params.value.power_law_index, params.value.cross_lambda_s, params.value.cross_mu_inf_pa_s],
  () => {
    window.clearTimeout(probeTimer)
    probeTimer = window.setTimeout(refreshProbe, 350)
  },
)

onMounted(() => {
  void refreshProbe()
})

async function confirmReset(): Promise<void> {
  try {
    await ElMessageBox.confirm('恢复所有参数为默认值？', '确认', { type: 'warning' })
    store.resetParams()
  } catch {
    /* user cancelled */
  }
}

function exportNow(kind: 'time_series' | 'particles' | 'depth_profile' | 'full', fmt: 'csv' | 'json') {
  if (!store.runId) {
    ElMessage.warning('请先运行一次仿真')
    return
  }
  api
    .download(store.runId, kind, fmt)
    .then(() => ElMessage.success('导出已开始下载'))
    .catch((error: unknown) => ElMessage.error(`导出失败：${(error as Error).message}`))
}

const labelMap: Record<string, string> = {
  time_series: '时间序列',
  particles: '粒子快照',
  depth_profile: '流深剖面',
  full: '完整记录',
}
</script>

<template>
  <div class="param-panel dfs-scroll">
    <!-- ============================ 运行控制 ============================ -->
    <div class="run-bar">
      <el-button
        type="primary"
        :loading="store.status === 'pending'"
        :disabled="store.isRunning"
        @click="store.start()"
      >
        开始仿真
      </el-button>
      <el-button v-if="store.status === 'running'" @click="store.pause()">暂停</el-button>
      <el-button v-if="store.status === 'paused'" type="warning" @click="store.resume()">
        继续
      </el-button>
      <el-button :disabled="!store.runId || !store.isRunning" @click="store.stop()">
        停止
      </el-button>
    </div>

    <el-progress
      v-if="store.runId"
      :percentage="Math.round(store.progress * 100)"
      :stroke-width="10"
      :status="store.status === 'failed' ? 'exception' : undefined"
      class="run-progress"
    />

    <div v-if="store.isSlowBackend && store.isRunning" class="warn-box">
      <strong>当前后端为粒子级 SPH</strong>，运行速度慢于实时（约
      {{ Math.round(store.estimatedWallSeconds) }} 秒）。如需实时观察，请选择
      「CPU 深度积分浅水方程」。
    </div>

    <!-- ============================== 预设 ============================== -->
    <div class="preset-row">
      <span class="dfs-hint">工况预设</span>
      <el-button-group size="small">
        <el-button @click="store.applyPreset('gentle')">缓坡低速</el-button>
        <el-button @click="store.applyPreset('moderate')">典型泥石流</el-button>
        <el-button @click="store.applyPreset('catastrophic')">陡坡高速</el-button>
      </el-button-group>
    </div>

    <el-collapse v-model="activeSections">
      <!-- =========================== 坡度几何 =========================== -->
      <el-collapse-item name="slope" title="① 坡度几何">
        <el-form label-position="top" size="small">
          <el-form-item label="坡度角 θ (°)　规格要求 0–60">
            <el-slider
              v-model="params.slope_angle_deg"
              :min="0"
              :max="60"
              :step="0.5"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>
          <el-form-item label="坡长 L (m)">
            <el-slider
              v-model="params.slope_length_m"
              :min="1"
              :max="120"
              :step="0.5"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>
          <el-form-item label="释放区沿坡长度 (m)">
            <el-input-number
              v-model="params.release_length_m"
              :min="0.5"
              :max="Math.min(100, params.slope_length_m)"
              :step="0.5"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
          <el-form-item label="释放区宽度 (m)">
            <el-input-number
              v-model="params.channel_width_m"
              :min="0.5"
              :max="100"
              :step="0.5"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
          <el-form-item label="初始体积 (m³)">
            <el-input-number
              v-model="params.initial_volume_m3"
              :min="1"
              :max="50000"
              :step="10"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
          <el-form-item label="初始泥深 (m)　0 = 自动推算">
            <el-input-number
              v-model="params.release_height_m"
              :min="0"
              :max="30"
              :step="0.1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-form>
      </el-collapse-item>

      <!-- =========================== 非牛顿本构 =========================== -->
      <el-collapse-item name="rheology" title="② 非牛顿本构与流变参数">
        <el-form label-position="top" size="small">
          <el-form-item label="本构模型">
            <el-select v-model="params.rheology" style="width: 100%">
              <el-option
                v-for="option in rheologyOptions"
                :key="option.value"
                :label="option.label"
                :value="option.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item v-if="showsYieldStress" label="屈服应力 τy (Pa)">
            <el-slider
              v-model="params.yield_stress_pa"
              :min="0"
              :max="8000"
              :step="10"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="粘性系数 μ (Pa·s)">
            <el-slider
              v-model="params.viscosity_pa_s"
              :min="0.5"
              :max="2000"
              :step="0.5"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item v-if="showsPowerLawIndex" label="流性指数 n">
            <el-slider
              v-model="params.power_law_index"
              :min="0.1"
              :max="1.8"
              :step="0.05"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <template v-if="showsCrossParams">
            <el-form-item label="Cross 时间常数 λ (s)">
              <el-input-number
                v-model="params.cross_lambda_s"
                :min="0.01"
                :max="500"
                :step="0.1"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
            <el-form-item label="Cross 无穷剪切粘度 μ∞ (Pa·s)">
              <el-input-number
                v-model="params.cross_mu_inf_pa_s"
                :min="0"
                :max="5000"
                :step="0.5"
                controls-position="right"
                style="width: 100%"
              />
            </el-form-item>
          </template>

          <el-form-item label="Manning 粗糙度 n (s/m^⅓)">
            <el-slider
              v-model="params.manning_n"
              :min="0"
              :max="0.2"
              :step="0.002"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="体积浓度 Cv">
            <el-slider
              v-model="params.volume_concentration"
              :min="0.05"
              :max="0.75"
              :step="0.01"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="混合体密度 ρ (kg/m³)">
            <el-slider
              v-model="params.density_kg_m3"
              :min="1000"
              :max="3000"
              :step="10"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="基底库仑摩擦系数 μbed">
            <el-slider
              v-model="params.basal_friction_coeff"
              :min="0"
              :max="0.6"
              :step="0.005"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <div v-if="rheologyProbe" class="probe-box">
            <div class="probe-title">{{ rheologyProbe.model_label }}</div>
            <table class="probe-table dfs-mono">
              <thead>
                <tr>
                  <th>剪切速率 (1/s)</th>
                  <th>表观粘度 (Pa·s)</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(g, i) in rheologyProbe.shear_rate_probe"
                  :key="g"
                >
                  <td>{{ g }}</td>
                  <td>{{ rheologyProbe.apparent_viscosity_probe[i].toFixed(1) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else-if="probePending" class="dfs-hint">正在计算本构曲线…</div>
        </el-form>
      </el-collapse-item>

      <!-- =========================== 求解设置 =========================== -->
      <el-collapse-item name="solver" title="③ 求解与推流设置">
        <el-form label-position="top" size="small">
          <el-form-item label="求解后端">
            <el-select v-model="params.backend" style="width: 100%">
              <el-option
                v-for="option in backendOptions"
                :key="option.value"
                :label="option.label"
                :value="option.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="模拟时长 (s)">
            <el-slider
              v-model="params.duration_s"
              :min="1"
              :max="60"
              :step="0.5"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="粒子数量">
            <el-slider
              v-model="params.particle_count"
              :min="500"
              :max="20000"
              :step="500"
              show-input
              :show-input-controls="false"
            />
          </el-form-item>

          <el-form-item label="推帧间隔（每 N 个时间步一帧）">
            <el-input-number
              v-model="params.frame_stride"
              :min="1"
              :max="100"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="前端最大渲染粒子数（降采样上限）">
            <el-input-number
              v-model="store.maxFrameParticles"
              :min="1000"
              :max="20000"
              :step="500"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="CFL 数">
            <el-input-number
              v-model="params.cfl_number"
              :min="0.05"
              :max="0.7"
              :step="0.01"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="坡脚边界条件">
            <el-radio-group v-model="params.boundary" size="small">
              <el-radio-button value="open">开放</el-radio-button>
              <el-radio-button value="wall">设障</el-radio-button>
            </el-radio-group>
          </el-form-item>

          <el-form-item label="冲击力监测断面距坡脚 (m)">
            <el-input-number
              v-model="params.barrier_distance_m"
              :min="0"
              :max="80"
              :step="0.5"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>

          <el-form-item label="随机种子">
            <el-input-number
              v-model="params.seed"
              :min="0"
              :max="999999"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-form>
      </el-collapse-item>

      <!-- ============================= 导出 ============================= -->
      <el-collapse-item name="export" title="④ 数据导出">
        <div class="export-grid">
          <el-button
            v-for="kind in (['time_series', 'particles', 'depth_profile', 'full'] as const)"
            :key="`${kind}-csv`"
            size="small"
            :disabled="!store.runId"
            @click="exportNow(kind, 'csv')"
          >
            {{ labelMap[kind] }} CSV
          </el-button>
          <el-button
            v-for="kind in (['time_series', 'particles', 'full'] as const)"
            :key="`${kind}-json`"
            size="small"
            :disabled="!store.runId"
            @click="exportNow(kind, 'json')"
          >
            {{ labelMap[kind] }} JSON
          </el-button>
        </div>
        <div class="dfs-hint" style="margin-top: 8px">
          粒子快照导出的是最后一帧；时间序列包含流速、流深、动能与冲击力。
        </div>
      </el-collapse-item>
    </el-collapse>

    <div class="panel-footer">
      <el-button size="small" @click="confirmReset">恢复默认参数</el-button>
    </div>
  </div>
</template>

<style scoped>
.param-panel {
  height: 100%;
  padding: 12px 14px 24px;
}

.run-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}

.run-bar :deep(.el-button) {
  flex: 1;
}

.run-progress {
  margin-bottom: 10px;
}

.warn-box {
  padding: 8px 10px;
  margin-bottom: 10px;
  border-radius: 6px;
  font-size: 11.5px;
  line-height: 1.5;
  color: #ffd9a8;
  background: rgba(255, 165, 61, 0.12);
  border: 1px solid rgba(255, 165, 61, 0.35);
}

.preset-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.probe-box {
  margin-top: 4px;
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(66, 214, 245, 0.07);
  border: 1px solid rgba(66, 214, 245, 0.22);
}

.probe-title {
  font-size: 11.5px;
  color: var(--dfs-accent);
  margin-bottom: 6px;
}

.probe-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

.probe-table th,
.probe-table td {
  text-align: right;
  padding: 2px 4px;
  border-bottom: 1px solid rgba(143, 163, 200, 0.12);
  color: var(--dfs-text-dim);
}

.probe-table th {
  color: var(--dfs-text);
  font-weight: 500;
}

.export-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.panel-footer {
  margin-top: 14px;
  display: flex;
  justify-content: flex-end;
}

:deep(.el-collapse-item__title) {
  font-weight: 600;
}
</style>
