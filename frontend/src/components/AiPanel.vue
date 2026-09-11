<script setup lang="ts">
/**
 * AI 预测面板 / AI surrogate panel.
 *
 * Calls `/api/ai/predict`. The backend serves a physics-informed surrogate by
 * default and will switch to a trained CNN-LSTM / DR-LSTM automatically once
 * weights are present in `backend/weights/`; this panel reports which one
 * actually answered so the distinction is never hidden from the user.
 */
import { onMounted, ref } from 'vue'

import { api } from '@/api/client'
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()

interface ModelInfo {
  name: string
  label: string
  kind: string
  description: string
  available: boolean
}

const models = ref<ModelInfo[]>([])
const selected = ref('heuristic-v1')
const torchAvailable = ref(false)

onMounted(async () => {
  try {
    const info = await api.aiModels()
    models.value = info.models
    selected.value = info.default
    torchAvailable.value = info.torch_available
  } catch {
    models.value = []
  }
})

const metrics: { key: keyof NonNullable<typeof store.aiPrediction>; label: string; unit: string }[] = [
  { key: 'runout_distance_m', label: '预测跑距', unit: 'm' },
  { key: 'max_velocity_m_s', label: '峰值流速', unit: 'm/s' },
  { key: 'arrival_time_s', label: '到达监测断面时间', unit: 's' },
  { key: 'max_flow_depth_m', label: '最大泥深', unit: 'm' },
  { key: 'peak_impact_force_kn', label: '峰值冲击力', unit: 'kN' },
  { key: 'affected_area_m2', label: '影响范围面积', unit: 'm²' },
]

function display(value: unknown): string {
  return typeof value === 'number' ? value.toFixed(2) : '—'
}
</script>

<template>
  <div class="ai-panel">
    <div class="ai-header">
      <el-select v-model="selected" size="small" style="flex: 1">
        <el-option
          v-for="model in models"
          :key="model.name"
          :label="`${model.label}${model.available ? '' : '（未就绪）'}`"
          :value="model.name"
          :disabled="!model.available"
        />
      </el-select>
      <el-button
        type="primary"
        size="small"
        :loading="store.aiPending"
        @click="store.predict()"
      >
        快速预测
      </el-button>
    </div>

    <div v-if="!torchAvailable" class="ai-note dfs-hint">
      PyTorch 未安装，学习型代理模型暂不可用；当前由<b>物理启发式代理</b>作答
      （与数值求解器共用同一套非牛顿本构与基底阻力公式，毫秒级返回）。
    </div>

    <div v-if="store.aiPrediction" class="ai-results">
      <div class="ai-model-line">
        回答模型：<b>{{ store.aiPrediction.model_name }}</b>
        <el-tag
          size="small"
          :type="store.aiPrediction.is_surrogate_trained ? 'success' : 'info'"
          style="margin-left: 6px"
        >
          {{ store.aiPrediction.is_surrogate_trained ? '已训练' : '物理代理' }}
        </el-tag>
        <span class="dfs-hint" style="margin-left: 6px">
          置信度 {{ (store.aiPrediction.confidence * 100).toFixed(0) }}%
        </span>
      </div>

      <div class="ai-grid">
        <div v-for="item in metrics" :key="String(item.key)" class="ai-metric">
          <span class="dfs-stat-label">{{ item.label }}</span>
          <span class="dfs-stat-value dfs-mono">
            {{ display(store.aiPrediction[item.key]) }}
            <em>{{ item.unit }}</em>
          </span>
        </div>
      </div>

      <div v-for="(note, index) in store.aiPrediction.notes" :key="index" class="dfs-hint">
        · {{ note }}
      </div>
    </div>

    <div v-else class="dfs-hint">
      尚未预测。点击「快速预测」可在不运行完整数值模拟的情况下，得到跑距、峰值流速、
      到达时间与冲击力的量级估计。
    </div>
  </div>
</template>

<style scoped>
.ai-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px 12px;
  height: 100%;
  overflow-y: auto;
}

.ai-header {
  display: flex;
  gap: 8px;
}

.ai-note {
  padding: 7px 9px;
  border-radius: 6px;
  background: rgba(66, 214, 245, 0.08);
  border: 1px solid rgba(66, 214, 245, 0.22);
  color: #a9dfee;
}

.ai-model-line {
  font-size: 12px;
  color: var(--dfs-text-dim);
  margin-bottom: 8px;
}

.ai-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 10px;
}

.ai-metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--dfs-panel-2);
  border: 1px solid var(--dfs-border);
}

.ai-metric em {
  font-style: normal;
  font-size: 10px;
  color: var(--dfs-text-dim);
  margin-left: 2px;
}
</style>
