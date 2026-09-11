<script setup lang="ts">
/**
 * 应用外壳 / Application shell.
 *
 * Layout: parameter panel (left) | 3D viewport (centre) | charts + AI (right),
 * with a status bar along the bottom.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import AiPanel from '@/components/AiPanel.vue'
import ChartsPanel from '@/components/ChartsPanel.vue'
import ParameterPanel from '@/components/ParameterPanel.vue'
import Viewport from '@/components/Viewport.vue'
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()
const rightTab = ref<'charts' | 'ai' | 'log'>('charts')
const showLeft = ref(true)
const showRight = ref(true)

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    idle: '待机',
    pending: '创建中',
    running: '运行中',
    paused: '已暂停',
    finished: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return map[store.status] ?? store.status
})

const statusType = computed(() => {
  if (store.status === 'failed') return 'danger'
  if (store.status === 'running') return 'success'
  if (store.status === 'paused') return 'warning'
  return 'info'
})

const stats = computed(() => [
  { label: '模拟时间', value: `${store.simulatedTime.toFixed(2)} s` },
  { label: '已收帧数', value: String(store.frameCount) },
  { label: '最大流速', value: `${(store.stats.max_speed ?? 0).toFixed(2)} m/s` },
  { label: '最大泥深', value: `${(store.stats.max_depth ?? 0).toFixed(2)} m` },
  { label: '冲击力', value: `${((store.stats.impact_force ?? 0) / 1000).toFixed(1)} kN` },
  { label: '前锋位置', value: `${(store.stats.front_x ?? 0).toFixed(1)} m` },
])

onMounted(async () => {
  await Promise.all([store.loadBackends(), store.loadDefaults()])
})

onBeforeUnmount(() => {
  store.dispose()
})
</script>

<template>
  <div class="app-shell">
    <!-- ============================== 顶栏 ============================== -->
    <header class="app-header">
      <div class="brand">
        <div class="brand-mark" />
        <div>
          <div class="brand-title">基于流体动力学的泥石流坡度滚落物理动态分析系统</div>
          <div class="brand-sub">
            Debris-Flow Slope Runout Physical Dynamics Analysis · SPH + 深度积分浅水方程 ·
            非牛顿本构
          </div>
        </div>
      </div>

      <div class="header-right">
        <el-tag :type="statusType" size="small" effect="dark">{{ statusLabel }}</el-tag>
        <el-tag v-if="store.backend" size="small" type="info" effect="plain">
          {{ store.backend.active }}{{ store.backend.accelerated ? ' · 加速' : ' · CPU' }}
        </el-tag>
        <el-button size="small" text @click="showLeft = !showLeft">
          {{ showLeft ? '隐藏参数' : '显示参数' }}
        </el-button>
        <el-button size="small" text @click="showRight = !showRight">
          {{ showRight ? '隐藏图表' : '显示图表' }}
        </el-button>
      </div>
    </header>

    <div v-if="store.errorMessage" class="error-bar">
      {{ store.errorMessage }}
    </div>

    <!-- ============================== 主体 ============================== -->
    <main class="app-body">
      <aside v-if="showLeft" class="app-left dfs-panel">
        <div class="dfs-panel-title">参数控制台</div>
        <div class="left-scroll">
          <ParameterPanel />
        </div>
      </aside>

      <section class="app-center">
        <Viewport />
      </section>

      <aside v-if="showRight" class="app-right dfs-panel">
        <el-tabs v-model="rightTab" class="right-tabs">
          <el-tab-pane label="曲线" name="charts">
            <ChartsPanel />
          </el-tab-pane>
          <el-tab-pane label="AI 预测" name="ai">
            <AiPanel />
          </el-tab-pane>
          <el-tab-pane label="日志" name="log">
            <div class="log-view dfs-scroll dfs-mono">
              <div v-for="(message, index) in store.messages" :key="index" class="log-line">
                {{ message }}
              </div>
              <div v-if="!store.messages.length" class="dfs-hint">暂无日志</div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </aside>
    </main>

    <!-- ============================== 状态栏 ============================== -->
    <footer class="app-footer">
      <div v-for="item in stats" :key="item.label" class="footer-stat">
        <span class="dfs-stat-label">{{ item.label }}</span>
        <span class="footer-value dfs-mono">{{ item.value }}</span>
      </div>

      <div v-if="store.backendReport" class="footer-backends">
        <span
          v-for="(label, key) in { PySPH: store.backendReport.pysph.available, SynxFlow: store.backendReport.synxflow.available, PyTorch: store.backendReport.torch.available }"
          :key="String(key)"
          class="backend-chip"
          :class="{ on: label }"
          :title="label ? '可用' : '未安装，已自动回退到 CPU 求解器'"
        >
          {{ key }}
        </span>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--dfs-bg);
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--dfs-border);
  background: linear-gradient(180deg, #0e1730 0%, #0a1122 100%);
  flex: 0 0 auto;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.brand-mark {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  flex: 0 0 auto;
  background: linear-gradient(135deg, #42d6f5 0%, #7b5cff 55%, #ff6b57 100%);
}

.brand-title {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.3px;
  color: var(--dfs-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.brand-sub {
  font-size: 10.5px;
  color: var(--dfs-text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.error-bar {
  padding: 8px 16px;
  font-size: 12px;
  color: #ffc9c0;
  background: rgba(255, 107, 87, 0.14);
  border-bottom: 1px solid rgba(255, 107, 87, 0.35);
}

.app-body {
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
  gap: 10px;
  padding: 10px;
}

.app-left {
  width: 336px;
  flex: 0 0 336px;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.left-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
}

.app-center {
  flex: 1 1 auto;
  min-width: 0;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--dfs-border);
}

.app-right {
  width: 400px;
  flex: 0 0 400px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.right-tabs {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.right-tabs :deep(.el-tabs__header) {
  margin: 0;
  padding: 0 10px;
  border-bottom: 1px solid var(--dfs-border);
}

.right-tabs :deep(.el-tabs__content) {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
}

.right-tabs :deep(.el-tab-pane) {
  height: 100%;
}

.right-tabs :deep(.el-tabs__item) {
  color: var(--dfs-text-dim);
  font-size: 12.5px;
}

.right-tabs :deep(.el-tabs__item.is-active) {
  color: var(--dfs-accent);
}

.log-view {
  height: 100%;
  padding: 10px 12px;
  font-size: 11px;
  line-height: 1.65;
  color: var(--dfs-text-dim);
}

.log-line {
  padding: 2px 0;
  border-bottom: 1px dashed rgba(143, 163, 200, 0.1);
  word-break: break-all;
}

.app-footer {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 6px 16px;
  border-top: 1px solid var(--dfs-border);
  background: #0a1122;
  overflow-x: auto;
}

.footer-stat {
  display: flex;
  align-items: baseline;
  gap: 6px;
  white-space: nowrap;
}

.footer-value {
  font-size: 12.5px;
  color: var(--dfs-text);
  font-weight: 600;
}

.footer-backends {
  margin-left: auto;
  display: flex;
  gap: 6px;
}

.backend-chip {
  font-size: 10.5px;
  padding: 1px 7px;
  border-radius: 10px;
  border: 1px solid rgba(143, 163, 200, 0.25);
  color: var(--dfs-text-dim);
  cursor: help;
}

.backend-chip.on {
  color: #7ef0b0;
  border-color: rgba(126, 240, 176, 0.45);
  background: rgba(126, 240, 176, 0.1);
}
</style>
