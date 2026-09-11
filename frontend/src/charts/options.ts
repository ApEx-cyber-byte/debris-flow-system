/**
 * ECharts option builders for the three required charts:
 *
 *  1. 流速时间曲线      — max / mean velocity against time
 *  2. 流深距离剖面      — flow depth against down-slope distance
 *  3. 冲击力时间曲线    — impact force against time
 *
 * Options are plain objects so they can be memoised by the caller; the chart
 * components own the `echarts.init` lifecycle.
 */
import type { EChartsOption } from 'echarts'
import type { TimeSeriesPoint } from '@/types/simulation'

const AXIS_COLOR = '#8fa3c8'
const SPLIT_COLOR = 'rgba(143, 163, 200, 0.14)'
const TEXT_COLOR = '#c7d3ea'

const baseGrid = { left: 62, right: 24, top: 42, bottom: 40 }

const baseTooltip = {
  trigger: 'axis' as const,
  backgroundColor: 'rgba(12, 18, 34, 0.92)',
  borderColor: 'rgba(143, 163, 200, 0.3)',
  textStyle: { color: TEXT_COLOR, fontSize: 12 },
}

function timeAxis(name: string) {
  return {
    type: 'value' as const,
    name,
    nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
    axisLine: { lineStyle: { color: AXIS_COLOR } },
    axisLabel: { color: AXIS_COLOR, fontSize: 11 },
    splitLine: { lineStyle: { color: SPLIT_COLOR } },
  }
}

function valueAxis(name: string) {
  return {
    type: 'value' as const,
    name,
    nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
    axisLine: { lineStyle: { color: AXIS_COLOR } },
    axisLabel: { color: AXIS_COLOR, fontSize: 11 },
    splitLine: { lineStyle: { color: SPLIT_COLOR } },
  }
}

// ---------------------------------------------------------------------------
// 1. 流速时间曲线
// ---------------------------------------------------------------------------
export function velocityChartOption(
  series: TimeSeriesPoint[],
  peaks?: { max_speed?: number },
): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: baseGrid,
    tooltip: baseTooltip,
    legend: {
      data: ['最大流速', '平均流速'],
      textStyle: { color: TEXT_COLOR, fontSize: 11 },
      top: 6,
      itemWidth: 14,
      itemHeight: 8,
    },
    xAxis: timeAxis('时间 t (s)'),
    yAxis: valueAxis('流速 (m/s)'),
    series: [
      {
        name: '最大流速',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: series.map((p) => [p.t, p.max_speed]),
        lineStyle: { width: 2.2, color: '#ff6b57' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(255, 107, 87, 0.35)' },
              { offset: 1, color: 'rgba(255, 107, 87, 0.02)' },
            ],
          },
        },
        markLine: peaks?.max_speed
          ? {
              silent: true,
              symbol: 'none',
              label: {
                formatter: `峰值 ${peaks.max_speed.toFixed(2)} m/s`,
                color: '#ffb3a7',
                fontSize: 11,
              },
              lineStyle: { color: 'rgba(255, 179, 167, 0.6)', type: 'dashed' },
              data: [{ yAxis: peaks.max_speed }],
            }
          : undefined,
      },
      {
        name: '平均流速',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: series.map((p) => [p.t, p.mean_speed]),
        lineStyle: { width: 1.8, color: '#42d6f5' },
      },
    ],
  }
}

// ---------------------------------------------------------------------------
// 2. 流深距离剖面
// ---------------------------------------------------------------------------
export function depthProfileOption(
  profile: { x: number; depth: number }[],
  toeX: number,
): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: baseGrid,
    tooltip: baseTooltip,
    xAxis: timeAxis('沿坡距离 x (m)'),
    yAxis: valueAxis('泥深 h (m)'),
    series: [
      {
        name: '泥深',
        type: 'line',
        smooth: false,
        showSymbol: false,
        step: false,
        data: profile.map((p) => [p.x, p.depth]),
        lineStyle: { width: 2.2, color: '#5ad67d' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(90, 214, 125, 0.42)' },
              { offset: 1, color: 'rgba(90, 214, 125, 0.02)' },
            ],
          },
        },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { formatter: '坡脚', color: '#9fb4d8', fontSize: 11 },
          lineStyle: { color: 'rgba(159, 180, 216, 0.55)', type: 'dashed' },
          data: [{ xAxis: toeX }],
        },
      },
    ],
  }
}

// ---------------------------------------------------------------------------
// 3. 冲击力时间曲线
// ---------------------------------------------------------------------------
export function impactForceOption(series: TimeSeriesPoint[]): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: baseGrid,
    tooltip: baseTooltip,
    legend: {
      data: ['冲击力', '总动能'],
      textStyle: { color: TEXT_COLOR, fontSize: 11 },
      top: 6,
      itemWidth: 14,
      itemHeight: 8,
    },
    xAxis: timeAxis('时间 t (s)'),
    yAxis: [
      valueAxis('冲击力 (kN)'),
      { ...valueAxis('总动能 (kJ)'), splitLine: { show: false } },
    ],
    series: [
      {
        name: '冲击力',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: series.map((p) => [p.t, p.impact_force_kn]),
        lineStyle: { width: 2.2, color: '#ffa53d' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(255, 165, 61, 0.38)' },
              { offset: 1, color: 'rgba(255, 165, 61, 0.02)' },
            ],
          },
        },
      },
      {
        name: '总动能',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        showSymbol: false,
        data: series.map((p) => [p.t, p.kinetic_energy / 1000]),
        lineStyle: { width: 1.6, color: '#8f7dff', type: 'dashed' },
      },
    ],
  }
}

// ---------------------------------------------------------------------------
// 4. 本构探针曲线（参数面板用）
// ---------------------------------------------------------------------------
export function rheologyProbeOption(
  shearRates: number[],
  viscosities: number[],
  label: string,
): EChartsOption {
  return {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 18, top: 34, bottom: 34 },
    tooltip: baseTooltip,
    title: {
      text: label,
      left: 6,
      top: 2,
      textStyle: { color: TEXT_COLOR, fontSize: 12, fontWeight: 'normal' },
    },
    xAxis: { ...timeAxis('剪切速率 (1/s)'), type: 'log' as const },
    yAxis: { ...valueAxis('表观粘度 (Pa·s)'), type: 'log' as const },
    series: [
      {
        name: '表观粘度',
        type: 'line',
        smooth: true,
        showSymbol: true,
        symbolSize: 6,
        data: shearRates.map((g, i) => [g, viscosities[i]]),
        lineStyle: { width: 2, color: '#42d6f5' },
        itemStyle: { color: '#7fe6ff' },
      },
    ],
  }
}
