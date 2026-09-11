/**
 * WebSocket client for the keyframe stream.
 *
 * Wire format (see `backend/app/services/frames.py`):
 *
 *   [uint32 LE header length][UTF-8 JSON header][float32 arrays, in header order]
 *
 * Text messages carry control events (completed / error / pong).
 */
import type { DecodedFrame, FrameHeader } from '@/types/simulation'

export interface StreamHandlers {
  onMeta?: (header: FrameHeader, bed: Float32Array) => void
  onFrame?: (frame: DecodedFrame) => void
  onCompleted?: (header: FrameHeader) => void
  onError?: (message: string) => void
  onOpen?: () => void
  onClose?: () => void
}

/**
 * Downsample a decoded frame to at most `maxParticles` points.
 *
 * The backend already strides large runs, but a user may still open the page on
 * a machine that cannot push 6 000 particles at 50 fps — this second stage of
 * downsampling is what makes the viewport survive on modest hardware.
 */
export function downsample(frame: DecodedFrame, maxParticles: number): DecodedFrame {
  const count = frame.positions.length / 3
  if (count <= maxParticles) return frame

  const stride = Math.ceil(count / maxParticles)
  const target = Math.floor((count + stride - 1) / stride)

  const positions = new Float32Array(target * 3)
  const velocities = new Float32Array(target * 3)
  let cursor = 0
  for (let i = 0; i < count && cursor < target; i += stride) {
    positions[cursor * 3] = frame.positions[i * 3]
    positions[cursor * 3 + 1] = frame.positions[i * 3 + 1]
    positions[cursor * 3 + 2] = frame.positions[i * 3 + 2]
    velocities[cursor * 3] = frame.velocities[i * 3]
    velocities[cursor * 3 + 1] = frame.velocities[i * 3 + 1]
    velocities[cursor * 3 + 2] = frame.velocities[i * 3 + 2]
    cursor += 1
  }

  return { ...frame, positions, velocities }
}

export class SimulationStream {
  private socket: WebSocket | null = null
  private bed: Float32Array | null = null
  private closedByCaller = false

  constructor(
    private readonly runId: string,
    private readonly handlers: StreamHandlers,
  ) {}

  connect(): void {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${scheme}://${window.location.host}/ws/simulations/${this.runId}`

    this.closedByCaller = false
    const socket = new WebSocket(url)
    socket.binaryType = 'arraybuffer'
    this.socket = socket

    socket.onopen = () => this.handlers.onOpen?.()

    socket.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
      if (typeof event.data === 'string') {
        this.handleText(event.data)
        return
      }
      this.handleBinary(event.data)
    }

    socket.onerror = () => this.handlers.onError?.('WebSocket 连接错误')
    socket.onclose = () => {
      this.socket = null
      this.handlers.onClose?.()
    }
  }

  private handleText(raw: string): void {
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(raw)
    } catch {
      return
    }
    const type = payload.type as string
    if (type === 'completed') {
      this.handlers.onCompleted?.(payload as unknown as FrameHeader)
    } else if (type === 'error') {
      this.handlers.onError?.(String(payload.message ?? '仿真失败'))
    }
  }

  private handleBinary(buffer: ArrayBuffer): void {
    const view = new DataView(buffer)
    const headerLength = view.getUint32(0, true)
    const headerStart = 4
    const headerEnd = headerStart + headerLength
    const header = JSON.parse(
      new TextDecoder('utf-8').decode(new Uint8Array(buffer, headerStart, headerLength)),
    ) as FrameHeader & { arrays: { name: string; count: number }[] }

    // The float32 payload starts after the header, and every array is
    // 4-byte aligned by construction, so a direct typed-array view is safe.
    let offset = headerEnd
    const arrays: Record<string, Float32Array> = {}
    for (const descriptor of header.arrays ?? []) {
      arrays[descriptor.name] = new Float32Array(buffer, offset, descriptor.count)
      offset += descriptor.count * 4
    }

    if (header.type === 'meta') {
      this.bed = arrays.bed ?? null
      this.handlers.onMeta?.(header, this.bed ?? new Float32Array(0))
      return
    }

    if (header.type !== 'frame') return

    this.handlers.onFrame?.({
      header,
      positions: arrays.pos ?? new Float32Array(0),
      velocities: arrays.vel ?? new Float32Array(0),
      depth: arrays.depth ?? new Float32Array(0),
      velocityX: arrays.vx ?? new Float32Array(0),
      velocityY: arrays.vy ?? new Float32Array(0),
      receivedAt: performance.now(),
    })
  }

  send(message: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message))
    }
  }

  pause(): void {
    this.send({ type: 'pause' })
  }

  resume(): void {
    this.send({ type: 'resume' })
  }

  stop(): void {
    this.send({ type: 'stop' })
  }

  close(): void {
    this.closedByCaller = true
    this.socket?.close()
    this.socket = null
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  get wasClosedByCaller(): boolean {
    return this.closedByCaller
  }
}
