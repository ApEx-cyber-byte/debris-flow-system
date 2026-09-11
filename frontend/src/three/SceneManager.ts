/**
 * Three.js viewport: terrain, SPH particles, depth heat-map and velocity field.
 *
 * Design notes
 * ------------
 * * **Zero-copy-ish updates.** Particle positions arrive as `Float32Array`s and
 *   are written straight into pre-allocated `BufferAttribute`s. The geometry is
 *   only re-allocated when the particle count actually changes, so a steady
 *   stream of frames does no garbage generation.
 * * **Colour by speed.** The particle shader maps speed to a blue→yellow→red
 *   ramp, which is what makes the acceleration down the slope legible.
 * * **Depth heat-map.** The gridded column height is uploaded as a DataTexture
 *   and drawn on a plane that lies exactly on the bed, so it lines up with the
 *   particles and the terrain mesh (both driven by the same bed raster).
 * * **Velocity field.** Gridded velocity is drawn as an instanced arrow field,
 *   decimated so the arrows stay readable.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { DecodedFrame, GridInfo, TerrainInfo } from '@/types/simulation'
import { bedHeight, buildTerrain, terrainFromInfo, type TerrainLike } from './terrain'

const SPEED_RAMP_STOPS: [number, [number, number, number]][] = [
  [0.0, [0.15, 0.35, 0.85]],
  [0.25, [0.1, 0.75, 0.85]],
  [0.5, [0.35, 0.85, 0.35]],
  [0.75, [0.95, 0.8, 0.2]],
  [1.0, [0.95, 0.2, 0.15]],
]

function speedToColor(t: number): [number, number, number] {
  const clamped = Math.min(Math.max(t, 0), 1)
  for (let i = 0; i < SPEED_RAMP_STOPS.length - 1; i += 1) {
    const [a, ca] = SPEED_RAMP_STOPS[i]
    const [b, cb] = SPEED_RAMP_STOPS[i + 1]
    if (clamped <= b) {
      const f = (clamped - a) / (b - a)
      return [
        ca[0] + (cb[0] - ca[0]) * f,
        ca[1] + (cb[1] - ca[1]) * f,
        ca[2] + (cb[2] - ca[2]) * f,
      ]
    }
  }
  return SPEED_RAMP_STOPS[SPEED_RAMP_STOPS.length - 1][1]
}

export interface SceneOptions {
  particleSize: number
  showParticles: boolean
  showDepth: boolean
  showVectors: boolean
  showWireframe: boolean
  colorBySpeed: boolean
  maxSpeed: number
}

export const DEFAULT_SCENE_OPTIONS: SceneOptions = {
  particleSize: 0.32,
  showParticles: true,
  showDepth: true,
  showVectors: false,
  showWireframe: false,
  colorBySpeed: true,
  maxSpeed: 12,
}

export class DebrisFlowScene {
  readonly renderer: THREE.WebGLRenderer
  readonly scene: THREE.Scene
  readonly camera: THREE.PerspectiveCamera
  readonly controls: OrbitControls

  private readonly container: HTMLElement
  private readonly resizeObserver: ResizeObserver

  private terrainMesh: THREE.Mesh | null = null
  private terrainEdges: THREE.LineSegments | null = null
  private depthMesh: THREE.Mesh | null = null
  private depthTexture: THREE.DataTexture | null = null
  private vectors: THREE.InstancedMesh | null = null

  private particles: THREE.Points | null = null
  private particlePositions: Float32Array = new Float32Array(0)
  private particleColors: Float32Array = new Float32Array(0)
  private particleCapacity = 0

  private terrain: TerrainLike
  private grid: GridInfo | null = null
  private bedRaster: Float32Array | null = null
  private options: SceneOptions = { ...DEFAULT_SCENE_OPTIONS }
  private animationHandle = 0
  private disposed = false

  constructor(container: HTMLElement, slopeAngleDeg = 30, slopeLength = 20, channelWidth = 8) {
    this.container = container
    this.terrain = buildTerrain(slopeAngleDeg, slopeLength, channelWidth)

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setSize(container.clientWidth || 800, container.clientHeight || 600, false)
    this.renderer.domElement.style.display = 'block'
    this.renderer.domElement.style.width = '100%'
    this.renderer.domElement.style.height = '100%'
    container.appendChild(this.renderer.domElement)

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0b1020)
    // A light exponential fog gives the runout plane depth without hiding the flow.
    this.scene.fog = new THREE.FogExp2(0x0b1020, 0.006)

    this.camera = new THREE.PerspectiveCamera(
      50,
      (container.clientWidth || 800) / (container.clientHeight || 600),
      0.1,
      5000,
    )

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.maxPolarAngle = Math.PI * 0.495
    this.controls.screenSpacePanning = true

    this.addLights()
    this.focusOnTerrain()
    this.setTerrain(this.terrain, null)
    this.animate()

    this.resizeObserver = new ResizeObserver(() => this.handleResize())
    this.resizeObserver.observe(container)
  }

  // ------------------------------------------------------------------
  private addLights(): void {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55))

    const key = new THREE.DirectionalLight(0xffffff, 1.15)
    key.position.set(-40, 60, 35)
    this.scene.add(key)

    const fill = new THREE.DirectionalLight(0x88aaff, 0.4)
    fill.position.set(40, 25, -30)
    this.scene.add(fill)

    // A grid helps the eye judge distance on the flat runout plane.
    const gridSize = 200
    const grid = new THREE.GridHelper(gridSize, 40, 0x1d2b4a, 0x141d33)
    grid.position.y = -0.02
    this.scene.add(grid)
  }

  private focusOnTerrain(): void {
    const t = this.terrain
    const cx = (t.domainXMin + t.domainXMax) / 2
    const cz = (t.domainYMin + t.domainYMax) / 2
    const span = Math.max(t.domainXMax - t.domainXMin, t.domainYMax - t.domainYMin)

    this.controls.target.set(cx, t.crestHeight * 0.3, cz)
    this.camera.position.set(cx - span * 0.45, t.crestHeight + span * 0.5, cz + span * 0.75)
    this.camera.lookAt(this.controls.target)
    this.controls.update()
  }

  /** Frame the whole domain (bound to the "复位视角" button). */
  resetView(): void {
    this.focusOnTerrain()
  }

  /** A vertical, down-slope view �?useful for reading the flow profile. */
  setSideView(): void {
    const t = this.terrain
    const cx = (t.domainXMin + t.domainXMax) / 2
    const cz = (t.domainYMin + t.domainYMax) / 2
    const span = t.domainXMax - t.domainXMin
    this.controls.target.set(cx, t.crestHeight * 0.25, cz)
    this.camera.position.set(cx, t.crestHeight * 0.5, cz + span * 0.85)
    this.controls.update()
  }

  /** A plan view, which is how runout extent is normally read off a hazard map. */
  setTopView(): void {
    const t = this.terrain
    const cx = (t.domainXMin + t.domainXMax) / 2
    const cz = (t.domainYMin + t.domainYMax) / 2
    const span = Math.max(t.domainXMax - t.domainXMin, t.domainYMax - t.domainYMin)
    this.controls.target.set(cx, 0, cz)
    this.camera.position.set(cx, span * 1.05, cz + 0.001)
    this.controls.update()
  }

  // ------------------------------------------------------------------
  // Terrain
  // ------------------------------------------------------------------
  setTerrain(terrain: TerrainLike, bedRaster: Float32Array | null): void {
    this.terrain = terrain
    this.bedRaster = bedRaster ?? this.bedRaster

    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh)
      this.terrainMesh.geometry.dispose()
      ;(this.terrainMesh.material as THREE.Material).dispose()
    }
    if (this.terrainEdges) {
      this.scene.remove(this.terrainEdges)
      this.terrainEdges.geometry.dispose()
      ;(this.terrainEdges.material as THREE.Material).dispose()
    }

    // A modest segment count keeps the analytic surface smooth without
    // competing with the particle stream for GPU time.
    const nx = 140
    const ny = 70
    const geometry = new THREE.PlaneGeometry(
      terrain.domainXMax - terrain.domainXMin,
      terrain.domainYMax - terrain.domainYMin,
      nx,
      ny,
    )
    geometry.rotateX(-Math.PI / 2)

    const position = geometry.attributes.position as THREE.BufferAttribute
    const colors = new Float32Array(position.count * 3)
    const cx = (terrain.domainXMin + terrain.domainXMax) / 2
    const cz = (terrain.domainYMin + terrain.domainYMax) / 2

    for (let i = 0; i < position.count; i += 1) {
      const x = position.getX(i) + cx
      const z = position.getZ(i) + cz
      const y = this.sampleBed(x, z)
      position.setXYZ(i, x, y, z)

      // Steeper ground reads darker; the runout plane reads as a pale apron.
      const shade = 0.30 + 0.42 * Math.min(1, y / Math.max(terrain.crestHeight, 1))
      colors[i * 3] = 0.16 + shade * 0.24
      colors[i * 3 + 1] = 0.20 + shade * 0.26
      colors[i * 3 + 2] = 0.24 + shade * 0.32
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.95,
      metalness: 0.02,
      flatShading: false,
      wireframe: this.options.showWireframe,
      side: THREE.DoubleSide,
    })

    this.terrainMesh = new THREE.Mesh(geometry, material)
    this.scene.add(this.terrainMesh)

    const edges = new THREE.EdgesGeometry(geometry, 32)
    this.terrainEdges = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x2b3d63, transparent: true, opacity: 0.45 }),
    )
    this.terrainEdges.visible = this.options.showWireframe
    this.scene.add(this.terrainEdges)

    if (this.depthMesh) this.rebuildDepthPlane()
  }

  /** Bed elevation at a world (x, z=y_world) point. */
  private sampleBed(x: number, y: number): number {
    if (this.bedRaster && this.grid) {
      const g = this.grid
      const fx = (x - g.x0) / g.dx - 0.5
      const fy = (y - g.y0) / g.dy - 0.5
      const i0 = Math.floor(fx)
      const j0 = Math.floor(fy)
      if (i0 >= 0 && i0 < g.nx - 1 && j0 >= 0 && j0 < g.ny - 1) {
        const tx = fx - i0
        const ty = fy - j0
        const at = (jj: number, ii: number) => this.bedRaster![jj * g.nx + ii]
        return (
          at(j0, i0) * (1 - tx) * (1 - ty) +
          at(j0, i0 + 1) * tx * (1 - ty) +
          at(j0 + 1, i0) * (1 - tx) * ty +
          at(j0 + 1, i0 + 1) * tx * ty
        )
      }
    }
    return bedHeight(this.terrain, x)
  }

  setGrid(grid: GridInfo, bedRaster: Float32Array | null): void {
    this.grid = grid
    this.bedRaster = bedRaster
    // The bed raster is authoritative, so rebuild the mesh from it.
    this.setTerrain(this.terrain, bedRaster)
    this.rebuildDepthPlane()
    this.buildVectors()
  }

  // ------------------------------------------------------------------
  // Depth heat-map + velocity field planes
  // ------------------------------------------------------------------
  private rebuildDepthPlane(): void {
    if (!this.grid) return
    if (this.depthMesh) {
      this.scene.remove(this.depthMesh)
      this.depthMesh.geometry.dispose()
      ;(this.depthMesh.material as THREE.Material).dispose()
      this.depthMesh = null
    }
    if (this.depthTexture) {
      this.depthTexture.dispose()
      this.depthTexture = null
    }

    const g = this.grid
    const data = new Float32Array(g.nx * g.ny)
    const texture = new THREE.DataTexture(data, g.nx, g.ny, THREE.RedFormat, THREE.FloatType)
    texture.minFilter = THREE.LinearFilter
    texture.magFilter = THREE.LinearFilter
    texture.needsUpdate = true
    this.depthTexture = texture

    const geometry = new THREE.PlaneGeometry(g.nx * g.dx, g.ny * g.dy, g.nx - 1, g.ny - 1)
    geometry.rotateX(-Math.PI / 2)

    const position = geometry.attributes.position as THREE.BufferAttribute
    const cx = g.x0 + (g.nx * g.dx) / 2
    const cz = g.y0 + (g.ny * g.dy) / 2
    for (let i = 0; i < position.count; i += 1) {
      const x = position.getX(i) + cx
      const z = position.getZ(i) + cz
      // Lift slightly so it never z-fights with the terrain surface.
      position.setXYZ(i, x, this.sampleBed(x, z) + 0.05, z)
    }
    geometry.computeVertexNormals()

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uDepth: { value: texture },
        uMaxDepth: { value: 3.0 },
        uOpacity: { value: 0.85 },
        uBounds: { value: new THREE.Vector2(0.0, 3.0) },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uDepth;
        uniform float uMaxDepth;
        uniform float uOpacity;
        varying vec2 vUv;

        // Perceptually ordered depth ramp: transparent -> blue -> green -> yellow -> red
        vec3 ramp(float t) {
          t = clamp(t, 0.0, 1.0);
          vec3 c0 = vec3(0.05, 0.20, 0.65);
          vec3 c1 = vec3(0.05, 0.75, 0.85);
          vec3 c2 = vec3(0.30, 0.85, 0.35);
          vec3 c3 = vec3(0.98, 0.80, 0.15);
          vec3 c4 = vec3(0.95, 0.15, 0.10);
          if (t < 0.25) return mix(c0, c1, t / 0.25);
          if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
          if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
          return mix(c3, c4, (t - 0.75) / 0.25);
        }

        void main() {
          // The plane's v axis runs opposite to the backend's y axis after
          // the rotateX(-90deg) that lays it flat, so v must be flipped for
          // the raster to line up with the particles and the terrain mesh.
          float h = texture2D(uDepth, vec2(vUv.x, 1.0 - vUv.y)).r;
          float t = h / max(uMaxDepth, 1e-4);
          // Everything below ~1 cm of column is dry ground: draw nothing.
          float alpha = smoothstep(0.004, 0.03, t) * uOpacity;
          if (alpha < 0.01) discard;
          gl_FragColor = vec4(ramp(t), alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    })

    this.depthMesh = new THREE.Mesh(geometry, material)
    this.depthMesh.visible = this.options.showDepth
    this.depthMesh.renderOrder = 1
    this.scene.add(this.depthMesh)
  }

  private buildVectors(): void {
    if (this.vectors) {
      this.scene.remove(this.vectors)
      this.vectors.geometry.dispose()
      ;(this.vectors.material as THREE.Material).dispose()
      this.vectors = null
    }
    if (!this.grid) return

    const g = this.grid
    // Decimate so arrows remain legible rather than becoming a solid mat.
    const strideX = Math.max(1, Math.round(g.nx / 34))
    const strideY = Math.max(1, Math.round(g.ny / 17))
    const cols = Math.floor(g.nx / strideX)
    const rows = Math.floor(g.ny / strideY)
    const count = cols * rows
    if (count <= 0) return

    // A flat arrow in the XY plane, pointing along +Y before rotation.
    const shape = new THREE.Shape()
    shape.moveTo(0, -0.5)
    shape.lineTo(0.28, -0.15)
    shape.lineTo(0.1, -0.15)
    shape.lineTo(0.1, 0.5)
    shape.lineTo(-0.1, 0.5)
    shape.lineTo(-0.1, -0.15)
    shape.lineTo(-0.28, -0.15)
    shape.closePath()
    const geometry = new THREE.ShapeGeometry(shape)

    const material = new THREE.MeshBasicMaterial({
      color: 0x9fe8ff,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    })

    this.vectors = new THREE.InstancedMesh(geometry, material, count)
    this.vectors.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.vectors.frustumCulled = false
    this.vectors.visible = this.options.showVectors
    this.vectors.renderOrder = 2
    this.vectors.count = count
    this.scene.add(this.vectors)
    this.vectorLayout = { strideX, strideY, cols, rows }
  }

  private vectorLayout = { strideX: 1, strideY: 1, cols: 0, rows: 0 }

  // ------------------------------------------------------------------
  // Particles
  // ------------------------------------------------------------------
  private ensureParticleCapacity(count: number): void {
    if (count <= this.particleCapacity && this.particles) return

    if (this.particles) {
      this.scene.remove(this.particles)
      this.particles.geometry.dispose()
      ;(this.particles.material as THREE.Material).dispose()
      this.particles = null
    }

    // Grow in blocks: reallocating on every frame would thrash the GPU buffers.
    this.particleCapacity = Math.max(1024, Math.ceil(count * 1.25))
    this.particlePositions = new Float32Array(this.particleCapacity * 3)
    this.particleColors = new Float32Array(this.particleCapacity * 3)

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.particlePositions, 3).setUsage(THREE.DynamicDrawUsage),
    )
    geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(this.particleColors, 3).setUsage(THREE.DynamicDrawUsage),
    )
    geometry.setDrawRange(0, 0)

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: this.options.particleSize },
        uScale: { value: this.renderer.getSize(new THREE.Vector2()).y * 0.5 },
      },
      vertexShader: /* glsl */ `
        attribute vec3 color;
        uniform float uSize;
        uniform float uScale;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          // Perspective-correct round splat.
          gl_PointSize = uSize * uScale / max(-mv.z, 0.001);
          gl_PointSize = clamp(gl_PointSize, 1.0, 24.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        void main() {
          vec2 d = gl_PointCoord - vec2(0.5);
          float r2 = dot(d, d);
          if (r2 > 0.25) discard;
          // Soft edge keeps dense clusters from aliasing into a hard mass.
          float alpha = smoothstep(0.25, 0.06, r2);
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
    })

    this.particles = new THREE.Points(geometry, material)
    this.particles.frustumCulled = false
    this.particles.visible = this.options.showParticles
    this.particles.renderOrder = 3
    this.scene.add(this.particles)
  }

  /** Push one decoded keyframe into the GPU buffers. */
  updateFrame(frame: DecodedFrame): void {
    const count = Math.floor(frame.positions.length / 3)
    if (count === 0) return

    this.ensureParticleCapacity(count)
    if (!this.particles) return

    const positions = this.particlePositions
    const colors = this.particleColors
    const velocities = frame.velocities

    // Use a running max so the colour ramp stays stable frame to frame
    // instead of re-normalising (and visibly flickering) every update.
    let peak = this.options.maxSpeed
    if (this.options.colorBySpeed) {
      for (let i = 0; i < count; i += 8) {
        const vx = velocities[i * 3] ?? 0
        const vy = velocities[i * 3 + 1] ?? 0
        const vz = velocities[i * 3 + 2] ?? 0
        const s = Math.sqrt(vx * vx + vy * vy + vz * vz)
        if (s > peak) peak = s
      }
      this.options.maxSpeed = peak
    }

    const inv = 1 / Math.max(peak, 0.5)
    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = frame.positions[i * 3]
      positions[i * 3 + 1] = frame.positions[i * 3 + 2] // backend z -> three.js y
      positions[i * 3 + 2] = frame.positions[i * 3 + 1] // backend y -> three.js z

      if (this.options.colorBySpeed) {
        const vx = velocities[i * 3] ?? 0
        const vy = velocities[i * 3 + 1] ?? 0
        const vz = velocities[i * 3 + 2] ?? 0
        const s = Math.sqrt(vx * vx + vy * vy + vz * vz)
        const [r, g, b] = speedToColor(s * inv)
        colors[i * 3] = r
        colors[i * 3 + 1] = g
        colors[i * 3 + 2] = b
      } else {
        colors[i * 3] = 0.95
        colors[i * 3 + 1] = 0.55
        colors[i * 3 + 2] = 0.15
      }
    }

    const geometry = this.particles.geometry
    geometry.setDrawRange(0, count)
    ;(geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true
    ;(geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true
    geometry.computeBoundingSphere()

    this.updateDepthTexture(frame)
    if (this.options.showVectors) this.updateVectors(frame)
  }

  private updateDepthTexture(frame: DecodedFrame): void {
    if (!this.depthTexture || !this.depthMesh) return
    const data = this.depthTexture.image.data as unknown as Float32Array
    if (data.length !== frame.depth.length) return

    let maxDepth = 0
    for (let i = 0; i < frame.depth.length; i += 1) {
      const v = frame.depth[i]
      data[i] = v
      if (v > maxDepth) maxDepth = v
    }
    // Smooth the auto-range so the colour scale does not pump.
    const material = this.depthMesh.material as THREE.ShaderMaterial
    const current = material.uniforms.uMaxDepth.value as number
    material.uniforms.uMaxDepth.value = Math.max(current * 0.92, maxDepth * 0.9, 0.25)
    this.depthTexture.needsUpdate = true
  }

  private updateVectors(frame: DecodedFrame): void {
    if (!this.vectors || !this.grid) return
    const g = this.grid
    const { strideX, strideY, cols, rows } = this.vectorLayout

    const matrix = new THREE.Matrix4()
    const quaternion = new THREE.Quaternion()
    const qFlat = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      -Math.PI / 2,
    )
    const qYaw = new THREE.Quaternion()
    const axisY = new THREE.Vector3(0, 1, 0)
    const scale = new THREE.Vector3()
    const position = new THREE.Vector3()

    let index = 0
    const arrowScale = Math.max(0.6, Math.min(g.dx * strideX, g.dy * strideY) * 1.8)

    for (let jj = 0; jj < rows; jj += 1) {
      const j = jj * strideY
      for (let ii = 0; ii < cols; ii += 1) {
        const i = ii * strideX
        const cell = j * g.nx + i
        const vx = frame.velocityX[cell] ?? 0
        const vy = frame.velocityY[cell] ?? 0
        const speed = Math.hypot(vx, vy)
        const depth = frame.depth[cell] ?? 0

        if (depth < 1e-3 || speed < 0.05 || index >= this.vectors.count) {
          matrix.makeScale(0, 0, 0)
          this.vectors.setMatrixAt(index, matrix)
          index += 1
          continue
        }

        const x = g.x0 + (i + 0.5) * g.dx
        const z = g.y0 + (j + 0.5) * g.dy
        position.set(x, this.sampleBed(x, z) + 0.12, z)

        // The shape points along local +Y; qFlat lays it into the ground
        // plane pointing along -Z, so the heading that lands it on (vx, vy)
        // in the world XZ plane is atan2(-vx, -vy).
        qYaw.setFromAxisAngle(axisY, Math.atan2(-vx, -vy))
        quaternion.copy(qYaw).multiply(qFlat)

        const length = arrowScale * Math.min(1.6, speed / 6)
        scale.set(length, length, 1)

        matrix.compose(position, quaternion, scale)
        this.vectors.setMatrixAt(index, matrix)
        index += 1
      }
    }

    for (; index < this.vectors.count; index += 1) {
      matrix.makeScale(0, 0, 0)
      this.vectors.setMatrixAt(index, matrix)
    }

    this.vectors.instanceMatrix.needsUpdate = true
  }

  // ------------------------------------------------------------------
  setOptions(options: Partial<SceneOptions>): void {
    this.options = { ...this.options, ...options }
    if (this.particles) {
      this.particles.visible = this.options.showParticles
      const material = this.particles.material as THREE.ShaderMaterial
      material.uniforms.uSize.value = this.options.particleSize
    }
    if (this.depthMesh) this.depthMesh.visible = this.options.showDepth
    if (this.vectors) this.vectors.visible = this.options.showVectors
    if (this.terrainMesh) {
      ;(this.terrainMesh.material as THREE.MeshStandardMaterial).wireframe =
        this.options.showWireframe
    }
    if (this.terrainEdges) this.terrainEdges.visible = this.options.showWireframe
  }

  getOptions(): SceneOptions {
    return { ...this.options }
  }

  /** Reset the colour-ramp auto-range (call when a new run starts). */
  resetSpeedRange(): void {
    this.options.maxSpeed = 12
  }

  clearParticles(): void {
    if (this.particles) {
      this.particles.geometry.setDrawRange(0, 0)
    }
    if (this.depthTexture) {
      const data = this.depthTexture.image.data as unknown as Float32Array
      data.fill(0)
      this.depthTexture.needsUpdate = true
    }
  }

  // ------------------------------------------------------------------
  private handleResize(): void {
    if (this.disposed) return
    const width = this.container.clientWidth
    const height = this.container.clientHeight
    if (width === 0 || height === 0) return

    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height, false)

    if (this.particles) {
      const material = this.particles.material as THREE.ShaderMaterial
      material.uniforms.uScale.value = height * 0.5
    }
  }

  private animate = (): void => {
    if (this.disposed) return
    this.animationHandle = requestAnimationFrame(this.animate)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  /** PNG data URL of the current viewport (used by the screenshot button). */
  captureScreenshot(): string {
    this.renderer.render(this.scene, this.camera)
    return this.renderer.domElement.toDataURL('image/png')
  }

  dispose(): void {
    this.disposed = true
    cancelAnimationFrame(this.animationHandle)
    this.resizeObserver.disconnect()
    this.controls.dispose()

    if (this.particles) {
      this.particles.geometry.dispose()
      ;(this.particles.material as THREE.Material).dispose()
    }
    if (this.terrainMesh) {
      this.terrainMesh.geometry.dispose()
      ;(this.terrainMesh.material as THREE.Material).dispose()
    }
    if (this.terrainEdges) {
      this.terrainEdges.geometry.dispose()
      ;(this.terrainEdges.material as THREE.Material).dispose()
    }
    if (this.depthMesh) {
      this.depthMesh.geometry.dispose()
      ;(this.depthMesh.material as THREE.Material).dispose()
    }
    if (this.vectors) {
      this.vectors.geometry.dispose()
      ;(this.vectors.material as THREE.Material).dispose()
    }
    if (this.depthTexture) this.depthTexture.dispose()

    this.renderer.dispose()
    this.renderer.domElement.remove()
  }

  /** Adopt geometry streamed by the backend so mesh and raster stay aligned. */
  applyTerrainInfo(info: TerrainInfo, grid: GridInfo, bedRaster: Float32Array | null): void {
    this.terrain = terrainFromInfo(info)
    this.grid = grid
    this.bedRaster = bedRaster
    this.setTerrain(this.terrain, bedRaster)
    this.rebuildDepthPlane()
    this.buildVectors()
    this.focusOnTerrain()
  }
}
