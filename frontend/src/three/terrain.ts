/**
 * Analytic bed surface — the TypeScript twin of `backend/app/physics/terrain.py`.
 *
 * The backend streams its own bed raster inside the `meta` message, so the
 * viewport can (and does) use that as the source of truth for the heat-map
 * plane. This module exists so the mesh can be rebuilt instantly while the
 * user drags the slope slider, before any run has been started.
 *
 *     z_b(x) = R * tan(theta) * softplus((x_toe - x) / R)
 */
import type { TerrainInfo } from '@/types/simulation'

export function softplus(t: number): number {
  return Math.max(t, 0) + Math.log1p(Math.exp(-Math.abs(t)))
}

export interface TerrainLike {
  slopeAngleDeg: number
  slopeAngleRad: number
  slopeLength: number
  toeX: number
  crestHeight: number
  turnRadius: number
  domainXMin: number
  domainXMax: number
  domainYMin: number
  domainYMax: number
}

/** Derive terrain geometry from raw parameters (mirrors `build_terrain`). */
export function buildTerrain(
  slopeAngleDeg: number,
  slopeLength: number,
  channelWidth: number,
): TerrainLike {
  const rad = (slopeAngleDeg * Math.PI) / 180
  const toeX = slopeLength * Math.cos(rad)
  const crestHeight = slopeLength * Math.sin(rad)
  // 8% of the horizontal run keeps the softplus hinge confined to the toe —
  // a wider radius would round the whole slope and weaken the driving force.
  // Must stay in step with `build_terrain` in backend/app/physics/terrain.py.
  const turnRadius = Math.max(0.4, 0.08 * Math.max(toeX, 1))
  const runoutLength = Math.max(8, 2.5 * Math.max(toeX, 1))
  const marginX = 0.15 * Math.max(toeX, 1)
  const marginY = 2 * channelWidth

  return {
    slopeAngleDeg,
    slopeAngleRad: rad,
    slopeLength,
    toeX,
    crestHeight,
    turnRadius,
    domainXMin: -marginX,
    domainXMax: toeX + runoutLength,
    domainYMin: -0.5 * channelWidth - marginY,
    domainYMax: 0.5 * channelWidth + marginY,
  }
}

/** Adopt the authoritative geometry the backend returned. */
export function terrainFromInfo(info: TerrainInfo): TerrainLike {
  return {
    slopeAngleDeg: info.slope_angle_deg,
    slopeAngleRad: (info.slope_angle_deg * Math.PI) / 180,
    slopeLength: info.slope_length_m,
    toeX: info.toe_x_m,
    crestHeight: info.crest_height_m,
    turnRadius: info.turn_radius_m,
    domainXMin: info.domain_x_min,
    domainXMax: info.domain_x_max,
    domainYMin: info.domain_y_min,
    domainYMax: info.domain_y_max,
  }
}

export function bedHeight(t: TerrainLike, x: number): number {
  return t.turnRadius * Math.tan(t.slopeAngleRad) * softplus((t.toeX - x) / t.turnRadius)
}
