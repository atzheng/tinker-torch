/** Pure helpers for the neuron view: no React, no store. Unit tested in test/neuronPure.test.ts. */
import type { HostParams, NeuronLayout, NodeId } from '@tinker/core';
import { HIT_MIN_PX, type HitRect } from './pure.js';

/** Radius of a unit circle in abstract units. */
export const UNIT_R = 12;

export type NeuronEdge = NeuronLayout['edges'][number];
export type NeuronUnit = NeuronLayout['units'][number];

/** Edge colour sign and thickness (abstract units) for a weight value. Thickness grows with |w|. */
export function edgeStyle(w: number): { sign: 'pos' | 'neg'; width: number } {
  const m = Number.isFinite(w) ? Math.min(Math.abs(w), 1.5) : 0;
  return { sign: w < 0 ? 'neg' : 'pos', width: 0.6 + 3.4 * m };
}

/** `W[i][j]` of a Linear node, or undefined when the entry is missing or has another size. */
export function weightAt(
  host: HostParams | undefined,
  node: NodeId,
  i: number,
  j: number,
  inWidth: number,
  out: number,
  /** [in, out] the host was built for, when known (`hostShape`): a swapped shape is refused. */
  shape?: readonly [number, number],
): number | undefined {
  const e = host?.[node];
  if (!e?.W || e.W.length !== inWidth * out || i < 0 || j < 0 || i >= inWidth || j >= out) {
    return undefined;
  }
  if (shape && (shape[0] !== inWidth || shape[1] !== out)) return undefined;
  return e.W[i * out + j];
}

/** Bias `b[j]` of a Linear node, undefined when there is none. */
export function biasAt(host: HostParams | undefined, node: NodeId, j: number): number | undefined {
  return host?.[node]?.b?.[j];
}

/** The segment from `from` to `to` shortened by `r1` at the start and `r2` at the end. */
export function trimSegment(
  from: readonly [number, number],
  to: readonly [number, number],
  r1: number,
  r2: number,
): [number, number, number, number] {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (len <= r1 + r2) return [from[0], from[1], to[0], to[1]];
  const ux = dx / len;
  const uy = dy / len;
  return [from[0] + ux * r1, from[1] + uy * r1, to[0] - ux * r2, to[1] - uy * r2];
}

/** Distance from point (px, py) to the segment a-b. */
export function distToSegment(
  px: number,
  py: number,
  a: readonly [number, number],
  b: readonly [number, number],
): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / len2));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

/**
 * The edge nearest to (x, y) in abstract units within `maxDist`, if any. Neuron edges fan out
 * and cross, so the one whose line passes closest to the press wins, not the one painted last.
 * Index into `edges`. Ties go to the earlier edge.
 */
export function nearestEdge(
  edges: readonly NeuronEdge[],
  x: number,
  y: number,
  maxDist: number,
): number | undefined {
  let best: number | undefined;
  let bestD = maxDist;
  edges.forEach((e, k) => {
    const d = distToSegment(x, y, e.from, e.to);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  });
  return best;
}

export type NeuronHit =
  | ({ kind: 'unit'; node: NodeId; index: number } & HitRect)
  | ({ kind: 'node'; id: NodeId } & HitRect);

/**
 * Hit rectangles of the neuron view in paint order (the last wins): every piece's box, then every
 * unit. A unit's rectangle is its whole cell of the column (the full pitch tall, so cells tile and
 * never overlap even when the view is scaled down to 0.5) and at least 32 screen px wide. A
 * piece's rectangle is its box grown to 32 px each way.
 */
export function neuronHitShapes(
  layout: Pick<NeuronLayout, 'units' | 'boxes'>,
  scale: number,
  pitch: number,
  unitW: number,
): NeuronHit[] {
  const out: NeuronHit[] = [];
  const min = HIT_MIN_PX / scale;
  for (const [id, b] of Object.entries(layout.boxes)) {
    const w = Math.max(b.w, min);
    const h = Math.max(b.h, min);
    out.push({ kind: 'node', id, x: b.x - (w - b.w) / 2, y: b.y - (h - b.h) / 2, w, h });
  }
  const w = Math.max(unitW, min);
  for (const u of layout.units) {
    out.push({
      kind: 'unit',
      node: u.node,
      index: u.index,
      x: u.x - w / 2,
      y: u.y - pitch / 2,
      w,
      h: pitch,
    });
  }
  return out;
}

/** Topmost neuron hit shape containing the point (abstract units). */
export function neuronHitAt(
  shapes: readonly NeuronHit[],
  x: number,
  y: number,
): NeuronHit | undefined {
  for (let i = shapes.length - 1; i >= 0; i--) {
    const s = shapes[i];
    if (s && x >= s.x && x <= s.x + s.w && y >= s.y && y <= s.y + s.h) return s;
  }
  return undefined;
}

/**
 * What a press at (x, y) means: a unit or piece if the point is inside one of their rectangles,
 * else the nearest edge within `edgePx` screen pixels. Pure mirror of what the view does.
 */
export function pickNeuron(
  layout: Pick<NeuronLayout, 'units' | 'boxes' | 'edges'>,
  shapes: readonly NeuronHit[],
  scale: number,
  edgePx: number,
  x: number,
  y: number,
): { kind: 'unit'; node: NodeId; index: number } | { kind: 'edge'; k: number } | undefined {
  const hit = neuronHitAt(shapes, x, y);
  if (hit?.kind === 'unit') return { kind: 'unit', node: hit.node, index: hit.index };
  const k = nearestEdge(layout.edges, x, y, edgePx / 2 / scale);
  if (k !== undefined) return { kind: 'edge', k };
  return undefined;
}
