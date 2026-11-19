/** Pure helpers for the loss-surface view: axes, range, contours, colour scale, path. No DOM. */
import type { HostParams, NodeId } from '@tinker/core';

/** Grid points per axis. Fixed (CONTRACTS section 7: about 0.3 ms for two_param, 89 ms for an MLP at n = 500). */
export const SURFACE_N = 50;
/** Minimum half-width of the range around the current parameters. */
export const MIN_HALF = 3;
/** Largest half-width the range follows the path to; a run that runs off further keeps the last range. */
export const MAX_HALF = 100;
/** The path keeps at most this many points (halved, keeping the last, when it grows past it). */
export const MAX_PATH = 4000;

export type SurfaceAxis = { node: NodeId; tensor: 'W' | 'b'; index: number };

/**
 * The two parameters of a model that has exactly two (weights first, then biases, in the order of
 * `order`), with their current values; null for any other count.
 */
export function twoParams(
  host: HostParams,
  order: readonly NodeId[],
): { axes: [SurfaceAxis, SurfaceAxis]; values: [number, number] } | null {
  const axes: SurfaceAxis[] = [];
  const values: number[] = [];
  for (const node of order) {
    const p = Object.hasOwn(host, node) ? host[node] : undefined;
    if (!p) continue;
    for (let i = 0; i < p.W.length; i++) {
      axes.push({ node, tensor: 'W', index: i });
      values.push(p.W[i] ?? 0);
    }
    if (p.b) {
      for (let i = 0; i < p.b.length; i++) {
        axes.push({ node, tensor: 'b', index: i });
        values.push(p.b[i] ?? 0);
      }
    }
    if (axes.length > 2) return null;
  }
  const [a0, a1] = axes;
  const [v0, v1] = values;
  if (axes.length !== 2 || !a0 || !a1 || v0 === undefined || v1 === undefined) return null;
  return { axes: [a0, a1], values: [v0, v1] };
}

/** Short axis label: "weight" / "bias", prefixed with the node id when the two axes sit on different nodes. */
/** Axis titles as `$...$` text: `$w_{1}$`, `$b$`, or the node's name before the symbol. */
export function axisLabels(axes: [SurfaceAxis, SurfaceAxis]): [string, string] {
  const sameNode = axes[0].node === axes[1].node;
  const bothW = sameNode && axes[0].tensor === 'W' && axes[1].tensor === 'W';
  const one = (a: SurfaceAxis) => {
    const sym = a.tensor === 'W' ? 'w' : 'b';
    const n = bothW ? `${sym}_{${a.index + 1}}` : sym;
    return sameNode ? `$${n}$` : `${a.node} $${n}$`;
  };
  return [one(axes[0]), one(axes[1])];
}

export type Pt = readonly [number, number];
export type SurfaceRange = { a: [number, number]; b: [number, number] };

/** Appends a point; past `max` points every second one is dropped (the last stays). Returns a new array only then. */
export function appendPath(path: Pt[], p: Pt, max = MAX_PATH): Pt[] {
  path.push(p);
  if (path.length <= max) return path;
  const last = path.length - 1;
  return path.filter((_, i) => i % 2 === 0 || i === last);
}

function inside(r: SurfaceRange, p: Pt): boolean {
  return p[0] >= r.a[0] && p[0] <= r.a[1] && p[1] >= r.b[0] && p[1] <= r.b[1];
}

/**
 * Range of the surface. Centred on the current point (the last of `path`), half-width per axis
 * `max(MIN_HALF, 1.25 x the largest distance of any path point from it)`. `prev` is kept as long
 * as every path point is still inside it; a null `prev` (new run) or a path that left it picks a
 * new range. Returns null for an empty path.
 */
export function nextRange(prev: SurfaceRange | null, path: readonly Pt[]): SurfaceRange | null {
  const cur = path[path.length - 1];
  if (!cur) return null;
  if (prev && path.every((p) => inside(prev, p))) return prev;
  let da = 0;
  let db = 0;
  for (const p of path) {
    da = Math.max(da, Math.abs(p[0] - cur[0]));
    db = Math.max(db, Math.abs(p[1] - cur[1]));
  }
  const ha = Math.max(MIN_HALF, 1.25 * da);
  const hb = Math.max(MIN_HALF, 1.25 * db);
  if (!(ha <= MAX_HALF && hb <= MAX_HALF) && prev) return prev; // diverging: keep the last good range
  if (!(Number.isFinite(cur[0]) && Number.isFinite(cur[1]))) return prev;
  return { a: [cur[0] - ha, cur[0] + ha], b: [cur[1] - hb, cur[1] + hb] };
}

/** `n` evenly spaced values from lo to hi inclusive. */
export function linspace(lo: number, hi: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = n > 1 ? lo + ((hi - lo) * i) / (n - 1) : lo;
  return out;
}

export type LossScale = { min: number; hi: number };

/**
 * Loss range of a grid: `min` is the smallest finite value, `hi` the 90th percentile (so one steep
 * corner does not wash out the basin). Null when no value is finite.
 */
export function lossScale(grid: Float32Array): LossScale | null {
  const v: number[] = [];
  for (const x of grid) if (Number.isFinite(x)) v.push(x);
  if (v.length === 0) return null;
  v.sort((p, q) => p - q);
  const min = v[0] ?? 0;
  const hi = v[Math.min(v.length - 1, Math.floor(0.9 * v.length))] ?? min;
  return { min, hi: hi > min ? hi : min + 1 };
}

/** Contour levels between the minimum and the 90th percentile, denser near the minimum. */
export function contourLevels(s: LossScale, count = 8): number[] {
  const out: number[] = [];
  for (let k = 1; k <= count; k++) out.push(s.min + (s.hi - s.min) * (k / (count + 1)) ** 1.6);
  return out;
}

/** 0 at the minimum, 1 at the 90th percentile and above (square root: more contrast near the basin). */
export function heat(v: number, s: LossScale): number {
  if (!Number.isFinite(v)) return Number.NaN;
  return Math.sqrt(Math.min(1, Math.max(0, (v - s.min) / (s.hi - s.min))));
}

/**
 * Marching squares on `grid` ([na, nb], row-major, entry i * nb + j). Returns segments as flat
 * quadruples (i1, j1, i2, j2) in fractional grid indices. Cells with a non-finite corner are
 * skipped; saddle cells are resolved with the cell mean.
 */
export function contourSegments(
  grid: Float32Array,
  na: number,
  nb: number,
  level: number,
): number[] {
  const out: number[] = [];
  const at = (i: number, j: number) => grid[i * nb + j] ?? Number.NaN;
  for (let i = 0; i + 1 < na; i++) {
    for (let j = 0; j + 1 < nb; j++) {
      const v00 = at(i, j);
      const v10 = at(i + 1, j);
      const v11 = at(i + 1, j + 1);
      const v01 = at(i, j + 1);
      if (!(
        Number.isFinite(v00) &&
        Number.isFinite(v10) &&
        Number.isFinite(v11) &&
        Number.isFinite(v01)
      )) {
        continue;
      }
      const code =
        (v00 >= level ? 1 : 0) |
        (v10 >= level ? 2 : 0) |
        (v11 >= level ? 4 : 0) |
        (v01 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const t = (p: number, q: number) => (level - p) / (q - p);
      // edge crossing points: bottom (j fixed, i varies), right, top, left
      const eB: Pt = [i + t(v00, v10), j];
      const eR: Pt = [i + 1, j + t(v10, v11)];
      const eT: Pt = [i + t(v01, v11), j + 1];
      const eL: Pt = [i, j + t(v00, v01)];
      const seg = (p: Pt, q: Pt) => out.push(p[0], p[1], q[0], q[1]);
      switch (code) {
        case 1:
        case 14:
          seg(eL, eB);
          break;
        case 2:
        case 13:
          seg(eB, eR);
          break;
        case 3:
        case 12:
          seg(eL, eR);
          break;
        case 4:
        case 11:
          seg(eR, eT);
          break;
        case 6:
        case 9:
          seg(eB, eT);
          break;
        case 7:
        case 8:
          seg(eL, eT);
          break;
        case 5:
        case 10: {
          const centre = (v00 + v10 + v11 + v01) / 4 >= level;
          // code 5: corners 00 and 11 above; code 10: corners 10 and 01 above
          const joined = code === 5 ? centre : !centre;
          if (joined) {
            seg(eL, eT);
            seg(eB, eR);
          } else {
            seg(eL, eB);
            seg(eR, eT);
          }
          break;
        }
        default:
          break;
      }
    }
  }
  return out;
}
