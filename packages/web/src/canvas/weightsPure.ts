/** Pure helpers for the weight inspector: no React, no store. Unit tested in test/weightsPure.test.ts. */
import { parsePort, type Graph, type HostParams, type Node } from '@tinker/core';

/** Largest block of cells drawn at once; a bigger layer is paged (see `windowOf`). */
export const GRID_MAX = 32;

export type WeightGrid = {
  /** Inputs (rows of W); 0 for a lone Bias. */
  rows: number;
  /** Output units (columns of W, and the length of b). */
  cols: number;
  /** Row-major `[rows, cols]`, or null when the piece has no W. */
  W: Float32Array | null;
  /** One bias per column, or null when the piece has none. */
  b: Float32Array | null;
};

/**
 * The values to show for a piece, or null when the model has none that fit the graph (a stale
 * snapshot, a piece off the loss path). `inWidth` and `shape` are the Linear's current and
 * snapshot-time [in, out] (`hostShape`), so a swapped shape is refused as in the neuron view.
 */
export function gridOf(
  host: HostParams | undefined,
  node: Node,
  inWidth: number | undefined,
  shape: readonly [number, number] | undefined,
): WeightGrid | null {
  const e = host?.[node.id];
  if (!e) return null;
  if (node.type === 'scalew') {
    return e.W?.length === 1 ? { rows: 1, cols: 1, W: e.W, b: null } : null;
  }
  if (node.type === 'bias') {
    return e.b?.length === 1 ? { rows: 0, cols: 1, W: null, b: e.b } : null;
  }
  if (node.type !== 'linear' || inWidth === undefined || !e.W) return null;
  if (e.W.length !== inWidth * node.out) return null;
  if (shape && (shape[0] !== inWidth || shape[1] !== node.out)) return null;
  const b = node.bias && e.b?.length === node.out ? e.b : null;
  return { rows: inWidth, cols: node.out, W: e.W, b };
}

export type WeightStats = { min: number; max: number; mean: number; std: number; count: number };

/** Min, max, mean and (population) standard deviation of the finite values; null for none. */
export function statsOf(values: ArrayLike<number>): WeightStats | null {
  let n = 0;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let k = 0; k < values.length; k++) {
    const v = values[k] as number;
    if (!Number.isFinite(v)) continue;
    n++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (n === 0) return null;
  const mean = sum / n;
  let ss = 0;
  for (let k = 0; k < values.length; k++) {
    const v = values[k] as number;
    if (Number.isFinite(v)) ss += (v - mean) ** 2;
  }
  return { min, max, mean, std: Math.sqrt(ss / n), count: n };
}

/** The page of `total` items (pages of GRID_MAX) that is shown: index range and the page count. */
export function windowOf(total: number, page: number): { from: number; to: number; pages: number } {
  const pages = Math.max(1, Math.ceil(total / GRID_MAX));
  const p = Math.min(Math.max(0, page), pages - 1);
  return { from: p * GRID_MAX, to: Math.min(total, (p + 1) * GRID_MAX), pages };
}

/**
 * Percentage of the diverging colour in a cell: a quiet tint for zero, at most 50 so the text on
 * it keeps its contrast. `maxAbs` is the largest magnitude in the whole layer.
 */
export function tintPct(v: number, maxAbs: number): number {
  if (!Number.isFinite(v) || maxAbs <= 0) return 0;
  return Math.round(4 + 46 * Math.min(1, Math.abs(v) / maxAbs));
}

/** A value inside a cell: two decimals, fewer as it grows (the limit is 100); a true minus sign. */
export function cellText(v: number): string {
  if (!Number.isFinite(v)) return '?';
  const a = Math.abs(v);
  const s = a < 10 ? v.toFixed(2) : a < 100 ? v.toFixed(1) : v.toFixed(0);
  return s.replace('-', '−');
}

/** TeX of a name: a name already in `$..$` is unwrapped, `x2` is `x_{2}`, any other is set upright. */
function texOf(name: string): string {
  if (name.startsWith('$') && name.endsWith('$')) return name.slice(1, -1);
  const m = /^x(\d+)$/.exec(name);
  return m ? `x_{${m[1]}}` : `\\mathrm{${name.replace(/_/g, '\\_')}}`;
}

/** Names of the columns a Feature expands, in the engine's order: powers, sin, cos, products. */
function featureNames(
  f: { degree: number; sin: boolean; cos: boolean; products: boolean },
  base: readonly string[],
): string[] {
  const t = base.map(texOf);
  const out: string[] = [];
  for (let p = 1; p <= f.degree; p++) {
    for (const x of t) out.push(p === 1 ? `$${x}$` : `$(${x})^{${p}}$`);
  }
  if (f.sin) for (const x of t) out.push(`$\\sin ${x}$`);
  if (f.cos) for (const x of t) out.push(`$\\cos ${x}$`);
  if (f.products) {
    for (let i = 0; i < t.length; i++) {
      for (let j = i + 1; j < t.length; j++) out.push(`$${t[i]}\\,${t[j]}$`);
    }
  }
  return out;
}

/**
 * Names for the rows (inputs) of a Linear: the columns of the Input that feeds it, followed
 * upstream through pieces that keep the width (Activation, Dropout, Link), the Column's name, or a
 * Feature's expanded columns (TeX names in `$..$`: render with `TexText`). Falls back to
 * `input i` when the names are not known or do not match the width.
 */
export function rowNames(
  graph: Graph,
  node: string,
  inWidth: number,
  columns: readonly string[],
): string[] {
  const known = (at: string, depth: number): string[] | null => {
    if (depth > 32) return null;
    const e = graph.edges.find((x) => parsePort(x[1]).node === at);
    const from = e && graph.nodes.find((n) => n.id === parsePort(e[0]).node);
    if (!from) return null;
    if (from.type === 'input') return from.columns.length > 0 ? from.columns : [...columns];
    if (from.type === 'column') return [from.name];
    if (from.type === 'activation' || from.type === 'dropout' || from.type === 'link') {
      return known(from.id, depth + 1);
    }
    if (from.type === 'feature') {
      const base = known(from.id, depth + 1);
      return base && featureNames(from, base);
    }
    return null;
  };
  const names = known(node, 0);
  return names && names.length === inWidth
    ? names
    : Array.from({ length: inWidth }, (_, i) => `input ${i}`);
}
