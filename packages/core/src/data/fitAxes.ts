import { standardizeRows, type Standardization } from './sample.js';
import type { Grid } from './grid.js';

type Info = {
  columns: string[];
  columnKinds?: ('numeric' | 'categorical')[];
  categories?: Record<string, string[]>;
};

/** What the fit plot draws for a dataset's columns (tinker-hxq). */
export type FitView =
  { type: 'surface' } | { type: 'curve'; col: number } | { type: 'categories'; col: number };

/** Column indices that can be a continuous axis (every non-categorical column). */
export function continuousCols(info: Info): number[] {
  return info.columns.map((_, c) => c).filter((c) => info.columnKinds?.[c] !== 'categorical');
}

/** Category names of column `col` (empty for a numeric column). */
export function categoryNames(info: Info, col: number): string[] {
  if (info.columnKinds?.[col] !== 'categorical') return [];
  const id = info.columns[col] ?? '';
  return info.categories?.[id] ?? [];
}

/**
 * Indices of the dataset columns the graph reads: every Input piece's columns and every Column
 * piece's name, in dataset order. Empty when the graph reads none (a no-input model).
 */
export function usedColumns(
  graph: { nodes: readonly { type: string; columns?: unknown; name?: unknown }[] },
  info: Info,
): number[] {
  const names = new Set<string>();
  for (const n of graph.nodes) {
    if (n.type === 'input' && Array.isArray(n.columns)) {
      for (const c of n.columns) if (typeof c === 'string') names.add(c);
    } else if (n.type === 'column' && typeof n.name === 'string') names.add(n.name);
  }
  return info.columns.map((c, i) => i).filter((i) => names.has(info.columns[i] ?? ''));
}

/**
 * Which fit plot to draw. Only the columns the model reads count (`used`, dataset column indices;
 * undefined or empty means every column, the no-input case). Two or more continuous ones: the 2-D
 * surface (axis pickers offer only those). Exactly one: the curve over it. None: the per-category
 * marks over the first column read (the others held at their usual value).
 */
export function fitView(info: Info, used?: readonly number[]): FitView {
  const all = info.columns.map((_, c) => c);
  const cols = used && used.length > 0 ? all.filter((c) => used.includes(c)) : all;
  const cont = continuousCols(info).filter((c) => cols.includes(c));
  if (cont.length >= 2) return { type: 'surface' };
  if (cont.length >= 1) return { type: 'curve', col: cont[0] ?? 0 };
  return { type: 'categories', col: cols[0] ?? 0 };
}

/** Number of categories on a categorical axis: the named ones, or the data's largest index + 1. */
export function axisCategoryCount(info: Info, col: number, maxIndex: number): number {
  return Math.max(categoryNames(info, col).length, Math.floor(maxIndex) + 1, 1);
}

function rows(
  d: number,
  col: number,
  values: number[],
  heldRaw: readonly number[],
  st: Standardization | null,
): Grid {
  const x = new Float32Array(values.length * d);
  values.forEach((v, i) => {
    for (let c = 0; c < d; c++) x[i * d + c] = c === col ? v : (heldRaw[c] ?? 0);
  });
  return { n: values.length, d, xRaw: x, xStd: st ? standardizeRows(x, d, st) : x };
}

/**
 * `m` points along column `col` (spanning its domain), every other column held at `heldRaw`
 * (raw units, length d). With d = 1 it equals `grid1d`.
 */
export function grid1dSlice(
  domain: readonly [number, number][],
  heldRaw: readonly number[],
  col: number,
  m: number,
  st: Standardization | null,
): Grid {
  const [lo, hi] = domain[col] ?? [0, 1];
  const v = Array.from({ length: m }, (_, i) =>
    m <= 1 ? (lo + hi) / 2 : lo + ((hi - lo) * i) / (m - 1),
  );
  return rows(heldRaw.length, col, v, heldRaw, st);
}

/** One row per category index 0..k-1 along column `col`, the other columns held at `heldRaw`. */
export function gridCategories(
  heldRaw: readonly number[],
  col: number,
  k: number,
  st: Standardization | null,
): Grid {
  return rows(
    heldRaw.length,
    col,
    Array.from({ length: k }, (_, i) => i),
    heldRaw,
    st,
  );
}

/**
 * Deterministic horizontal offset in [-0.3, 0.3] for the i-th point, so points of one category
 * spread out instead of overlapping (no Math.random: the picture must not change on redraw).
 */
export function categoryJitter(i: number): number {
  const h = Math.sin((i + 1) * 12.9898) * 43758.5453;
  return (h - Math.floor(h) - 0.5) * 0.6;
}

/**
 * The model output at the K category rows as one series per curve: regression and binary give one
 * series (binary with two outputs: P(class 1)), multi-class one per class. Null when `out` does
 * not fit K rows.
 */
export function categorySeries(
  out: Float32Array,
  k: number,
  kind: 'regression' | 'binary' | 'multiclass',
  numClasses: number,
): Float32Array[] | null {
  if (k <= 0 || out.length === 0) return null;
  if (kind === 'multiclass') {
    if (out.length !== k * numClasses) return null;
    return Array.from({ length: numClasses }, (_, c) => {
      const y = new Float32Array(k);
      for (let i = 0; i < k; i++) y[i] = out[i * numClasses + c] ?? Number.NaN;
      return y;
    });
  }
  const width = Math.max(1, Math.round(out.length / k));
  const col = kind === 'binary' && width === 2 ? 1 : 0;
  const y = new Float32Array(k);
  for (let i = 0; i < k; i++) y[i] = out[i * width + col] ?? Number.NaN;
  return [y];
}
