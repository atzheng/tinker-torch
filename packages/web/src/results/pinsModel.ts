import { fitView, type Graph, type HostParams } from '@tinker/core';
import type { HistoryPoint } from '../state/trainSlice.js';

/** Pinned runs are plain data (no engine objects). At most this many are kept; a fourth is refused. */
export const PIN_LIMIT = 3;
/** A pinned history is cut to at most this many points, whatever the length of the run. */
export const PIN_POINTS = 200;

/** One point of a pinned loss curve. Same shape as a history point, without the batch series. */
export type PinPoint = { step: number; train: number; val: number | null };

export type Pin = {
  id: string;
  /** Colour slot 0..PIN_LIMIT-1 (the `--pin-N` theme token); kept until the pin is removed. */
  slot: number;
  label: string;
  /** The graph (including data settings and training settings) the run used. */
  graph: Graph;
  /** The weights at the time of the pin: loading the pin restores the fit from them. */
  host: HostParams;
  history: PinPoint[];
  /** The 1-D model curve at pin time (raw x and y on the fit plot's fixed grid), or null for 2-D data. */
  fit: { x: Float32Array; y: Float32Array } | null;
  /** Last validation loss (or training loss when there is no validation set), for the chip. */
  finalLoss: number | null;
};

/**
 * Cut `history` to at most `max` points: the history is split into `max - 1` runs of consecutive
 * points after the first; each run becomes its last point, carrying the last non-null validation
 * loss of the run (validation is null on steps where it was not evaluated). The first and the
 * last point are kept.
 */
export function decimateHistory(history: readonly HistoryPoint[], max = PIN_POINTS): PinPoint[] {
  const n = history.length;
  const plain = (p: HistoryPoint): PinPoint => ({ step: p.step, train: p.train, val: p.val });
  if (n <= max) return history.map(plain);
  const out: PinPoint[] = [];
  const first = history[0];
  if (first) out.push(plain(first));
  const buckets = max - 1;
  const span = n - 1; // indices 1..n-1 are split into `buckets` runs
  for (let b = 0; b < buckets; b++) {
    const from = 1 + Math.floor((b * span) / buckets);
    const to = Math.floor(((b + 1) * span) / buckets);
    const last = history[to];
    if (!last || to < from) continue;
    let val: number | null = null;
    for (let i = to; i >= from; i--) {
      const v = history[i]?.val;
      if (v !== null && v !== undefined) {
        val = v;
        break;
      }
    }
    out.push({ step: last.step, train: last.train, val });
  }
  return out;
}

/** A short name for a run from its graph: activations, hidden widths, optimizer and learning rate. */
export function pinLabel(graph: Graph): string {
  const acts: string[] = [];
  const widths: number[] = [];
  const lins = graph.nodes.filter((n) => n.type === 'linear');
  for (const n of graph.nodes) {
    if (n.type === 'activation' && !acts.includes(n.fn)) acts.push(n.fn);
  }
  for (const n of lins.slice(0, -1)) if (n.type === 'linear') widths.push(n.out);
  const t = graph.train;
  // `$...$` parts are typeset as maths by the chip (and stripped for titles): the widths read
  // `$16 \\times 8$` and the learning rate `$\\eta = 0.01$`.
  return [
    acts.join('/') || 'linear',
    widths.length > 1 ? `$${widths.join(' \\times ')}$` : widths.join(''),
    `${t.optimizer} $\\eta=${t.lr}$`,
  ]
    .filter((s) => s !== '')
    .join(' · ');
}

/** The lowest colour slot not used by `pins`, or null when all are taken. */
export function freeSlot(pins: readonly Pin[]): number | null {
  for (let s = 0; s < PIN_LIMIT; s++) if (!pins.some((p) => p.slot === s)) return s;
  return null;
}

let counter = 0;

/** Build a pin from the current run, or null when there is no history or no free slot. */
export function makePin(
  graph: Graph,
  host: HostParams,
  history: readonly HistoryPoint[],
  pins: readonly Pin[],
  fit: Pin['fit'] = null,
): Pin | null {
  const slot = freeSlot(pins);
  if (slot === null || history.length === 0) return null;
  const pts = decimateHistory(history);
  let finalLoss: number | null = pts[pts.length - 1]?.train ?? null;
  for (let i = pts.length - 1; i >= 0; i--) {
    const v = pts[i]?.val;
    if (v !== null && v !== undefined) {
      finalLoss = v;
      break;
    }
  }
  counter += 1;
  return {
    id: `pin-${counter}`,
    slot,
    label: pinLabel(graph),
    graph,
    host,
    history: pts,
    fit,
    finalLoss,
  };
}

/** Whether a pin stores a 1-D curve: one column that is numeric (a categorical axis draws none). */
export function shouldSnapshotFit(info: Parameters<typeof fitView>[0]): boolean {
  return info.columns.length === 1 && fitView(info).type === 'curve';
}

/**
 * The model curve on a 1-D grid for a pin: `predict(xStd, n)` is the store's `predictGrid` (output
 * [n, width], first output used). Null when there is no prediction. Copies, so the pin owns its data.
 */
export function snapshotFit(
  grid: { xRaw: Float32Array; xStd: Float32Array; n: number },
  predict: (x: Float32Array, n: number) => Float32Array | null,
): Pin['fit'] {
  const out = predict(grid.xStd, grid.n);
  if (!out) return null;
  const width = Math.max(1, Math.round(out.length / grid.n));
  const y = new Float32Array(grid.n);
  for (let i = 0; i < y.length; i++) y[i] = out[i * width] ?? Number.NaN;
  return { x: Float32Array.from(grid.xRaw), y };
}

export type LossBounds = { stepMax: number; lo: number; hi: number };

/** Widen `base` so it also covers `other` (same kind of range: both linear or both log). */
export function mergeBounds(base: LossBounds, other: LossBounds, log: boolean): LossBounds {
  return {
    stepMax: Math.max(base.stepMax, other.stepMax),
    lo: log ? Math.min(base.lo, other.lo) : 0,
    hi: Math.max(base.hi, other.hi),
  };
}
