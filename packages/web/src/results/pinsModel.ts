import type { Graph, HostParams } from '@tinker/core';
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
  return [acts.join('/') || 'linear', widths.join('x'), `${t.optimizer} ${t.lr}`]
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
    finalLoss,
  };
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
