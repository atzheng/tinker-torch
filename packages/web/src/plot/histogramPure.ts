/** Pure helpers for the target histogram panel (tinker-6t7.8). No DOM, no React. */
import { parsePort, type Graph } from '@tinker/core';

/** Where the line may go, as a probability, for a sigmoid prediction: keeps the logit finite. */
export const P_MIN = 0.005;
export const P_MAX = 0.995;

export const logit = (p: number): number => Math.log(p / (1 - p));
export const sigmoid = (b: number): number => 1 / (1 + Math.exp(-b));

export type Chain = {
  /** Does an Input (or Feature) reach the prediction? Then there is no single prediction number. */
  hasInput: boolean;
  /** The Bias piece the line drags, and whether the prediction is sigmoid(b) rather than b. */
  param: { node: string; sigmoid: boolean } | null;
};

/**
 * Walks upstream from the prediction node. A Bias at the end of a chain of at most one sigmoid
 * Link is draggable; anything else (more pieces, several parents) is shown but not draggable.
 */
export function predictionChain(graph: Pick<Graph, 'nodes' | 'edges'>, pred: string): Chain {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const parents = (id: string): string[] =>
    graph.edges.filter(([, dst]) => parsePort(dst).node === id).map(([src]) => parsePort(src).node);
  // any Input or Feature upstream
  const seen = new Set<string>();
  const todo = [pred];
  let hasInput = false;
  while (todo.length > 0) {
    const id = todo.pop() as string;
    if (seen.has(id)) continue;
    seen.add(id);
    const t = byId.get(id)?.type;
    if (t === 'input' || t === 'feature') hasInput = true;
    todo.push(...parents(id));
  }
  // the simple chain: bias, or sigmoid link <- bias
  // a Bias with an incoming edge outputs parent + b, so the line would not follow the cursor
  const free = (id: string) => byId.get(id)?.type === 'bias' && parents(id).length === 0;
  const first = byId.get(pred);
  if (first?.type === 'bias' && free(pred))
    return { hasInput, param: { node: pred, sigmoid: false } };
  if (first?.type === 'link' && first.fn === 'sigmoid') {
    const ps = parents(pred);
    const only = ps.length === 1 ? ps[0] : undefined;
    if (only !== undefined && free(only)) {
      return { hasInput, param: { node: only, sigmoid: true } };
    }
  }
  return { hasInput, param: null };
}

/** The line's value to the parameter: b = yhat, or b = logit(yhat) with yhat kept off 0 and 1. */
export function valueToParam(v: number, sig: boolean): number {
  return sig ? logit(Math.min(P_MAX, Math.max(P_MIN, v))) : v;
}

export type Bins = { lo: number; hi: number; counts: number[]; width: number; max: number };

/**
 * Histogram bins. Binary targets get two bars centred on 0 and 1 (width 1);
 * regression targets get about sqrt(n) equal bins (5 to 24) over the data range.
 */
export function histogramBins(values: ArrayLike<number>, binary: boolean): Bins {
  if (binary) {
    const counts = [0, 0];
    for (let i = 0; i < values.length; i++) {
      const k = (values[i] ?? 0) >= 0.5 ? 1 : 0;
      counts[k] = (counts[k] ?? 0) + 1;
    }
    return { lo: -0.5, hi: 1.5, counts, width: 1, max: Math.max(1, ...counts) };
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? Number.NaN;
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!(hi >= lo)) return { lo: 0, hi: 1, counts: [], width: 1, max: 1 };
  if (hi === lo) {
    lo -= 0.5;
    hi += 0.5;
  }
  const k = Math.min(24, Math.max(5, Math.ceil(Math.sqrt(values.length))));
  const width = (hi - lo) / k;
  const counts = new Array<number>(k).fill(0);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? Number.NaN;
    if (!Number.isFinite(v)) continue;
    const j = Math.min(k - 1, Math.floor((v - lo) / width));
    counts[j] = (counts[j] ?? 0) + 1;
  }
  return { lo, hi, counts, width, max: Math.max(1, ...counts) };
}

export type Domain = { dlo: number; dhi: number };

/** The range the line may take: the data range plus 10% of its span either side (binary: -0.1 to 1.1). */
export function lineDomain(values: ArrayLike<number>, binary: boolean): Domain {
  if (binary) return { dlo: -0.1, dhi: 1.1 };
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? Number.NaN;
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!(hi >= lo)) return { dlo: -1, dhi: 1 };
  const s = hi - lo || 1;
  return { dlo: lo - 0.1 * s, dhi: hi + 0.1 * s };
}

/**
 * The horizontal axis. Without squares the line's range fills the plot width. With squares the
 * axis must have the same pixels per unit as the squares' height, and the tallest square that can
 * occur (data at one end of the range, the line at the other) must fit the plot height, so the
 * scale is the smaller of width/range and height/range and the range is centred. The scale
 * depends on the data range, never on the line, so dragging does not rescale the plot.
 */
export function axisFor(
  dom: Domain,
  pw: number,
  ph: number,
  squares: boolean,
): { xlo: number; xhi: number; unit: number } {
  const range = dom.dhi - dom.dlo || 1;
  const unit = squares ? Math.min(pw / range, ph / range) : pw / range;
  const shown = pw / unit;
  const mid = (dom.dlo + dom.dhi) / 2;
  return { xlo: mid - shown / 2, xhi: mid + shown / 2, unit };
}

/** One square per example: side |y - yhat|. Largest first, so small squares stay visible on top. */
export function squareSides(values: ArrayLike<number>, yhat: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? Number.NaN;
    if (Number.isFinite(v)) out.push(Math.abs(v - yhat));
  }
  return out.sort((a, b) => b - a);
}
