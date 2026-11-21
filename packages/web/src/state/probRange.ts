import type { Graph } from '@tinker/core';

/** What `engine.probabilityRange` returns for one unfused probability loss. */
export type ProbRange = { node: string; min: number; max: number };

/** Three significant figures, no negative zero: 1.27, -0.05, 0. */
function show(v: number): string {
  if (v === 0) return '0';
  return String(Number(v.toPrecision(3)));
}

/**
 * The plain sentence for the first probability loss whose prediction left (0, 1), or null when
 * every range is inside (0, 1) or not a number (a NaN is real divergence, not a range problem).
 */
export function probabilitySentence(graph: Graph, ranges: readonly ProbRange[]): string | null {
  for (const r of ranges) {
    if (Number.isNaN(r.min) || Number.isNaN(r.max)) continue;
    if (r.min > 0 && r.max < 1) continue;
    const high = r.max - 1;
    const low = 0 - r.min;
    const reached = high >= low ? r.max : r.min;
    const type = graph.nodes.find((n) => n.id === r.node)?.type;
    const label = type === 'cross_entropy' ? 'Cross-entropy' : 'Log loss';
    return `${label} takes probabilities between 0 and 1, but the prediction reached ${show(reached)}.`;
  }
  return null;
}
