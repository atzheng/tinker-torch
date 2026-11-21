import type { Graph } from '@tinker/core';

/** What `engine.probabilityRange` returns for one unfused probability loss. */
export type ProbRange = { node: string; min: number; max: number };

/** Three significant figures in plain decimals (no exponent): 1.27, -0.05, 0, 5.06, 1300, -0.000000001. */
function show(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1) return String(Number(v.toPrecision(3)));
  const digits = Math.min(20, Math.max(0, 2 - Math.floor(Math.log10(a))));
  return v
    .toFixed(digits)
    .replace(/(\.\d*?)0+$/, '$1')
    .replace(/\.$/, '');
}

/** What the prediction did, as the end of the sentence. */
function reachedText(v: number): string {
  if (Number.isNaN(v) || v === -Infinity) return 'is not a number';
  if (v === Infinity || Math.abs(v) >= 1e6) return 'became too large';
  return `reached ${show(v)}`;
}

/**
 * The plain sentence for the first probability loss whose prediction left (0, 1), or null when
 * every range is inside (0, 1).
 */
export function probabilitySentence(graph: Graph, ranges: readonly ProbRange[]): string | null {
  for (const r of ranges) {
    if (r.min > 0 && r.max < 1) continue;
    // the value furthest outside (0, 1); a NaN is reported as such
    const reached =
      Number.isNaN(r.min) || Number.isNaN(r.max) ? NaN : r.max - 1 >= 0 - r.min ? r.max : r.min;
    const type = graph.nodes.find((n) => n.id === r.node)?.type;
    const label = type === 'cross_entropy' ? 'Cross-entropy' : 'Log loss';
    return `${label} takes probabilities between 0 and 1, but the prediction ${reachedText(reached)}.`;
  }
  return null;
}
