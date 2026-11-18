/** Pure helpers for the layer view: no React, no store. Unit tested in test/canvasPure.test.ts. */
import { resolveEdge, type Graph, type Issue, type NodeId, type PortRef } from '@tinker/core';

export type Transform = { scale: number; tx: number; ty: number };

/** Scale and centre a drawing of `size` abstract units into a `view` of pixels, with a margin. */
export function fitTransform(
  size: readonly [number, number],
  view: { w: number; h: number },
  margin = 24,
  maxScale = 1.4,
): Transform {
  const sw = Math.max(1, size[0]);
  const sh = Math.max(1, size[1]);
  const scale = Math.max(
    0.05,
    Math.min(maxScale, (view.w - 2 * margin) / sw, (view.h - 2 * margin) / sh),
  );
  return { scale, tx: (view.w - sw * scale) / 2, ty: (view.h - sh * scale) / 2 };
}

/** Smallest touch target, in screen pixels (CONTRACTS.md section 10). */
export const HIT_MIN_PX = 32;

/**
 * Extra margin to add on each side of a shape that is `size` abstract units across so its hit
 * shape is at least `minPx` on screen at `scale`. Drawn shapes keep their size; only the
 * transparent hit shape grows (neuron view scales down to about 0.5).
 */
export function hitPad(size: number, scale: number, minPx = HIT_MIN_PX): number {
  return Math.max(0, (minPx / scale - size) / 2);
}

/** Index of the first edge with these endpoints (as stored), or -1. */
export function edgeIndexOf(graph: Graph, from: PortRef, to: PortRef): number {
  return graph.edges.findIndex((e) => e[0] === from && e[1] === to);
}

export type PlacedIssues = {
  /** Issues drawn at a wire (that has geometry). */
  byEdge: Map<number, string[]>;
  /** Issues drawn at a piece. */
  byNode: Map<NodeId, string[]>;
  /** Issues with no place on the canvas: one line each. */
  global: string[];
};

/**
 * Decide where each issue is shown. An issue with an edge that has a wire goes on the wire (its
 * node is still outlined, see `errorNodes`); otherwise on its node when that exists; otherwise it
 * is global. An edge that does not resolve has no wire, so it falls back to its node or to the
 * global line.
 */
export function placeIssues(
  issues: readonly Issue[],
  wireEdges: ReadonlySet<number>,
  nodeIds: ReadonlySet<NodeId>,
  wireOfNode: (node: NodeId) => number | undefined = () => undefined,
): PlacedIssues {
  const out: PlacedIssues = { byEdge: new Map(), byNode: new Map(), global: [] };
  const add = <K>(m: Map<K, string[]>, k: K, msg: string) => {
    const list = m.get(k);
    if (!list) m.set(k, [msg]);
    else if (!list.includes(msg)) list.push(msg);
  };
  for (const { at, message } of issues) {
    if (at.edge !== undefined && wireEdges.has(at.edge)) add(out.byEdge, at.edge, message);
    else if (at.node !== undefined && nodeIds.has(at.node)) {
      const wire = wireOfNode(at.node);
      if (wire !== undefined && wireEdges.has(wire)) add(out.byEdge, wire, message);
      else add(out.byNode, at.node, message);
    } else if (!out.global.includes(message)) out.global.push(message);
  }
  return out;
}

/**
 * The wire a node-level issue is really about: for an Output piece (its width, link or loss
 * does not fit) it is the wire into the loss, where the student sees the width that is wrong.
 * Other pieces have none, so their sentence sits at the piece.
 */
export function wireOfNode(graph: Graph, node: NodeId): number | undefined {
  const n = graph.nodes.find((x) => x.id === node);
  if (n?.type !== 'output') return undefined;
  const i = graph.edges.findIndex((e) => {
    const r = resolveEdge(graph.nodes, e);
    if ('error' in r) return false;
    const to = graph.nodes.find((x) => x.id === r.to.node);
    return r.from.node === node && to?.type === 'loss';
  });
  return i >= 0 ? i : undefined;
}

/** Pieces to outline red: any issue naming them, wherever the sentence is shown. */
export function errorNodes(issues: readonly Issue[]): Set<NodeId> {
  const s = new Set<NodeId>();
  for (const i of issues) if (i.at.node !== undefined) s.add(i.at.node);
  return s;
}

/** Indices of the edges that resolve (the layout draws exactly these). */
export function resolvedEdges(graph: Graph): Set<number> {
  const s = new Set<number>();
  graph.edges.forEach((e, i) => {
    if (!('error' in resolveEdge(graph.nodes, e))) s.add(i);
  });
  return s;
}

export type HeatMap = { rows: number; cols: number; values: number[]; maxAbs: number };

/**
 * Weights `W` (row-major [rows, cols]) reduced to at most `maxCells` per side by averaging
 * blocks, for the small heat map in a Linear box. Never writes into `W`. Returns null when
 * the array does not have rows * cols entries.
 */
export function heatMap(
  W: Float32Array,
  rows: number,
  cols: number,
  maxCells = 10,
): HeatMap | null {
  if (rows < 1 || cols < 1 || W.length !== rows * cols) return null;
  const r = Math.min(rows, maxCells);
  const c = Math.min(cols, maxCells);
  const values: number[] = [];
  let maxAbs = 0;
  for (let bi = 0; bi < r; bi++) {
    const i0 = Math.floor((bi * rows) / r);
    const i1 = Math.floor(((bi + 1) * rows) / r);
    for (let bj = 0; bj < c; bj++) {
      const j0 = Math.floor((bj * cols) / c);
      const j1 = Math.floor(((bj + 1) * cols) / c);
      let sum = 0;
      for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) sum += W[i * cols + j] ?? 0;
      const v = sum / ((i1 - i0) * (j1 - j0));
      values.push(v);
      maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  return { rows: r, cols: c, values, maxAbs };
}

export function pointsToPath(points: readonly (readonly [number, number])[]): string {
  return points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x} ${y}`).join(' ');
}

export type InspectorPlacement = { left: number } & ({ top: number } | { bottom: number });

/**
 * Where to put the inspector (`width` px wide, about `needH` px tall) for a target spanning
 * `top`..`bottom` px at horizontal centre `cx` inside a `view`: centred under the target when there
 * is room below, else above it (anchored by `bottom`, so its height need not be known), kept
 * inside the view horizontally.
 */
export function placeInspector(
  target: { cx: number; top: number; bottom: number },
  view: { w: number; h: number },
  width: number,
  needH = 230,
  gap = 16,
): InspectorPlacement {
  const left = Math.min(Math.max(4, target.cx - width / 2), Math.max(4, view.w - width - 4));
  if (view.h - (target.bottom + gap) >= needH || target.top - gap < needH) {
    return { left, top: Math.min(target.bottom + gap, Math.max(4, view.h - 120)) };
  }
  return { left, bottom: view.h - (target.top - gap) };
}
