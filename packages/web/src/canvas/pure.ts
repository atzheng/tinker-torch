/** Pure helpers for the layer view: no React, no store. Unit tested in test/canvasPure.test.ts. */
import {
  resolveEdge,
  type Analysis,
  type Graph,
  type LayerLayout,
  type Issue,
  type NodeId,
  type PortRef,
} from '@tinker/core';

/** Smallest on-screen font size (px) at which secondary drawing text is still drawn. */
export const MIN_TEXT_PX = 8;

/** Whether text of `fontUnits` abstract units is legible at `scale` px per unit (unknown scale: yes). */
export function legibleAt(scale: number | undefined, fontUnits: number): boolean {
  return scale === undefined || fontUnits * scale >= MIN_TEXT_PX;
}

const NAME_MIN_UNITS = 10;
const NAME_EM = 0.52;

/**
 * A piece name fitted to `maxW` units: the font is `base`, shrunk (to at least 10) so the text
 * fits, then raised again, within that fit and `cap` units, so it stays at least MIN_TEXT_PX on
 * screen. When even 10 units is too wide the text is cut with an ellipsis, so it never exceeds
 * `maxW` (estimated, 0.52 em per character) and is never hidden.
 */
export function fitName(
  scale: number | undefined,
  text: string,
  maxW: number,
  base: number,
  cap = base * 1.4,
): { text: string; size: number } {
  let t = text;
  let fit = maxW / Math.max(1, t.length * NAME_EM);
  if (fit < NAME_MIN_UNITS) {
    const chars = Math.max(2, Math.floor(maxW / (NAME_MIN_UNITS * NAME_EM)));
    if (chars < t.length) t = `${t.slice(0, chars - 1)}\u2026`;
    fit = Math.max(NAME_MIN_UNITS, maxW / Math.max(1, t.length * NAME_EM));
  }
  let f = Math.min(base, fit);
  if (scale !== undefined && f * scale < MIN_TEXT_PX) {
    f = Math.max(f, Math.min(MIN_TEXT_PX / scale, cap, fit));
  }
  return { text: t, size: f };
}

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

/** An output handle sits this far inside its piece's right edge, clear of the wire label's region. */
export const PORT_INSET = 28;

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

/**
 * Whether an edge or unit selection still fits the graph and its widths. Wire and node
 * selections are always live here (`useSelection` checks those). Pure; the store uses it to
 * clear a selection when an edit changes widths, so undo or a regrow cannot resurrect it.
 */
export function neuronSelectionLive(
  sel:
    | { kind: 'edge'; node: NodeId; i: number; j: number }
    | { kind: 'unit'; node: NodeId; index: number },
  graph: Graph,
  analysis?: Analysis,
): boolean {
  const node = graph.nodes.find((n) => n.id === sel.node);
  if (!node) return false;
  if (sel.kind === 'unit') {
    return sel.index >= 0 && !(node.type === 'linear' && sel.index >= node.out);
  }
  if (node.type !== 'linear' || sel.i < 0 || sel.j < 0 || sel.j >= node.out) return false;
  const inW = analysis?.inWidth[node.id];
  return !analysis || (inW !== undefined && sel.i < inW);
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

/** Size in abstract units of the pill drawn behind a wire's width text, at `scale` px per unit. */
export function labelSize(text: string, scale: number): { fs: number; lw: number; lh: number } {
  const fs = 12 / scale;
  return { fs, lw: text.length * 0.62 * fs + 0.8 * fs, lh: 1.4 * fs };
}

export type HitRect = { x: number; y: number; w: number; h: number };
export type HitShape =
  | ({ kind: 'node'; id: NodeId } & HitRect)
  | ({ kind: 'wire'; edge: number } & HitRect)
  | ({ kind: 'port'; node: NodeId } & HitRect);

/**
 * Hit shapes of the layer view in paint order (the last one wins a press): every piece, then every
 * wire label on top, so a label is always tappable. A label's rectangle is at least 32 screen px
 * each way. A piece's is its box, grown to 32 px tall when short, and horizontally only when the box
 * is narrower than 32 px, by at most 6 units, so it never reaches into the gap where labels sit.
 * `widths[i]` is the label text of wire edge i (null: no label, so no label shape). `outputs` lists
 * the pieces that have an output port: each gets a 32 px square at the middle of its right edge,
 * painted above the pieces and below the labels (the handle a wire is dragged from).
 */
export function layerHitShapes(
  layout: LayerLayout,
  scale: number,
  widths: readonly (number | null | undefined)[],
  outputs: ReadonlySet<NodeId> = new Set(),
): HitShape[] {
  const shapes: HitShape[] = [];
  for (const [id, b] of Object.entries(layout.nodes)) {
    const px = b.w * scale >= HIT_MIN_PX ? 0 : Math.min(hitPad(b.w, scale), 6);
    const py = hitPad(b.h, scale);
    shapes.push({ kind: 'node', id, x: b.x - px, y: b.y - py, w: b.w + 2 * px, h: b.h + 2 * py });
  }
  for (const [id, b] of Object.entries(layout.nodes)) {
    if (!outputs.has(id)) continue;
    const s = HIT_MIN_PX / scale;
    shapes.push({
      kind: 'port',
      node: id,
      x: b.x + b.w - PORT_INSET - s / 2,
      y: b.y + b.h / 2 - s / 2,
      w: s,
      h: s,
    });
  }
  const labels: { shape: HitShape & { kind: 'wire' }; cy: number; half: number }[] = [];
  for (const w of layout.wires) {
    const width = widths[w.edge];
    if (width === null || width === undefined) continue;
    const { lw, lh } = labelSize(String(width), scale);
    const hw = Math.max(lw, HIT_MIN_PX / scale);
    const hh = Math.max(lh, HIT_MIN_PX / scale);
    labels.push({
      shape: {
        kind: 'wire',
        edge: w.edge,
        x: w.labelAt[0] - hw / 2,
        y: w.labelAt[1] - hh / 2,
        w: hw,
        h: hh,
      },
      cy: w.labelAt[1],
      half: lh / 2,
    });
  }
  // Two labels whose rectangles overlap (a skip wire's label sits just above a row of labels):
  // the higher one moves up until its rectangle ends where the other begins (its centre stays inside), so each
  // label owns a full region.
  for (let pass = 0; pass < labels.length; pass++) {
    let moved = false;
    for (const a of labels) {
      for (const b of labels) {
        if (a === b || a.cy > b.cy) continue;
        const sa = a.shape;
        const sb = b.shape;
        const overlap =
          sa.x < sb.x + sb.w && sb.x < sa.x + sa.w && sa.y < sb.y + sb.h && sb.y < sa.y + sa.h;
        if (!overlap || (a.cy === b.cy && a.shape.edge > b.shape.edge)) continue;
        const y = Math.min(sa.y, sb.y - sa.h);
        if (y + sa.h >= a.cy && y !== sa.y) {
          sa.y = y;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
  for (const l of labels) shapes.push(l.shape);
  return shapes;
}

/** Topmost shape containing the point (abstract units), if any. Mirrors how the browser resolves a press. */
export function hitAt(shapes: readonly HitShape[], x: number, y: number): HitShape | undefined {
  for (let i = shapes.length - 1; i >= 0; i--) {
    const s = shapes[i];
    if (s && x >= s.x && x <= s.x + s.w && y >= s.y && y <= s.y + s.h) return s;
  }
  return undefined;
}
