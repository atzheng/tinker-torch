/** Pure helpers for the layer view: no React, no store. Unit tested in test/canvasPure.test.ts. */
import {
  parsePort,
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

/** Room right of an output dot's centre (its radius 4.5 plus a margin) that a wire label's hit rectangle leaves to the dot. */
const DOT_CLEAR = 6;

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

/** Pins: the ŷ and L role markers at a node's output (CONTRACTS.md section 10, "Role pins"). */
export type PinRole = 'pred' | 'loss';
/** Geometry of a pin in abstract units: radius and the offset of its centre from the output port. */
export const PIN = { r: 12, dx: 34, dy: 44 } as const;
/** Extra room right of the drawing so a pin on the last column is not cut off. */
export const PIN_ROOM_X = 24;

export const pinKey = (role: PinRole, node: NodeId): string => `${role}:${node}`;

/** Every pin of the graph: the ŷ pins in order, then the L pin. */
export function pinsOf(
  graph: Pick<Graph, 'predictions' | 'loss'>,
): { role: PinRole; node: NodeId }[] {
  const out: { role: PinRole; node: NodeId }[] = graph.predictions.map((p) => ({
    role: 'pred',
    node: parsePort(p).node,
  }));
  if (graph.loss !== null) out.push({ role: 'loss', node: parsePort(graph.loss).node });
  return out;
}

/** Whether a pin of this role is on this node. */
export function pinLive(
  graph: Pick<Graph, 'predictions' | 'loss'>,
  role: PinRole,
  node: NodeId,
): boolean {
  return pinsOf(graph).some((p) => p.role === role && p.node === node);
}

type Rect = { x: number; y: number; w: number; h: number };

/** Centre of a pin of `role` on the node `box`: up and right of the output port for ŷ, down and right for L. */
export function pinCentre(box: Rect, role: PinRole): [number, number] {
  return [box.x + box.w + PIN.dx, box.y + box.h / 2 + (role === 'pred' ? -PIN.dy : PIN.dy)];
}

/** Offsets (dx right of the box's right edge, dy from the port row) tried in turn, by role. */
const PIN_SLOTS: Record<PinRole, [number, number][]> = {
  pred: [
    [PIN.dx, -PIN.dy],
    [PIN.dx, PIN.dy],
    [PIN.dx, -PIN.dy - 30],
    [PIN.dx, PIN.dy + 30],
    [PIN.dx + 32, -PIN.dy],
    [PIN.dx + 32, PIN.dy],
  ],
  loss: [
    [PIN.dx, PIN.dy],
    [PIN.dx, -PIN.dy],
    [PIN.dx, PIN.dy + 30],
    [PIN.dx, -PIN.dy - 30],
    [PIN.dx + 32, PIN.dy],
    [PIN.dx + 32, -PIN.dy],
  ],
};

/** Clearance around a pin disc (and its ring) when it is kept off pieces and other pins. */
const PIN_CLEAR = PIN.r + 6;

/**
 * Where each pin sits. By default (`avoid` false) exactly `pinCentre`. With `avoid` (the graph has
 * pieces placed by hand, so another piece may be where the default spot is) a pin takes the first
 * of a short list of spots right of its piece (default, the other side, further out, further
 * right) whose disc clears every piece and every pin already placed; when none is free it keeps the
 * default. Marks are placed in the order given, so real pins win over the add pins after them.
 * Keys are `pinKey(role, node)`.
 */
export function placePins(
  boxes: Readonly<Record<NodeId, Rect | undefined>>,
  marks: readonly { role: PinRole; node: NodeId }[],
  avoid: boolean,
): Map<string, [number, number]> {
  const out = new Map<string, [number, number]>();
  const placed: [number, number][] = [];
  const hitsBox = (c: [number, number], b: Rect) =>
    c[0] + PIN_CLEAR > b.x &&
    c[0] - PIN_CLEAR < b.x + b.w &&
    c[1] + PIN_CLEAR > b.y &&
    c[1] - PIN_CLEAR < b.y + b.h;
  for (const { role, node } of marks) {
    const box = boxes[node];
    if (!box) continue;
    let at = pinCentre(box, role);
    if (avoid) {
      const mid = box.y + box.h / 2;
      for (const [dx, dy] of PIN_SLOTS[role]) {
        const c: [number, number] = [box.x + box.w + dx, mid + dy];
        const clear =
          !Object.values(boxes).some((b) => b && b !== box && hitsBox(c, b)) &&
          !placed.some((q) => Math.hypot(q[0] - c[0], q[1] - c[1]) < 2 * PIN_CLEAR);
        if (clear) {
          at = c;
          break;
        }
      }
    }
    placed.push(at);
    out.set(pinKey(role, node), at);
  }
  return out;
}

/**
 * The press rectangle of a pin: at least 32 screen px each way where there is room, but never
 * reaching into the piece (it starts right of the box) nor into the 32 px band around the output
 * port's row, where the wire label and the connect handle are, so those keep winning their presses.
 * `centre` is where the pin is drawn (`placePins`); left out, the default spot.
 */
export function pinHitRect(
  box: Rect,
  role: PinRole,
  scale: number,
  centre: readonly [number, number] = pinCentre(box, role),
): Rect {
  const [cx, cy] = centre;
  const half = Math.max(PIN.r, HIT_MIN_PX / 2 / scale);
  const mid = box.y + box.h / 2;
  const band = HIT_MIN_PX / 2 / scale;
  const x0 = Math.max(cx - half, box.x + box.w + 1);
  const below = cy > mid;
  const y0 = below ? Math.max(cy - half, mid + band) : cy - half;
  const y1 = below ? cy + half : Math.min(cy + half, mid - band);
  return { x: x0, y: y0, w: cx + half - x0, h: Math.max(1, y1 - y0) };
}

export type PlacedIssues = {
  /** Issues drawn at a pin, by `pinKey` (only when the caller passes `pinNode`). */
  byPin: Map<string, string[]>;
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
  /** The node a pin issue's pin sits on; when given, pin issues are drawn at the pin, not the node. */
  pinNode?: (pin: NonNullable<Issue['at']['pin']>) => NodeId | undefined,
): PlacedIssues {
  const out: PlacedIssues = { byPin: new Map(), byEdge: new Map(), byNode: new Map(), global: [] };
  const add = <K>(m: Map<K, string[]>, k: K, msg: string) => {
    const list = m.get(k);
    if (!list) m.set(k, [msg]);
    else if (!list.includes(msg)) list.push(msg);
  };
  for (const { at, message } of issues) {
    const pinOn = at.pin && pinNode ? pinNode(at.pin) : undefined;
    if (at.pin && pinOn !== undefined && nodeIds.has(pinOn)) {
      add(out.byPin, pinKey(at.pin.role === 'loss' ? 'loss' : 'pred', pinOn), message);
    } else if (at.edge !== undefined && wireEdges.has(at.edge)) add(out.byEdge, at.edge, message);
    else if (at.node !== undefined && nodeIds.has(at.node)) {
      const wire = wireOfNode(at.node);
      if (wire !== undefined && wireEdges.has(wire)) add(out.byEdge, wire, message);
      else add(out.byNode, at.node, message);
    } else if (!out.global.includes(message)) out.global.push(message);
  }
  return out;
}

/**
 * Pieces to outline red: any issue naming them, wherever the sentence is shown. With `skipPins`
 * (the layer view) a pin issue outlines its pin instead, not the piece.
 */
export function errorNodes(issues: readonly Issue[], skipPins = false): Set<NodeId> {
  const s = new Set<NodeId>();
  for (const i of issues) if (i.at.node !== undefined && !(skipPins && i.at.pin)) s.add(i.at.node);
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

/** TeX (no dollars) of a Weight's name: w1 is `w_{1}`, any other id is set upright. */
export function weightTex(id: NodeId): string {
  const m = /^w(\d+)$/.exec(id);
  if (m) return `w_{${m[1]}}`;
  return id === 'w' ? 'w' : `\\mathrm{${id.replace(/_/g, '\\_')}}`;
}

/** A weight value as shown in its pill: two decimals, fewer as it grows (the limit is 100). */
export function fmtWeight(v: number): string {
  const a = Math.abs(v);
  return a < 10 ? v.toFixed(2) : a < 100 ? v.toFixed(1) : v.toFixed(0);
}

/** The text of a weight pill without maths markers, for sizing and labels: `w1 = 0.42`. */
export function weightPillPlain(id: NodeId, value: number | undefined): string {
  return value === undefined ? id : `${id} = ${fmtWeight(value)}`;
}

/** Size in abstract units of a weight pill (a control, so a little roomier than a width label). */
export function pillSize(text: string, scale: number): { fs: number; lw: number; lh: number } {
  const fs = 12 / scale;
  return { fs, lw: (text.length * 0.6 + 1.4) * fs, lh: 2 * fs };
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
 * `widths[i]` is the label text of wire edge i (null: no label, so no label shape); a wire in
 * `pills` (edge to the plain text of its weight pill) has a pill instead, sized by `pillSize`. `outputs` lists
 * the pieces that have an output port: each gets a 32 px square centred on its output dot (the middle of its right edge),
 * painted above the pieces and below the labels (the handle a wire is dragged from).
 */
export function layerHitShapes(
  layout: LayerLayout,
  scale: number,
  widths: readonly (number | null | undefined)[],
  outputs: ReadonlySet<NodeId> = new Set(),
  pills: ReadonlyMap<number, string> = new Map(),
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
      x: b.x + b.w - s / 2, // centred on the drawn dot (outputPortAt)
      y: b.y + b.h / 2 - s / 2,
      w: s,
      h: s,
    });
  }
  const labels: {
    shape: HitShape & { kind: 'wire' };
    cy: number;
    half: number;
    textLeft: number;
  }[] = [];
  for (const w of layout.wires) {
    const pill = pills.get(w.edge);
    const width = widths[w.edge];
    if (pill === undefined && (width === null || width === undefined)) continue;
    const { lw, lh } = pill === undefined ? labelSize(String(width), scale) : pillSize(pill, scale);
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
      textLeft: w.labelAt[0] - lw / 2,
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
  // Zoomed out, a label's 32 px rectangle can cover the output dot to its left. The dot keeps its
  // press: the label's rectangle starts just right of any dot it would cover, never inside its text.
  const ports = shapes.filter((s): s is HitShape & { kind: 'port' } => s.kind === 'port');
  for (const l of labels) {
    const sl = l.shape;
    for (const p of ports) {
      const dx = p.x + p.w / 2;
      const dy = p.y + p.h / 2;
      if (dx < sl.x || dx > sl.x + sl.w || dy < sl.y || dy > sl.y + sl.h) continue;
      const left = Math.min(dx + DOT_CLEAR, l.textLeft);
      if (left <= sl.x) continue;
      sl.w -= left - sl.x;
      sl.x = left;
    }
  }
  for (const l of labels) shapes.push(l.shape);
  return shapes;
}

/**
 * The sizing text of every weight pill, by edge: a fixed width per weight (room for `-0.00`), so a
 * pill does not change size while its value moves.
 */
export function pillTexts(layout: LayerLayout): Map<number, string> {
  const out = new Map<number, string>();
  for (const w of layout.wires) {
    if (w.weightNode !== undefined) out.set(w.edge, `${w.weightNode} = -0.00`);
  }
  return out;
}

/** Topmost shape containing the point (abstract units), if any. Mirrors how the browser resolves a press. */
export function hitAt(shapes: readonly HitShape[], x: number, y: number): HitShape | undefined {
  for (let i = shapes.length - 1; i >= 0; i--) {
    const s = shapes[i];
    if (s && x >= s.x && x <= s.x + s.w && y >= s.y && y <= s.y + s.h) return s;
  }
  return undefined;
}
