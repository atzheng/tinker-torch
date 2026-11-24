/**
 * Drag-and-drop on the canvas, built on Pointer Events (`gestures/drag.ts`), never the HTML5 API.
 * Two gestures: a palette piece dragged onto a wire (insert) or a piece (swap), or a ready-made
 * group (a macro) dropped on a wire or on empty canvas (insert its primitives); and a wire dragged
 * from a piece's output port to an input port (connect).
 *
 * The pure half (`resolveDrop`, `editForDrop`) is unit tested in Node. The small store half holds
 * the gesture in progress so the palette (ghost), the canvas (highlights, preview line) and the
 * port handles can all see it; the canvas registers a handler that knows its layout and transform.
 */
import {
  CORE_PIECES,
  LAYOUT,
  addPiece,
  connect,
  inputPortsOf,
  insertMacro,
  isVariadicPort,
  insertOnWire,
  moveNodes,
  movePrediction,
  parsePort,
  resolveEdge,
  rewireEdge,
  setLoss,
  variadicSlotY,
  swapPiece,
  type EditResult,
  type Graph,
  type LayerLayout,
  type NodeId,
  type PaletteEntry,
  type PortRef,
} from '@tinker/core';
import { hitAt, layerHitShapes, HIT_MIN_PX, type HitShape } from './pure.js';

export type DragPayload =
  | { kind: 'palette'; entry: PaletteEntry }
  | { kind: 'wire'; from: NodeId }
  /** An existing wire's destination end dragged to another input port (one undo step). */
  | { kind: 'rewire'; edge: number; from: NodeId; to: PortRef }
  /** A ŷ or L pin dragged from the node carrying it to another node's output (one undo step). */
  | { kind: 'pin'; role: 'pred' | 'loss'; node: NodeId };

/** The piece a wire-like drag starts from (a new wire or a rewired one), else undefined. */
export function wireSource(p: DragPayload): NodeId | undefined {
  return p.kind === 'wire' || p.kind === 'rewire' ? p.from : undefined;
}

export type DropTarget =
  | { kind: 'wire'; edge: number }
  | { kind: 'node'; id: NodeId }
  | { kind: 'port'; node: NodeId; port: PortRef }
  /**
   * Empty canvas: a ready-made group goes in free-standing, and a source piece (Column, Input,
   * Target) is placed with its box's top-left at `at` (layout units, on the 8-unit grid).
   */
  | { kind: 'free'; at?: [number, number] };

/** Position of an output port (right edge, middle) in abstract units. */
export function outputPortAt(layout: LayerLayout, id: NodeId): [number, number] | undefined {
  const b = layout.nodes[id];
  return b ? [b.x + b.w, b.y + b.h / 2] : undefined;
}

/**
 * Position of input port `k` of `n` (left edge, spread evenly: the layout's rule). For a variadic
 * port that wires append to, `slot` of `count` is where the wire at that place enters
 * (`variadicSlotY`); left out, the port's centre.
 */
export function inputPortAt(
  layout: LayerLayout,
  id: NodeId,
  k: number,
  n: number,
  slot = 0,
  count = 1,
): [number, number] | undefined {
  const b = layout.nodes[id];
  return b ? [b.x, variadicSlotY(b, k, n, slot, count)] : undefined;
}

/**
 * Where a new wire from `src` would enter port `port` of node `id`. On a variadic port the wire
 * is appended, so the existing wires and this one share the band, ordered by the sources' heights
 * (the layout's rule); on any other port it replaces the wire there and enters at the centre.
 */
export function inputEndAt(
  graph: Graph,
  layout: LayerLayout,
  id: NodeId,
  port: string,
  src: NodeId,
): [number, number] | undefined {
  const node = graph.nodes.find((n) => n.id === id);
  if (!node) return undefined;
  const ports = inputPortsOf(node.type);
  const k = Math.max(0, ports.indexOf(port));
  if (!isVariadicPort(node.type, port)) return inputPortAt(layout, id, k, ports.length);
  const cy = (n: NodeId) => (layout.nodes[n]?.y ?? 0) + (layout.nodes[n]?.h ?? 0) / 2;
  const ends = graph.edges.flatMap((e) => {
    const r = resolveEdge(graph.nodes, e);
    return r.ok && r.to.node === id && r.to.port === port ? [r.from.node] : [];
  });
  const above = ends.filter((n) => cy(n) <= cy(src)).length;
  return inputPortAt(layout, id, k, ports.length, above, ends.length + 1);
}

/**
 * Whether a palette entry stands on its own when dropped on empty canvas: every piece and every
 * ready-made group does (unwired, it is inert until wired in; `analyse` says which pieces are).
 */
export function placeable(entry: PaletteEntry): boolean {
  return entry.kind === 'piece' || entry.kind === 'macro';
}

/** Top-left of a source piece dropped with its centre at `at`, on the 8-unit grid. */
function placeAt(at: { x: number; y: number }): [number, number] {
  const snap = (v: number) => Math.round(v / 8) * 8;
  return [snap(at.x - LAYOUT.boxW / 2), snap(at.y - LAYOUT.boxH / 2)];
}

/** The reference to write in an edge for input port `port` of `id`: bare for a single-port piece. */
export function portRefOf(id: NodeId, ports: readonly string[], port: string): PortRef {
  return ports.length === 1 ? id : `${id}.${port}`;
}

function distToSegment(px: number, py: number, a: readonly number[], b: readonly number[]): number {
  const [ax = 0, ay = 0] = a;
  const [bx = 0, by = 0] = b;
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** Whether a palette entry can go onto a wire: a piece with an input (every piece has an output), or a ready-made group. */
export function insertable(entry: PaletteEntry): boolean {
  return entry.kind === 'macro' || inputPortsOf(entry.make().type).length > 0;
}

/**
 * The drop target under a point (abstract units), or null when dropping here would do nothing.
 * Uses the same hit shapes as presses (`layerHitShapes`): a wire's label is its primary target;
 * the wire line itself (within `hitPx` screen px) is the fallback. Pieces are tested before wire
 * lines, as they are painted above them.
 */
export function resolveDrop(
  payload: DragPayload,
  graph: Graph,
  layout: LayerLayout,
  widths: readonly (number | null | undefined)[],
  scale: number,
  at: { x: number; y: number },
  hitPx: number = HIT_MIN_PX,
): DropTarget | null {
  const outputs = new Set(graph.nodes.map((n) => n.id));
  const shapes = layerHitShapes(layout, scale, widths, outputs);
  const nodeOf = (s: HitShape | undefined): NodeId | undefined =>
    s?.kind === 'node' ? s.id : s?.kind === 'port' ? s.node : undefined;

  if (payload.kind === 'wire' || payload.kind === 'rewire') {
    const id = nodeOf(
      hitAt(
        shapes.filter((s) => s.kind !== 'wire'),
        at.x,
        at.y,
      ),
    );
    const node = graph.nodes.find((n) => n.id === id);
    if (!node || node.id === payload.from) return null;
    const ports = inputPortsOf(node.type);
    if (ports.length === 0) return null;
    let best = 0;
    let bestD = Infinity;
    ports.forEach((_, k) => {
      const p = inputPortAt(layout, node.id, k, ports.length);
      const d = p ? Math.abs(p[1] - at.y) : Infinity;
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    });
    const port = ports[best];
    if (port === undefined) return null;
    const ref = portRefOf(node.id, ports, port);
    // Dropping a rewired wire back on the port it already ends at changes nothing.
    if (payload.kind === 'rewire' && ref === payload.to) return null;
    return { kind: 'port', node: node.id, port: ref };
  }

  if (payload.kind === 'pin') {
    // A pin lands on a piece (its body or its output port); not where it already is, nor on a
    // node that already carries a ŷ pin (the same node cannot be a ŷ twice).
    const id = nodeOf(
      hitAt(
        shapes.filter((s) => s.kind !== 'wire'),
        at.x,
        at.y,
      ),
    );
    return id !== undefined && pinCanLandOn(graph, payload, id) ? { kind: 'node', id } : null;
  }
  const entry = payload.entry;
  const hit = hitAt(shapes, at.x, at.y);
  if (hit?.kind === 'wire') return insertable(entry) ? { kind: 'wire', edge: hit.edge } : null;
  const id = nodeOf(hit);
  if (id !== undefined) {
    const node = graph.nodes.find((n) => n.id === id);
    if (!node || entry.kind === 'macro') return null;
    return CORE_PIECES[node.type].family === entry.family ? { kind: 'node', id } : null;
  }
  let best: DropTarget | null = null;
  let bestD = hitPx / 2 / scale;
  // A piece with no input (a source piece) cannot go on a wire: near one it is still free-standing.
  for (const w of insertable(entry) ? layout.wires : []) {
    for (let i = 0; i + 1 < w.points.length; i++) {
      const a = w.points[i];
      const b = w.points[i + 1];
      if (!a || !b) continue;
      const d = distToSegment(at.x, at.y, a, b);
      if (d <= bestD) {
        bestD = d;
        best = { kind: 'wire', edge: w.edge };
      }
    }
  }
  if (best) return best;
  // Anywhere else on the canvas puts any piece or group in free-standing, where it was dropped.
  return placeable(entry) ? { kind: 'free', at: placeAt(at) } : null;
}

/**
 * Whether a resolved drop lands on a stub (a ghost standing in for a piece of the other tab).
 * Pins and wire ends are rejected there; palette drops resolve to a wire or a free spot, or
 * replace a piece, and keep their own rules.
 */
export function dropsOnStub(
  target: DropTarget | null,
  ghosts: Readonly<Record<NodeId, unknown>>,
): boolean {
  if (target?.kind === 'node') return ghosts[target.id] !== undefined;
  if (target?.kind === 'port') return ghosts[parsePort(target.port).node] !== undefined;
  return false;
}

/** Whether the dragged pin may be dropped on node `id`: not on its own node, not on another ŷ for a ŷ. */
export function pinCanLandOn(
  graph: Graph,
  pin: { role: 'pred' | 'loss'; node: NodeId },
  id: NodeId,
): boolean {
  if (id === pin.node) return false;
  if (!graph.nodes.some((n) => n.id === id)) return false;
  return pin.role === 'loss' || !graph.predictions.some((p) => parsePort(p).node === id);
}

/**
 * Adds a piece on its own with its top-left at `at`. The layout cannot go left of or above its
 * margin, so a spot beyond it (the drawing is centred in a larger view) moves the pieces that
 * already have a position by the difference, in the same edit; pieces the automatic layout places
 * stay automatic. `boxes` are where the pieces are drawn now.
 */
function placeFree(
  graph: Graph,
  add: (g: Graph, at?: readonly [number, number], limit?: readonly [number, number]) => EditResult,
  at: readonly [number, number] | undefined,
  boxes: Readonly<Record<NodeId, { x: number; y: number }>> | undefined,
  limit?: readonly [number, number],
): EditResult {
  if (!at || !boxes) return add(graph, at, limit);
  const dx = Math.max(0, LAYOUT.pad - at[0]);
  const dy = Math.max(0, LAYOUT.pad - at[1]);
  const placed = new Set(
    Object.keys(graph.positions ?? {}).filter((id) => boxes[id] !== undefined),
  );
  const shift = placed.size > 0 && (dx > 0 || dy > 0);
  const where: [number, number][] = Object.entries(boxes).map(([id, b]) =>
    shift && placed.has(id) ? [b.x + dx, b.y + dy] : [b.x, b.y],
  );
  const slot = freeSlot([at[0] + dx, at[1] + dy], where);
  if (!shift) return add(graph, slot, limit);
  const moves: Record<NodeId, [number, number]> = {};
  for (const id of placed) {
    const b = boxes[id];
    if (b) moves[id] = [b.x + dx, b.y + dy];
  }
  const moved = moveNodes(graph, moves, limit);
  if (!moved.ok) return moved;
  return add(moved.graph, slot, limit);
}

/**
 * The spot nearest `at` (on the 8-unit grid, inside the margin) where a box does not overlap any of
 * the existing boxes (top-left corners in `others`, a gap of one grid step kept), so a piece dropped
 * on an occupied spot does not stack on the piece under it.
 */
export function freeSlot(
  at: readonly [number, number],
  others: readonly (readonly [number, number])[],
): [number, number] {
  const gap = 8;
  const free = (x: number, y: number) =>
    others.every(
      ([ox, oy]) =>
        x + LAYOUT.boxW + gap <= ox ||
        ox + LAYOUT.boxW + gap <= x ||
        y + LAYOUT.boxH + gap <= oy ||
        oy + LAYOUT.boxH + gap <= y,
    );
  if (free(at[0], at[1])) return [at[0], at[1]];
  const snap = (v: number) => Math.round(v / 8) * 8;
  let best: [number, number] | null = null;
  let bestD = Infinity;
  for (let ring = 1; ring <= 80 && best === null; ring++) {
    for (let i = -ring; i <= ring; i++) {
      for (const [sx, sy] of [
        [i, -ring],
        [i, ring],
        [-ring, i],
        [ring, i],
      ] as const) {
        const x = snap(at[0]) + sx * 8;
        const y = snap(at[1]) + sy * 8;
        if (x < LAYOUT.pad || y < LAYOUT.pad || !free(x, y)) continue;
        const d = (x - at[0]) ** 2 + (y - at[1]) ** 2;
        if (d < bestD) {
          bestD = d;
          best = [x, y];
        }
      }
    }
  }
  return best ?? [at[0], at[1]];
}

/**
 * The far corner a piece's top-left may reach so that its whole box stays visible: the canvas
 * element's bottom-right corner in layout units, minus one box. `t` maps layout units to element pixels.
 */
export function visibleLimit(
  t: { scale: number; tx: number; ty: number },
  view: { w: number; h: number },
): [number, number] {
  return [(view.w - t.tx) / t.scale - LAYOUT.boxW, (view.h - t.ty) / t.scale - LAYOUT.boxH];
}

/** The pure edit a drop stands for, or null when the pair makes no edit. */
export function editForDrop(
  graph: Graph,
  payload: DragPayload,
  target: DropTarget,
  opts?: {
    locked?: readonly NodeId[];
    boxes?: Readonly<Record<NodeId, { x: number; y: number }>>;
    /** Far corner a piece's top-left may reach (the visible canvas; `visibleLimit`), layout units. */
    limit?: readonly [number, number];
  },
): EditResult | null {
  if (payload.kind === 'wire') {
    return target.kind === 'port' ? connect(graph, payload.from, target.port, opts) : null;
  }
  if (payload.kind === 'rewire') {
    return target.kind === 'port' ? rewireEdge(graph, payload.edge, 'to', target.port, opts) : null;
  }
  if (payload.kind === 'pin') {
    if (target.kind !== 'node') return null;
    if (payload.role === 'loss') return setLoss(graph, target.id, opts);
    const index = graph.predictions.findIndex((p) => parsePort(p).node === payload.node);
    return movePrediction(graph, index, target.id, opts);
  }
  const entry = payload.entry;
  if (entry.kind === 'macro') {
    if (target.kind === 'wire') return insertMacro(graph, entry.key, { edge: target.edge });
    return target.kind === 'free'
      ? placeFree(
          graph,
          (g, at, lim) => insertMacro(g, entry.key, undefined, at, lim),
          target.at,
          opts?.boxes,
          opts?.limit,
        )
      : null;
  }
  const made = entry.make();
  if (target.kind === 'wire') return insertOnWire(graph, target.edge, made, opts);
  if (target.kind === 'node') return swapPiece(graph, target.id, made, opts);
  if (target.kind === 'free' && placeable(entry)) {
    return placeFree(
      graph,
      (g, at, lim) => addPiece(g, made, at, lim),
      target.at,
      opts?.boxes,
      opts?.limit,
    );
  }
  return null;
}

// ---- the gesture in progress ------------------------------------------------------------------

export type DndState = {
  payload: DragPayload;
  /** Pointer position, client coordinates. */
  x: number;
  y: number;
  target: DropTarget | null;
};

export type DndHandler = {
  /** The drop target under a client point. */
  resolve: (payload: DragPayload, x: number, y: number) => DropTarget | null;
  /** Apply the drop (store edit, selection, toast). */
  commit: (payload: DragPayload, target: DropTarget) => void;
};

let state: DndState | null = null;
/** The palette entry picked for tap-to-add: the next tap on a valid wire or piece applies it. */
let armed: PaletteEntry | null = null;
let handler: DndHandler | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export const dnd = {
  get: (): DndState | null => state,
  subscribe(l: () => void): () => void {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  /** The canvas registers while mounted; returns the unregister function. */
  setHandler(h: DndHandler): () => void {
    handler = h;
    return () => {
      if (handler === h) handler = null;
    };
  },
  armed: (): PaletteEntry | null => armed,
  /** Pick (or with null, drop) the palette entry for tap-to-add; picking the same one again drops it. */
  arm(entry: PaletteEntry | null) {
    armed = entry && armed?.key === entry.key ? null : entry;
    emit();
  },
  /** Apply the armed entry at a client point. Returns whether an edit was attempted; disarms either way. */
  tapAt(x: number, y: number): boolean {
    const entry = armed;
    armed = null;
    emit();
    if (!entry || !handler) return false;
    const payload: DragPayload = { kind: 'palette', entry };
    const target = handler.resolve(payload, x, y);
    if (target) handler.commit(payload, target);
    return target !== null;
  },
  begin(payload: DragPayload, x: number, y: number) {
    state = { payload, x, y, target: handler?.resolve(payload, x, y) ?? null };
    emit();
  },
  move(x: number, y: number) {
    if (!state) return;
    state = { ...state, x, y, target: handler?.resolve(state.payload, x, y) ?? null };
    emit();
  },
  /** Pointer released: apply the drop when there is a target, otherwise cancel quietly. */
  end(x: number, y: number) {
    const s = state;
    state = null;
    emit();
    if (!s || !handler) return;
    const target = handler.resolve(s.payload, x, y);
    if (target) handler.commit(s.payload, target);
  },
  cancel() {
    if (!state) return;
    state = null;
    emit();
  },
};

/**
 * Every wire and piece a palette entry could be dropped on or tapped onto, for the highlights
 * shown while an entry is dragged or armed. Mirrors `resolveDrop`: wires for pieces that have an
 * input and an output (or any wire for a ready-made group), pieces of the same family.
 */
export function dropCandidates(
  entry: PaletteEntry,
  graph: Graph,
): { edges: ReadonlySet<number>; nodes: ReadonlySet<NodeId> } {
  const edges = new Set<number>();
  const nodes = new Set<NodeId>();
  if (insertable(entry)) graph.edges.forEach((_, i) => edges.add(i));
  if (entry.kind === 'piece') {
    for (const n of graph.nodes) if (CORE_PIECES[n.type].family === entry.family) nodes.add(n.id);
  }
  return { edges, nodes };
}
