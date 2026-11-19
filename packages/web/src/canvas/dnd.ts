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
  connect,
  inputPortsOf,
  insertMacro,
  insertOnWire,
  movePrediction,
  parsePort,
  rewireEdge,
  setLoss,
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
  /** Empty canvas: a ready-made group goes in free-standing. */
  | { kind: 'free' };

/** Position of an output port (right edge, middle) in abstract units. */
export function outputPortAt(layout: LayerLayout, id: NodeId): [number, number] | undefined {
  const b = layout.nodes[id];
  return b ? [b.x + b.w, b.y + b.h / 2] : undefined;
}

/** Position of input port `k` of `n` (left edge, spread evenly: the layout's rule). */
export function inputPortAt(
  layout: LayerLayout,
  id: NodeId,
  k: number,
  n: number,
): [number, number] | undefined {
  const b = layout.nodes[id];
  return b ? [b.x, b.y + (b.h * (k + 1)) / (n + 1)] : undefined;
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
  if (!insertable(entry)) return null;
  let best: DropTarget | null = null;
  let bestD = hitPx / 2 / scale;
  for (const w of layout.wires) {
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
  // A ready-made group needs no wire: anywhere else on the canvas puts it in free-standing.
  return best ?? (entry.kind === 'macro' ? { kind: 'free' } : null);
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

/** The pure edit a drop stands for, or null when the pair makes no edit. */
export function editForDrop(
  graph: Graph,
  payload: DragPayload,
  target: DropTarget,
  opts?: { locked?: readonly NodeId[] },
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
    return target.kind === 'free' ? insertMacro(graph, entry.key) : null;
  }
  const made = entry.make();
  if (target.kind === 'wire') return insertOnWire(graph, target.edge, made, opts);
  if (target.kind === 'node') return swapPiece(graph, target.id, made, opts);
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
