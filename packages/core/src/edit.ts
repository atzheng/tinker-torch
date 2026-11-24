/**
 * Pure graph edits behind every canvas gesture (CONTRACTS.md section 8).
 *
 * Every edit returns a new graph and never touches its input. Every ok result has been through
 * `parseGraph`, so it always re-parses (defaults filled, huber delta set). Edits may leave a graph
 * that `analyse` reports as not trainable; they refuse only what cannot be represented (cycles,
 * unknown nodes or ports, a swap across families, a locked node, a value the schema rejects).
 * Pins (`predictions`, `loss`) follow their node: deleting a pinned node moves the pin to what fed
 * it, or drops it. The pin edits (`setLoss`, `addPrediction`, `movePrediction`, `removePrediction`)
 * change which output carries a role; a pin on a locked piece is fixed, so they refuse when the
 * pin's old or new node is locked.
 */
import { analyse, type Analysis } from './analyse.js';
import {
  parseGraph,
  parsePort,
  resolvePin,
  resolveEdge,
  inputPortsOf,
  isVariadicPort,
  optionalPortsOf,
  variadicPortsOf,
  type Edge,
  type Endpoint,
  type Graph,
  type Node,
  type NodeBody,
  type NodeId,
  type PortRef,
} from './graph.js';
import { CORE_PIECES, PALETTE, sameShape } from './registry.js';
import type { Namer } from './macros.js';
import { LAYOUT } from './layout.js';
import { SETS } from './data/sets.js';

export type EditResult = { ok: true; graph: Graph; select?: NodeId } | { ok: false; error: string };

/** `locked` is the preset field (CONTRACTS.md section 12). A locked node is never changed. */
export type EditOptions = { locked?: readonly NodeId[] };

export const LOCKED_MESSAGE = 'This piece is locked.';

const fail = (error: string): EditResult => ({ ok: false, error });

/** Nodes whose own settings or incoming wires an edit changes must not be locked. */
function lockedAmong(opts: EditOptions | undefined, ids: Iterable<NodeId>): boolean {
  const locked = opts?.locked;
  if (!locked || locked.length === 0) return false;
  for (const id of ids) if (locked.includes(id)) return true;
  return false;
}

/** Re-parses the edited graph, which validates every setting and gives back a normalised copy. */
function finish(g: Graph, nodes: Node[], edges: Edge[], select?: NodeId): EditResult {
  const parsed = parseGraph({ ...g, nodes, edges });
  if (!parsed.ok) return fail(parsed.error);
  return select === undefined
    ? { ok: true, graph: parsed.graph }
    : { ok: true, graph: parsed.graph, select };
}

type Resolved = { from: Endpoint; to: Endpoint } | null;

/** Resolved ends of every edge; null for an edge that cannot be resolved (never after parseGraph). */
function resolveAll(g: Graph): Resolved[] {
  return g.edges.map((e) => {
    const r = resolveEdge(g.nodes, e);
    return r.ok ? { from: r.from, to: r.to } : null;
  });
}

/** How an edge into `port` of node `id` is written: bare when the node has one input port. */
function toRef(g: Graph, id: NodeId, port: string): PortRef {
  const node = g.nodes.find((n) => n.id === id);
  return node && inputPortsOf(node.type).length === 1 ? id : `${id}.${port}`;
}

function findNode(g: Graph, id: NodeId): Node | undefined {
  return g.nodes.find((n) => n.id === id);
}

function noNode(id: NodeId): EditResult {
  return fail(`There is no piece called "${id}".`);
}

/** Id stems that differ from the piece type: columns are x1.., weights w1... */
const ID_STEMS: Record<string, string> = { column: 'x', scalew: 'w' };

/** Smallest free id of the form `${type}${k}`, k from 1. */
export function freshId(g: Graph, type: string): NodeId {
  const used = new Set(g.nodes.map((n) => n.id));
  const stem = ID_STEMS[type] ?? type;
  for (let k = 1; ; k++) if (!used.has(`${stem}${k}`)) return `${stem}${k}`;
}

/**
 * A new Column's name: the first dataset column no other Column node uses (x1 when all are used).
 * Other pieces pass through unchanged.
 */
export function withFreshColumn(g: Graph, piece: NodeBody): NodeBody {
  if (piece.type !== 'column') return piece;
  const d = SETS[g.data.set].d;
  const used = new Set(g.nodes.flatMap((n) => (n.type === 'column' ? [n.name] : [])));
  for (let i = 1; i <= d; i++) if (!used.has(`x${i}`)) return { ...piece, name: `x${i}` };
  return { ...piece, name: 'x1' };
}

/** Whether `target` can be reached from `start` by following wires forwards. */
function reaches(resolved: Resolved[], start: NodeId, target: NodeId): boolean {
  const seen = new Set<NodeId>([start]);
  const stack = [start];
  while (stack.length > 0) {
    const cur = stack.pop();
    if (cur === undefined) break;
    if (cur === target) return true;
    for (const r of resolved) {
      if (r && r.from.node === cur && !seen.has(r.to.node)) {
        seen.add(r.to.node);
        stack.push(r.to.node);
      }
    }
  }
  return false;
}

// ------------------------------------------------------------------ edits

/**
 * Puts a new piece on a wire: `a -> b` becomes `a -> new -> b`. The first half keeps the wire's
 * place in the edge list and the second half follows it. An Add goes in on its `a` input.
 * Refused for pieces with no input port (Input, Target, Weights), and when the wire ends at
 * a locked node. Selects the new node.
 */
export function insertOnWire(
  g: Graph,
  edgeIndex: number,
  piece: NodeBody,
  opts?: EditOptions,
): EditResult {
  const edge = g.edges[edgeIndex];
  const r = resolveAll(g)[edgeIndex];
  if (edge === undefined || !r) return fail('That wire does not exist.');
  const ports = inputPortsOf(piece.type);
  const first = ports[0];
  if (first === undefined) {
    return fail('That piece has no input, so it cannot go on a wire.');
  }
  if (lockedAmong(opts, [r.to.node])) return fail(LOCKED_MESSAGE);
  const id = freshId(g, piece.type);
  const node = { id, ...withFreshColumn(g, piece) } as Node;
  const edges = g.edges.slice();
  edges.splice(edgeIndex, 1, [edge[0], ports.length === 1 ? id : `${id}.${first}`], [id, edge[1]]);
  // A pin on the source follows the data when this is the source's only wire out.
  const src = r.from.node;
  const onlyOut = resolveAll(g).filter((x) => x?.from.node === src).length === 1;
  const move = (ref: PortRef): PortRef => (onlyOut && pinNode(ref) === src ? id : ref);
  const parsed = parseGraph({
    ...g,
    nodes: [...g.nodes, node],
    edges,
    predictions: g.predictions.map(move),
    loss: g.loss === null ? null : move(g.loss),
  });
  if (!parsed.ok) return fail(parsed.error);
  return { ok: true, graph: parsed.graph, select: id };
}

/**
 * Puts a new piece on the canvas with no wires (a source piece to start from, or a Sum, Weight or
 * Bias to wire up by hand). `at` is the top-left corner of its box in layout units (kept within the same bounds as
 * `moveNodes`); left out, the automatic layout places it. Selects the new node. One edit, so one
 * undo step. Any piece may be placed; it is inert until wired into what a ŷ or L pin needs (analyse).
 * A Weights tap reads the first Linear (refused when there is none).
 */
export function addPiece(
  g: Graph,
  piece: NodeBody,
  at?: readonly [number, number],
  limit?: readonly [number, number],
): EditResult {
  if (piece.type === 'weights') {
    // A Weights tap must name a Linear (graph rule): it reads the first one; the wire is re-aimed by editing `of`.
    const lin = g.nodes.find((n) => n.type === 'linear');
    if (!lin) return fail('A Weights tap reads a Linear piece; add a Linear first.');
    piece = { ...piece, of: lin.id };
  }
  const id = freshId(g, piece.type);
  const node = { id, ...withFreshColumn(g, piece) } as Node;
  let positions = g.positions;
  if (at) {
    if (!Number.isFinite(at[0]) || !Number.isFinite(at[1])) {
      return fail('A piece can only be placed at a position given as numbers.');
    }
    const [maxX, maxY] = positionLimits(g.nodes.length + 1, limit);
    positions = {
      ...g.positions,
      [id]: [
        Math.min(maxX, Math.max(LAYOUT.pad, at[0])),
        Math.min(maxY, Math.max(LAYOUT.pad, at[1])),
      ],
    };
  }
  const parsed = parseGraph({
    ...g,
    nodes: [...g.nodes, node],
    ...(positions ? { positions } : {}),
  });
  if (!parsed.ok) return fail(parsed.error);
  return { ok: true, graph: parsed.graph, select: id };
}

/** Keys that name which piece it is within its type; they are never carried over by a swap. */
const IDENTITY_KEYS: Partial<Record<Node['type'], string>> = {
  activation: 'fn',
  link: 'fn',
  norm: 'kind',
};

/**
 * Replaces a piece by another of the same family. Settings with the same name and a value the new
 * piece accepts carry over (so a Linear keeps its width); the rest take the new piece's values.
 * Wires follow the new piece's ports by position. Pins stay where they are (the id is unchanged).
 */
export function swapPiece(
  g: Graph,
  nodeId: NodeId,
  piece: NodeBody,
  opts?: EditOptions,
): EditResult {
  const old = findNode(g, nodeId);
  if (!old) return noNode(nodeId);
  if (lockedAmong(opts, [nodeId])) return fail(LOCKED_MESSAGE);
  const family = CORE_PIECES[old.type].family;
  if (CORE_PIECES[piece.type].family !== family) {
    return fail(`A ${old.type} piece can only be swapped for another piece of the same family.`);
  }
  // A swap keeps the wiring, so the two pieces must take the same input ports (and Input and
  // Target, which both have none, are different data and never swap).
  const same =
    inputPortsOf(old.type).join() === inputPortsOf(piece.type).join() &&
    variadicPortsOf(old.type).join() === variadicPortsOf(piece.type).join() &&
    optionalPortsOf(old.type).join() === optionalPortsOf(piece.type).join();
  if (!same || (old.type !== piece.type && family === 'inputs')) {
    return fail(
      `A ${piece.type} cannot replace the ${old.type}; they are different kinds of piece.`,
    );
  }
  const replace = (n: Node): Node[] => g.nodes.map((x) => (x.id === nodeId ? n : x));

  // Wires into the old piece move to the new piece's ports by position; extras are dropped.
  const oldPorts = inputPortsOf(old.type);
  const newPorts = inputPortsOf(piece.type);
  const resolved = resolveAll(g);
  const edges: Edge[] = [];
  let dropped = false;
  g.edges.forEach((e, i) => {
    const r = resolved[i];
    if (!r || r.to.node !== nodeId) {
      edges.push(e);
      return;
    }
    const idx = oldPorts.indexOf(r.to.port);
    const port = newPorts[idx];
    if (port === undefined) {
      dropped = true;
      return;
    }
    edges.push([e[0], newPorts.length === 1 ? nodeId : `${nodeId}.${port}`]);
  });

  if (dropped) {
    return fail(
      `A ${piece.type} cannot replace the ${old.type}, because a wire into it would be lost.`,
    );
  }
  const base = { id: nodeId, ...withFreshColumn(g, piece) } as Node;
  const identity = IDENTITY_KEYS[piece.type];
  const carried: Record<string, unknown> = {};
  const rest = old as unknown as Record<string, unknown>;
  const known = new Set<string>(Object.keys(CORE_PIECES[piece.type].defaults()));
  if (old.type === piece.type) {
    for (const key of Object.keys(rest)) {
      if (key === 'id' || key === 'type' || key === identity || !known.has(key)) continue;
      const trial = { ...base, ...carried, [key]: rest[key] } as Node;
      if (parseGraph({ ...g, nodes: replace(trial), edges }).ok) carried[key] = rest[key];
    }
  }
  return finish(g, replace({ ...base, ...carried } as Node), edges, nodeId);
}

/**
 * Adds the wire `from -> to`, replacing whatever fed that input port, except on a variadic port
 * (CONTRACTS.md section 3), which takes many wires: there the wire is appended after the others and
 * nothing is replaced. The wire is stored in the short form (`x`, `loss.pred`). Refused for a bad port, a cycle (including a node into itself),
 * a wire that already exists, and a locked destination. Does not enforce shape or role rules;
 * `analyse` reports those.
 */
export function connect(g: Graph, from: PortRef, to: PortRef, opts?: EditOptions): EditResult {
  const r = resolveEdge(g.nodes, [from, to]);
  if (!r.ok) return fail(r.error);
  if (lockedAmong(opts, [r.to.node])) return fail(LOCKED_MESSAGE);
  const resolved = resolveAll(g);
  if (r.from.node === r.to.node || reaches(resolved, r.to.node, r.from.node)) {
    return fail('That wire would make a loop, and a network cannot feed into itself.');
  }
  const feeders = resolved.filter((x) => x && x.to.node === r.to.node && x.to.port === r.to.port);
  const only = feeders[0];
  if (feeders.length === 1 && only && only.from.node === r.from.node) {
    return fail('Those two are already connected.');
  }
  const toNode = findNode(g, r.to.node);
  const variadic = toNode !== undefined && isVariadicPort(toNode.type, r.to.port);
  if (variadic && feeders.some((x) => x?.from.node === r.from.node)) {
    return fail('Those two are already connected.');
  }
  const edges = variadic
    ? g.edges.slice()
    : g.edges.filter(
        (_, i) => !(resolved[i]?.to.node === r.to.node && resolved[i]?.to.port === r.to.port),
      );
  edges.push([r.from.node, toRef(g, r.to.node, r.to.port)]);
  return finish(g, g.nodes, edges);
}

/**
 * Moves one end of an existing wire: `end: 'to'` points it at another input port (`ref`),
 * `end: 'from'` makes another piece feed the same input. One edit, so one undo step. The old wire
 * is removed first, so whatever already fed the new input port is replaced, as in `connect`.
 * On a variadic input port (CONTRACTS.md section 3) nothing is replaced: `end: 'to'` onto one appends
 * the wire there, and moving a wire off one removes just that wire; `end: 'from'` on a wire into a
 * variadic port keeps the wire's place in the edge list (so the input order is unchanged).
 * Refused when the wire does not exist, for a bad port, a cycle, and when the wire's old
 * destination or the new destination is locked.
 */
export function rewireEdge(
  g: Graph,
  edgeIndex: number,
  end: 'from' | 'to',
  ref: PortRef,
  opts?: EditOptions,
): EditResult {
  const old = resolveAll(g)[edgeIndex];
  const e = g.edges[edgeIndex];
  if (!old || !e) return fail('That wire does not exist.');
  if (lockedAmong(opts, [old.to.node])) return fail(LOCKED_MESSAGE);
  if (end === 'to') {
    // Dropping a wire back on its own port changes nothing (and must not move it in the fold order).
    const dest = resolveEdge(g.nodes, [e[0], ref]);
    if (dest.ok && dest.to.node === old.to.node && dest.to.port === old.to.port) {
      return { ok: true, graph: g };
    }
  }
  const rest: Graph = { ...g, edges: g.edges.filter((_, i) => i !== edgeIndex) };
  if (end === 'to') return connect(rest, e[0], ref, opts);
  const res = connect(rest, ref, e[1], opts);
  const dest = findNode(g, old.to.node);
  if (res.ok && dest !== undefined && isVariadicPort(dest.type, old.to.port)) {
    // connect appended the new wire; put it back where the old one was.
    const edges = res.graph.edges.slice();
    const moved = edges.pop();
    if (moved !== undefined) edges.splice(edgeIndex, 0, moved);
    return finish(g, res.graph.nodes, edges);
  }
  return res;
}

/** Analysis for shape questions only: the dataset kind and class count do not change shapes. */
function shapesOf(g: Graph): Analysis {
  return analyse(g, {
    columns: [
      ...new Set(
        g.nodes.flatMap((n) =>
          n.type === 'input' ? n.columns : n.type === 'column' ? [n.name] : [],
        ),
      ),
    ],
    kind: 'regression',
    numClasses: 2,
  });
}

/** The node a pin points at. */
const pinNode = (ref: PortRef): NodeId => parsePort(ref).node;

/**
 * Removes a piece and its wires. When it had exactly one wire in, and at least one wire out or a
 * pin, and the shape that reaches it equals the shape it passes on (or either is not known yet),
 * each wire out is reconnected to the upstream piece, keeping its place in the edge list, and a ŷ or
 * L pin on the piece moves to that upstream piece. Otherwise the wires out are removed, the pieces
 * downstream are left with an open input, and its pins are dropped. A Linear takes the Weights
 * pieces that read it along. Refused when the piece or any piece it feeds is locked.
 */
export function deleteNode(g: Graph, nodeId: NodeId, opts?: EditOptions): EditResult {
  const node = findNode(g, nodeId);
  if (!node) return noNode(nodeId);
  // Weights taps of a deleted Linear go with it (a tap needs its Linear).
  const gone = new Set<NodeId>([
    nodeId,
    ...(node.type === 'linear'
      ? g.nodes.flatMap((n) => (n.type === 'weights' && n.of === nodeId ? [n.id] : []))
      : []),
  ]);
  const resolved = resolveAll(g);
  const outs = g.edges.flatMap((_, i) => (resolved[i]?.from.node === nodeId ? [i] : []));
  const ins = g.edges.flatMap((_, i) => (resolved[i]?.to.node === nodeId ? [i] : []));
  const downstream = g.edges.flatMap((_, i) => {
    const r = resolved[i];
    return r && gone.has(r.from.node) ? [r.to.node] : [];
  });
  if (lockedAmong(opts, [...gone, ...downstream])) return fail(LOCKED_MESSAGE);

  const nodes = g.nodes.filter((n) => !gone.has(n.id));
  const inIdx = ins[0];
  const inEdge = inIdx === undefined ? undefined : g.edges[inIdx];
  const inRes = inIdx === undefined ? undefined : resolved[inIdx];
  const pinned =
    g.predictions.some((p) => pinNode(p) === nodeId) ||
    (g.loss !== null && pinNode(g.loss) === nodeId);
  let close =
    gone.size === 1 &&
    inputPortsOf(node.type).length === 1 &&
    ins.length === 1 &&
    (outs.length > 0 || pinned) &&
    inEdge !== undefined &&
    inRes != null;
  if (close && inRes && inRes.from.node === nodeId) close = false;
  if (close && inIdx !== undefined) {
    const a = shapesOf(g);
    const up = a.edgeShape[inIdx];
    const down = a.shape[nodeId];
    if (up && down && !sameShape(up, down)) close = false;
  }
  // A closed gap must not duplicate a wire the upstream piece already has into a variadic port.
  const keyOf = (from: NodeId, r: Resolved) => `${from}>${r?.to.node}.${r?.to.port}`;
  const kept = new Set<string>();
  g.edges.forEach((_, i) => {
    const r = resolved[i];
    if (
      r &&
      !ins.includes(i) &&
      !outs.includes(i) &&
      !gone.has(r.from.node) &&
      !gone.has(r.to.node)
    )
      kept.add(keyOf(r.from.node, r));
  });
  const edges: Edge[] = [];
  g.edges.forEach((e, i) => {
    const r = resolved[i];
    if (ins.includes(i)) return;
    if (outs.includes(i)) {
      if (close && inEdge !== undefined && inRes != null && r) {
        const dest = findNode(g, r.to.node);
        const key = keyOf(inRes.from.node, r);
        if (dest && isVariadicPort(dest.type, r.to.port) && kept.has(key)) return;
        kept.add(key);
        edges.push([inEdge[0], e[1]]);
      }
      return;
    }
    if (r && (gone.has(r.from.node) || gone.has(r.to.node))) return;
    edges.push(e);
  });
  const up = close && inEdge !== undefined ? pinNode(inEdge[0]) : null;
  const move = (ref: PortRef): PortRef | null =>
    pinNode(ref) === nodeId ? up : gone.has(pinNode(ref)) ? null : ref;
  const predictions: PortRef[] = [];
  for (const p of g.predictions) {
    const m = move(p);
    // The same node cannot be a ŷ twice; keep the first.
    if (m !== null && !predictions.some((q) => pinNode(q) === pinNode(m))) predictions.push(m);
  }
  const loss = g.loss === null ? null : move(g.loss);
  const parsed = parseGraph({ ...g, nodes, edges, predictions, loss });
  if (!parsed.ok) return fail(parsed.error);
  return { ok: true, graph: parsed.graph };
}

/** Removes one wire. Refused when it ends at a locked node. */
export function deleteEdge(g: Graph, edgeIndex: number, opts?: EditOptions): EditResult {
  const r = resolveAll(g)[edgeIndex];
  if (g.edges[edgeIndex] === undefined) return fail('That wire does not exist.');
  if (r && lockedAmong(opts, [r.to.node])) return fail(LOCKED_MESSAGE);
  return finish(
    g,
    g.nodes,
    g.edges.filter((_, i) => i !== edgeIndex),
  );
}

/**
 * Sets one setting of a piece, checked against the node schema (an unknown key, a wrong type or an
 * out-of-range value is refused with the schema's sentence). `undefined` removes an optional
 * setting. `id` and `type` cannot be set.
 */
export function setSetting(
  g: Graph,
  nodeId: NodeId,
  key: string,
  value: unknown,
  opts?: EditOptions,
): EditResult {
  const node = findNode(g, nodeId);
  if (!node) return noNode(nodeId);
  if (lockedAmong(opts, [nodeId])) return fail(LOCKED_MESSAGE);
  if (key === 'id' || key === 'type') return fail(`The ${key} of a piece cannot be changed.`);
  const settings = Object.keys(CORE_PIECES[node.type].defaults());
  if (!settings.includes(key)) return fail(`A ${node.type} piece has no setting called "${key}".`);
  const kept = Object.entries(node).filter(([k]) => k !== key);
  const next: Record<string, unknown> = Object.fromEntries(kept);
  if (value !== undefined) next[key] = value;
  return finish(
    g,
    g.nodes.map((n) => (n.id === nodeId ? (next as Node) : n)),
    g.edges,
    undefined,
  );
}

/** Neuron-view plus and minus: sets a Linear's output width, a whole number from 1 to 64. */
export function setLinearOut(
  g: Graph,
  nodeId: NodeId,
  out: number,
  opts?: EditOptions,
): EditResult {
  const node = findNode(g, nodeId);
  if (!node) return noNode(nodeId);
  if (node.type !== 'linear') return fail('Only a Linear piece has a width to set.');
  if (lockedAmong(opts, [nodeId])) return fail(LOCKED_MESSAGE);
  if (!Number.isInteger(out) || out < 1 || out > 64) {
    return fail('A Linear piece can have from 1 to 64 units.');
  }
  return finish(
    g,
    g.nodes.map((n) => (n.id === nodeId ? { ...node, out } : n)),
    g.edges,
  );
}

/** Makes unused ids: the suffix itself when free, else `suffix_2`, `suffix_3`, ... */
function namer(used: Set<string>): Namer {
  return (suffix) => {
    let id = suffix;
    for (let k = 2; used.has(id); k++) id = `${suffix}_${k}`;
    used.add(id);
    return id;
  };
}

/**
 * Adds a ready-made group (a palette macro, CONTRACTS.md section 3) as ordinary nodes and edges, in
 * one edit. `at` is the wire it was dropped on; without it the group goes in free-standing and is
 * wired as far as the graph allows:
 * - a loss group is fed its prediction from the wire's source (else the first ŷ pin), gets the
 *   graph's Target (a new one when there is none), and takes the L pin when none is set;
 * - the weight penalty reads every Linear, is fed the current loss from the wire's source (else the
 *   L pin), and takes the L pin over when that is where it was fed from.
 * The wire it was dropped on stays. Selects the group's result node. Nothing existing changes, so
 * locks do not apply.
 */
export function insertMacro(
  g: Graph,
  key: string,
  at?: { edge: number },
  pos?: readonly [number, number],
  limit?: readonly [number, number],
): EditResult {
  const entry = PALETTE.find((p) => p.kind === 'macro' && p.key === key);
  if (entry?.kind !== 'macro') return fail(`There is no ready-made group called "${key}".`);
  const wire = at === undefined ? undefined : resolveAll(g)[at.edge];
  if (at !== undefined && !wire) return fail('That wire does not exist.');
  const used = new Set(g.nodes.map((n) => n.id));
  const linears = g.nodes.flatMap((n) => (n.type === 'linear' ? [n.id] : []));
  const sub = entry.make(namer(used), { linears });
  if (sub.out === null || sub.nodes.length === 0) {
    return fail('There is no Linear piece for a weight penalty to read.');
  }
  const nodes: Node[] = [...g.nodes, ...sub.nodes];
  const edges: Edge[] = [...g.edges];
  let loss = g.loss;
  const wireTo = (from: PortRef | undefined, to: PortRef | undefined) => {
    if (from !== undefined && to !== undefined) edges.push([from, to]);
  };
  if (key === 'weight_penalty') {
    const src = wire ? wire.from.node : g.loss === null ? undefined : pinNode(g.loss);
    wireTo(src, sub.inputs['total']);
    edges.push(...sub.edges);
    if (src !== undefined && g.loss !== null && pinNode(g.loss) === src) loss = sub.out;
  } else {
    wireTo(wire ? wire.from.node : g.predictions[0], sub.inputs['pred']);
    let target = g.nodes.find((n) => n.type === 'target')?.id;
    if (target === undefined) {
      target = namer(used)('target');
      nodes.push({ id: target, type: 'target' });
    }
    wireTo(target, sub.inputs['target']);
    edges.push(...sub.edges);
    loss ??= sub.out;
  }
  let positions = g.positions;
  if (pos && at === undefined) {
    if (!Number.isFinite(pos[0]) || !Number.isFinite(pos[1])) {
      return fail('A piece can only be placed at a position given as numbers.');
    }
    positions = {
      ...g.positions,
      ...groupPositions(sub.nodes, sub.edges, pos, nodes.length, limit),
    };
  }
  const parsed = parseGraph({ ...g, nodes, edges, loss, ...(positions ? { positions } : {}) });
  if (!parsed.ok) return fail(parsed.error);
  return { ok: true, graph: parsed.graph, select: sub.out };
}

/** Positions for a free-standing group: columns by depth along its own wires, `at` the top-left, clamped like `moveNodes`. */
function groupPositions(
  sub: readonly Node[],
  edges: readonly Edge[],
  at: readonly [number, number],
  nodeCount: number,
  limit?: readonly [number, number],
): Record<NodeId, [number, number]> {
  const depth: Record<NodeId, number> = Object.create(null) as Record<NodeId, number>;
  for (const n of sub) depth[n.id] = 0;
  for (let pass = 0; pass < sub.length; pass++) {
    for (const [a, b] of edges) {
      const from = parsePort(a).node;
      const to = parsePort(b).node;
      if (depth[from] !== undefined && depth[to] !== undefined) {
        depth[to] = Math.max(depth[to] ?? 0, (depth[from] ?? 0) + 1);
      }
    }
  }
  const rows: Record<number, number> = {};
  const [maxX, maxY] = positionLimits(nodeCount, limit);
  const out: Record<NodeId, [number, number]> = {};
  for (const n of sub) {
    const d = depth[n.id] ?? 0;
    const r = rows[d] ?? 0;
    rows[d] = r + 1;
    out[n.id] = [
      Math.min(maxX, Math.max(LAYOUT.pad, at[0] + d * (LAYOUT.boxW + LAYOUT.layerGap))),
      Math.min(maxY, Math.max(LAYOUT.pad, at[1] + r * (LAYOUT.boxH + LAYOUT.rowGap))),
    ];
  }
  return out;
}

// ------------------------------------------------------------------ pin edits

/** Checks a pin target: it must be an existing node (a bare id or `id.out`). Returns the node id. */
function pinTargetOf(g: Graph, ref: PortRef): { id: NodeId } | { error: string } {
  const r = resolvePin(g.nodes, ref);
  return r.ok ? { id: r.at.node } : { error: r.error };
}

function withPins(g: Graph, predictions: PortRef[], loss: PortRef | null): EditResult {
  const parsed = parseGraph({ ...g, predictions, loss });
  return parsed.ok ? { ok: true, graph: parsed.graph } : fail(parsed.error);
}

/**
 * Moves the L pin to the output of `ref` (a node id), or removes it with `null`. There is exactly
 * one L pin, so dropping it elsewhere moves it. Refused when the old or the new node is locked.
 */
export function setLoss(g: Graph, ref: PortRef | null, opts?: EditOptions): EditResult {
  let next: NodeId | null = null;
  if (ref !== null) {
    const t = pinTargetOf(g, ref);
    if ('error' in t) return fail(t.error);
    next = t.id;
  }
  const old = g.loss === null ? [] : [pinNode(g.loss)];
  if (lockedAmong(opts, [...old, ...(next === null ? [] : [next])])) return fail(LOCKED_MESSAGE);
  if (next !== null && g.loss !== null && pinNode(g.loss) === next) {
    return fail('That is already the loss.');
  }
  return withPins(g, g.predictions, next);
}

/** Adds a ŷ pin on the output of `ref`. The same node cannot be a ŷ twice. Refused on a locked node. */
export function addPrediction(g: Graph, ref: PortRef, opts?: EditOptions): EditResult {
  const t = pinTargetOf(g, ref);
  if ('error' in t) return fail(t.error);
  if (lockedAmong(opts, [t.id])) return fail(LOCKED_MESSAGE);
  if (g.predictions.some((p) => pinNode(p) === t.id)) {
    return fail('That piece is already a prediction.');
  }
  return withPins(g, [...g.predictions, t.id], g.loss);
}

/** Moves the ŷ pin at `index` (into `graph.predictions`) to another node, keeping its place in the order. */
export function movePrediction(
  g: Graph,
  index: number,
  ref: PortRef,
  opts?: EditOptions,
): EditResult {
  const old = g.predictions[index];
  if (old === undefined) return fail('That prediction does not exist.');
  const t = pinTargetOf(g, ref);
  if ('error' in t) return fail(t.error);
  if (lockedAmong(opts, [pinNode(old), t.id])) return fail(LOCKED_MESSAGE);
  if (pinNode(old) === t.id) return fail('That is already the prediction.');
  if (g.predictions.some((p) => pinNode(p) === t.id)) {
    return fail('That piece is already a prediction.');
  }
  return withPins(
    g,
    g.predictions.map((p, i) => (i === index ? t.id : p)),
    g.loss,
  );
}

/** Removes the ŷ pin at `index`. Refused when its node is locked. */
export function removePrediction(g: Graph, index: number, opts?: EditOptions): EditResult {
  const old = g.predictions[index];
  if (old === undefined) return fail('That prediction does not exist.');
  if (lockedAmong(opts, [pinNode(old)])) return fail(LOCKED_MESSAGE);
  return withPins(
    g,
    g.predictions.filter((_, i) => i !== index),
    g.loss,
  );
}

/** Sets which data columns an Input piece uses, in the order given. Refused for repeats. */
export function setColumns(
  g: Graph,
  inputNodeId: NodeId,
  columns: readonly string[],
  opts?: EditOptions,
): EditResult {
  const node = findNode(g, inputNodeId);
  if (!node) return noNode(inputNodeId);
  if (node.type !== 'input') return fail('Only an Input piece has data columns.');
  if (lockedAmong(opts, [inputNodeId])) return fail(LOCKED_MESSAGE);
  if (new Set(columns).size !== columns.length) {
    return fail('Each data column can be used only once.');
  }
  return finish(
    g,
    g.nodes.map((n) => (n.id === inputNodeId ? { ...node, columns: [...columns] } : n)),
    g.edges,
  );
}

/**
 * Furthest a piece may be placed: two spare columns and rows past a drawing with every piece in its
 * own, or the optional far-corner `limit` (the visible canvas's bottom-right, in layout units) when
 * that is further on an axis, so any spot the user can see is a valid place.
 */
export function positionLimits(
  nodeCount: number,
  limit?: readonly [number, number],
): [number, number] {
  const spare = nodeCount + 2;
  const x = LAYOUT.pad + spare * (LAYOUT.boxW + LAYOUT.layerGap);
  const y = LAYOUT.pad + spare * (LAYOUT.boxH + LAYOUT.rowGap);
  if (!limit || !Number.isFinite(limit[0]) || !Number.isFinite(limit[1])) return [x, y];
  return [Math.max(x, limit[0]), Math.max(y, limit[1])];
}

/**
 * Puts pieces where the user dragged them (CONTRACTS.md section 2, `positions`): `moves` maps a
 * node id to the new top-left corner of its box, in layout units, clamped to `LAYOUT.pad` or more. One edit,
 * so one undo step, however many pieces. Only `positions` changes (nodes and edges keep their
 * identity), and a locked piece may be moved: a lock protects what a piece computes, not where it
 * sits. Refuses an unknown piece or a value that is not a finite number.
 */
export function moveNodes(
  g: Graph,
  moves: Readonly<Record<NodeId, readonly [number, number]>>,
  limit?: readonly [number, number],
): EditResult {
  // No further than two spare columns and rows past a drawing with every piece in its own column
  // and row, so a far drag cannot make the drawing (and the fit) grow without limit.
  const [maxX, maxY] = positionLimits(g.nodes.length, limit);
  const positions: Record<NodeId, [number, number]> = { ...g.positions };
  for (const [id, xy] of Object.entries(moves)) {
    if (!findNode(g, id)) return noNode(id);
    if (!Number.isFinite(xy[0]) || !Number.isFinite(xy[1])) {
      return fail('A piece can only be moved to a position given as numbers.');
    }
    positions[id] = [
      Math.min(maxX, Math.max(LAYOUT.pad, xy[0])),
      Math.min(maxY, Math.max(LAYOUT.pad, xy[1])),
    ];
  }
  return { ok: true, graph: { ...g, positions } };
}

/**
 * Gives pieces back to the automatic layout: removes the saved positions of `ids`, or of every
 * piece when `ids` is left out. Refused when none of them had one.
 */
export function clearPositions(g: Graph, ids?: readonly NodeId[]): EditResult {
  const drop = new Set(ids ?? Object.keys(g.positions ?? {}));
  const kept = Object.entries(g.positions ?? {}).filter(([id]) => !drop.has(id));
  if (kept.length === Object.keys(g.positions ?? {}).length) {
    return fail('No piece has been moved by hand.');
  }
  const { positions: _old, ...rest } = g;
  return {
    ok: true,
    graph: kept.length > 0 ? { ...rest, positions: Object.fromEntries(kept) } : rest,
  };
}
