/**
 * Pure graph edits behind every canvas gesture (CONTRACTS.md section 8).
 *
 * Every edit returns a new graph and never touches its input. Every ok result has been through
 * `parseGraph`, so it always re-parses (defaults filled, huber delta set). Edits may leave a graph
 * that `analyse` reports as not trainable; they refuse only what cannot be represented (cycles,
 * unknown nodes or ports, a swap across families, a locked node, a value the schema rejects).
 */
import { analyse } from './analyse.js';
import {
  parseGraph,
  resolveEdge,
  inputPortsOf,
  type Edge,
  type Endpoint,
  type Graph,
  type Node,
  type NodeBody,
  type NodeId,
  type Penalty,
  type PortRef,
} from './graph.js';
import { CORE_PIECES } from './registry.js';

export type EditResult = { ok: true; graph: Graph; select?: NodeId } | { ok: false; error: string };

/** `locked` is the preset field (CONTRACTS.md section 12). A locked node is never changed. */
export type EditOptions = { locked?: readonly NodeId[] };

const LOCKED_MESSAGE = 'This piece is locked.';

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

/** Smallest free id of the form `${type}${k}`, k from 1. */
function freshId(g: Graph, type: string): NodeId {
  const used = new Set(g.nodes.map((n) => n.id));
  for (let k = 1; ; k++) if (!used.has(`${type}${k}`)) return `${type}${k}`;
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
 * Refused for pieces with no input or no output (Input, Target, losses), and when the wire ends at
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
  if (first === undefined || piece.type === 'loss') {
    return fail('That piece has no input and output, so it cannot go on a wire.');
  }
  if (lockedAmong(opts, [r.to.node])) return fail(LOCKED_MESSAGE);
  const id = freshId(g, piece.type);
  const node = { id, ...piece } as Node;
  const edges = g.edges.slice();
  edges.splice(edgeIndex, 1, [edge[0], ports.length === 1 ? id : `${id}.${first}`], [id, edge[1]]);
  return finish(g, [...g.nodes, node], edges, id);
}

/** Keys that name which piece it is within its type; they are never carried over by a swap. */
const IDENTITY_KEYS: Partial<Record<Node['type'], string>> = { activation: 'fn', loss: 'fn' };

/**
 * Replaces a piece by another of the same family. Settings with the same name and a value the new
 * piece accepts carry over (so a Linear keeps its width, a loss keeps its penalty); the rest take
 * the new piece's values. A Huber loss gets delta 1 unless the piece says otherwise, and delta is
 * dropped when swapping to another loss. Wires follow the new piece's ports by position.
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
  if (old.type !== piece.type && family === 'outputs') {
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
  const base = { id: nodeId, ...piece } as Node;
  const identity = IDENTITY_KEYS[piece.type];
  const carried: Record<string, unknown> = {};
  const rest = old as unknown as Record<string, unknown>;
  const known = new Set<string>([
    ...Object.keys(CORE_PIECES[piece.type].defaults()),
    ...(piece.type === 'loss' ? ['penalty', 'delta'] : []),
  ]);
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
 * Adds the wire `from -> to`, replacing whatever fed that input port. The wire is stored in the
 * short form (`x`, `loss.pred`). Refused for a bad port, a cycle (including a node into itself),
 * a wire that already exists, and a locked destination. Does not enforce the Output/Target wiring
 * rules; `analyse` reports those.
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
  const edges = g.edges.filter(
    (_, i) => !(resolved[i]?.to.node === r.to.node && resolved[i]?.to.port === r.to.port),
  );
  edges.push([r.from.node, toRef(g, r.to.node, r.to.port)]);
  return finish(g, g.nodes, edges);
}

/**
 * Removes a piece and its wires. When it had exactly one wire in, at least one wire out, and the
 * width that reaches it equals the width it passes on (or either is not known yet), each wire out
 * is reconnected to the upstream piece, keeping its place in the edge list. Otherwise the wires
 * out are removed and the pieces downstream are left with an open input. Refused when the piece or
 * any piece it feeds is locked.
 */
export function deleteNode(g: Graph, nodeId: NodeId, opts?: EditOptions): EditResult {
  const node = findNode(g, nodeId);
  if (!node) return noNode(nodeId);
  const resolved = resolveAll(g);
  const outs = g.edges.flatMap((_, i) => (resolved[i]?.from.node === nodeId ? [i] : []));
  const ins = g.edges.flatMap((_, i) => (resolved[i]?.to.node === nodeId ? [i] : []));
  const downstream = outs.flatMap((i) => {
    const r = resolved[i];
    return r ? [r.to.node] : [];
  });
  if (lockedAmong(opts, [nodeId, ...downstream])) return fail(LOCKED_MESSAGE);

  const nodes = g.nodes.filter((n) => n.id !== nodeId);
  const inIdx = ins[0];
  const inEdge = inIdx === undefined ? undefined : g.edges[inIdx];
  const inRes = inIdx === undefined ? undefined : resolved[inIdx];
  let close =
    inputPortsOf(node.type).length === 1 &&
    ins.length === 1 &&
    outs.length > 0 &&
    inEdge !== undefined &&
    inRes != null;
  if (close && inRes && inRes.from.node === nodeId) close = false;
  if (close && inIdx !== undefined) {
    const a = analyse(g, {
      columns: [...new Set(g.nodes.flatMap((n) => (n.type === 'input' ? n.columns : [])))],
      kind: 'regression',
      numClasses: 2,
    });
    const up = a.edgeWidth[inIdx];
    const down = a.outWidth[nodeId];
    if (typeof up === 'number' && typeof down === 'number' && up !== down) close = false;
  }
  const edges: Edge[] = [];
  g.edges.forEach((e, i) => {
    if (i === inIdx || ins.includes(i)) return;
    if (outs.includes(i)) {
      if (close && inEdge !== undefined) edges.push([inEdge[0], e[1]]);
      return;
    }
    edges.push(e);
  });
  return finish(g, nodes, edges);
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
 * setting. Changing a loss's `fn` away from huber drops its delta. `id` and `type` cannot be set.
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
  const settings = [
    ...Object.keys(CORE_PIECES[node.type].defaults()),
    ...(node.type === 'loss' ? ['delta', 'penalty'] : []),
  ];
  if (!settings.includes(key)) return fail(`A ${node.type} piece has no setting called "${key}".`);
  const dropDelta = node.type === 'loss' && key === 'fn' && value !== 'huber';
  const kept = Object.entries(node).filter(([k]) => k !== key && !(dropDelta && k === 'delta'));
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

/** Sets or (with null) removes the weight penalty of a loss node. */
export function setPenalty(
  g: Graph,
  lossNodeId: NodeId,
  penalty: Penalty | null,
  opts?: EditOptions,
): EditResult {
  const node = findNode(g, lossNodeId);
  if (!node) return noNode(lossNodeId);
  if (node.type !== 'loss') return fail('A weight penalty can only go on a loss.');
  if (lockedAmong(opts, [lossNodeId])) return fail(LOCKED_MESSAGE);
  const { penalty: _old, ...rest } = node;
  const next: Node = penalty === null ? rest : { ...rest, penalty: { ...penalty } };
  return finish(
    g,
    g.nodes.map((n) => (n.id === lossNodeId ? next : n)),
    g.edges,
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
