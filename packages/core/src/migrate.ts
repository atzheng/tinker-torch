/**
 * Version 1 -> 2 migration (CONTRACTS.md section 2). The result computes exactly what the v1 graph did.
 *
 * - Output(sigmoid | softmax) becomes a Link piece with the same id. Output(identity) is spliced out:
 *   what fed it feeds its consumers, and the ŷ pin sits on that source.
 * - The ŷ pin is the Output feeding the (first) loss's `pred`; with no such Output, every Output.
 * - Each Loss becomes the primitive group of `lossSubgraph`; its final node (the Mean, or the Add of
 *   the penalty) keeps the loss's id and carries the L pin. Only the first Loss is pinned.
 * - A penalty becomes Weights -> norm for every Linear that feeds the loss (summed), then Scale(lambda),
 *   then an Add onto the Mean. With no Linear feeding the loss it contributed nothing and is dropped.
 * - Node ids of everything else are unchanged, so saved parameters keep their keys.
 */
import type { Graph } from './graph.js';
import type { Node } from './graph.js';
import type { Edge, NodeId, PortRef } from './common.js';
import { parsePort } from './common.js';
import { resolveEdge, type Graph as GraphV1, type Node as NodeV1 } from './v1.js';
import { lossSubgraph, weightPenaltySubgraph, type Namer, type Subgraph } from './macros.js';

type OutputNode = Extract<NodeV1, { type: 'output' }>;
type LossNode = Extract<NodeV1, { type: 'loss' }>;

/** Where each version 1 node id went: spliced identity Outputs map to none, a Loss to its whole group. */
export type IdMap = Record<NodeId, NodeId[]>;

export function migrateV1(g: GraphV1): Graph {
  return migrateV1WithMap(g).graph;
}

export function migrateV1WithMap(g: GraphV1): { graph: Graph; idMap: IdMap } {
  const used = new Set<string>(g.nodes.map((n) => n.id));
  const fresh = (base: string): string => {
    let id = base;
    for (let k = 2; used.has(id); k++) id = `${base}_${k}`;
    used.add(id);
    return id;
  };

  // Resolved edges (parseGraph already checked them).
  const edges = g.edges.map((e) => {
    const r = resolveEdge(g.nodes, e);
    if (!r.ok) throw new Error(r.error);
    return { raw: e, from: r.from.node, to: r.to.node, toPort: r.to.port };
  });
  const byId = new Map<NodeId, NodeV1>(g.nodes.map((n) => [n.id, n]));
  const isIdentityOut = (id: NodeId): boolean => {
    const n = byId.get(id);
    return n?.type === 'output' && n.link === 'identity';
  };

  /** The source port reference feeding an identity Output, looking through chains of them. */
  const sourceOf = (id: NodeId): PortRef | null => {
    let cur = id;
    for (let hops = 0; hops <= g.nodes.length; hops++) {
      const e = edges.find((x) => x.to === cur);
      if (!e) return null;
      if (!isIdentityOut(e.from)) return e.raw[0];
      cur = e.from;
    }
    return null; // a cycle of identity outputs
  };

  // Loss groups.
  const losses = g.nodes.filter((n): n is LossNode => n.type === 'loss');
  const groups = new Map<NodeId, Subgraph>();
  for (const l of losses) {
    const name: Namer = (suffix) => fresh(`${l.id}_${suffix}`);
    // Ancestors of the loss over the v1 edges: the Linear pieces whose weights v1 penalised.
    const anc = new Set<NodeId>();
    const todo: NodeId[] = [l.id];
    while (todo.length > 0) {
      const cur = todo.pop() as NodeId;
      for (const e of edges) {
        if (e.to === cur && !anc.has(e.from)) {
          anc.add(e.from);
          todo.push(e.from);
        }
      }
    }
    const linears = g.nodes.filter((n) => n.type === 'linear' && anc.has(n.id)).map((n) => n.id);
    const penalty =
      l.penalty && linears.length > 0
        ? weightPenaltySubgraph({
            kind: l.penalty.kind,
            lambda: l.penalty.lambda,
            linears,
            name,
            finalId: l.id,
          })
        : null;
    const sub = lossSubgraph(l.fn, {
      name,
      ...(l.delta !== undefined ? { delta: l.delta } : {}),
      meanId: penalty ? name('mean') : l.id,
    });
    if (penalty && sub.out !== null && penalty.inputs['total'] !== undefined) {
      sub.nodes.push(...penalty.nodes);
      sub.edges.push(...penalty.edges, [sub.out, penalty.inputs['total']]);
      sub.out = penalty.out;
    }
    groups.set(l.id, sub);
  }

  // Nodes, in the original order.
  const nodes: Node[] = [];
  for (const n of g.nodes) {
    if (n.type === 'output') {
      if (n.link !== 'identity') nodes.push({ id: n.id, type: 'link', fn: n.link });
    } else if (n.type === 'loss') {
      nodes.push(...(groups.get(n.id) as Subgraph).nodes);
    } else {
      nodes.push(n);
    }
  }

  // Edges.
  const out: Edge[] = [];
  for (const e of edges) {
    if (isIdentityOut(e.to)) continue; // spliced out: its input edge is gone
    let from: PortRef = e.raw[0];
    if (isIdentityOut(e.from)) {
      const src = sourceOf(e.from);
      if (src === null) continue;
      from = src;
    }
    let to: PortRef = e.raw[1];
    const toNode = byId.get(e.to);
    if (toNode?.type === 'loss') {
      const dest = (groups.get(toNode.id) as Subgraph).inputs[
        e.toPort === 'pred' ? 'pred' : 'target'
      ];
      to = dest as PortRef;
    }
    out.push([from, to]);
  }
  for (const l of losses) out.push(...(groups.get(l.id) as Subgraph).edges);

  // Pins.
  const first = losses[0];
  const outputs = g.nodes.filter((n): n is OutputNode => n.type === 'output');
  let pinned: OutputNode[] = outputs;
  if (first) {
    const feed = edges.find((e) => e.to === first.id && e.toPort === 'pred');
    const o = feed ? byId.get(feed.from) : undefined;
    if (o?.type === 'output') pinned = [o];
  }
  const predictions: PortRef[] = [];
  for (const o of pinned) {
    const ref = o.link === 'identity' ? sourceOf(o.id) : o.id;
    if (ref === null) continue;
    const node = parsePort(ref).node;
    if (!predictions.some((p) => parsePort(p).node === node)) predictions.push(ref);
  }

  const idMap: IdMap = Object.create(null) as IdMap;
  for (const n of g.nodes) {
    if (n.type === 'loss') idMap[n.id] = (groups.get(n.id) as Subgraph).nodes.map((m) => m.id);
    else idMap[n.id] = n.type === 'output' && n.link === 'identity' ? [] : [n.id];
  }
  return {
    graph: {
      version: 2,
      nodes,
      edges: out,
      predictions,
      loss: first ? first.id : null,
      data: g.data,
      train: g.train,
    },
    idMap,
  };
}
