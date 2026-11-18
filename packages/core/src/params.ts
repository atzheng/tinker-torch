/** Host-side parameters (CONTRACTS.md section 5): plain Float32Arrays, no jax-js. */
import type { Analysis } from './analyse.js';
import type { Graph, Node, NodeId } from './graph.js';
import { CORE_PIECES } from './registry.js';
import { makeRng } from './rng.js';

/** W row-major [in, out]; b [out]. Only linear nodes on `analysis.order` have an entry. */
export type HostParams = Record<NodeId, { W: Float32Array; b?: Float32Array }>;
export type JsonParams = Record<NodeId, { W: number[]; b?: number[] }>;

type Shapes = { W: [number, number]; b?: [number] };

/** Param shapes of the linear nodes on the loss path whose input width is known. */
function linearShapes(graph: Graph, analysis: Analysis): Map<NodeId, Shapes> {
  const byId = new Map<NodeId, Node>(graph.nodes.map((n) => [n.id, n]));
  const out = new Map<NodeId, Shapes>();
  for (const id of analysis.order) {
    const node = byId.get(id);
    const w = analysis.inWidth[id];
    if (!node || node.type !== 'linear' || w === undefined) continue;
    const s = CORE_PIECES.linear.paramShapes(node, w);
    if (s.W) out.set(id, { W: s.W, ...(s.b ? { b: s.b } : {}) });
  }
  return out;
}

function initNode(shapes: Shapes, seed: number, id: NodeId): { W: Float32Array; b?: Float32Array } {
  // One stream per node: adding a node does not change the others.
  const rng = makeRng(seed).fork(id);
  const bound = 1 / Math.sqrt(shapes.W[0]);
  const draw = (n: number): Float32Array => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (rng.next() * 2 - 1) * bound;
    return a;
  };
  const W = draw(shapes.W[0] * shapes.W[1]);
  return shapes.b ? { W, b: draw(shapes.b[0]) } : { W };
}

function matches(
  p: { W: ArrayLike<number>; b?: ArrayLike<number> } | undefined,
  s: Shapes,
): boolean {
  if (!p) return false;
  if (p.W.length !== s.W[0] * s.W[1]) return false;
  if (s.b) return p.b !== undefined && p.b.length === s.b[0];
  return p.b === undefined;
}

/** nn.Linear default: W and b uniform on +/- 1/sqrt(in). */
export function initParams(graph: Graph, analysis: Analysis, seed: number): HostParams {
  const host: HostParams = Object.create(null) as HostParams;
  for (const [id, s] of linearShapes(graph, analysis)) host[id] = initNode(s, seed, id);
  return host;
}

/**
 * Keep each node's arrays (same objects, bit-identical) when their shapes are unchanged between the old and
 * new graph. W is kept when [in, out] is unchanged, even if bias was toggled; b is kept when W and b exist in
 * both, drawn fresh when bias is turned on, and dropped when it is turned off. Everything else is re-inited.
 * Nodes no longer on the loss path are dropped. Does not mutate `old`. The returned arrays are shared with
 * `old` and must never be written in place.
 */
export function reconcileParams(
  old: HostParams,
  oldGraph: Graph,
  oldAnalysis: Analysis,
  graph: Graph,
  analysis: Analysis,
  seed: number,
): HostParams {
  const oldShapes = linearShapes(oldGraph, oldAnalysis);
  const host: HostParams = Object.create(null) as HostParams;
  for (const [id, s] of linearShapes(graph, analysis)) {
    const prev = Object.hasOwn(old, id) ? old[id] : undefined;
    const os = oldShapes.get(id);
    const sameW =
      prev !== undefined &&
      os !== undefined &&
      os.W[0] === s.W[0] &&
      os.W[1] === s.W[1] &&
      prev.W.length === s.W[0] * s.W[1];
    if (!prev || !sameW) {
      host[id] = initNode(s, seed, id);
    } else if (!s.b) {
      host[id] = { W: prev.W };
    } else if (prev.b && prev.b.length === s.b[0]) {
      host[id] = { W: prev.W, b: prev.b };
    } else {
      const fresh = initNode(s, seed, id).b;
      host[id] = fresh ? { W: prev.W, b: fresh } : { W: prev.W };
    }
  }
  return host;
}

export function countParams(host: HostParams): number {
  let n = 0;
  for (const id of Object.keys(host)) {
    const p = host[id];
    if (p) n += p.W.length + (p.b?.length ?? 0);
  }
  return n;
}

export function paramsToJson(host: HostParams): JsonParams {
  const out: JsonParams = Object.create(null) as JsonParams;
  for (const id of Object.keys(host)) {
    const p = host[id];
    if (!p) continue;
    out[id] = p.b ? { W: Array.from(p.W), b: Array.from(p.b) } : { W: Array.from(p.W) };
  }
  return out;
}

/**
 * Load saved params for `graph`. Entries whose shapes do not match (or that are malformed, or for nodes
 * that are not linear on the loss path) are left out and their ids listed in `dropped`.
 * Linear nodes with no saved entry are simply absent (the caller reconciles to fill them).
 */
export function paramsFromJson(
  json: unknown,
  graph: Graph,
  analysis: Analysis,
): { params: HostParams; dropped: NodeId[] } {
  const params: HostParams = Object.create(null) as HostParams;
  const dropped: NodeId[] = [];
  if (typeof json !== 'object' || json === null || Array.isArray(json)) return { params, dropped };
  const shapes = linearShapes(graph, analysis);
  const isNums = (v: unknown): v is number[] => {
    if (!Array.isArray(v)) return false;
    for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) return false; // also catches holes
    return true;
  };
  for (const id of Object.keys(json)) {
    const raw: unknown = (json as Record<string, unknown>)[id];
    const s = shapes.get(id);
    const r = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
    const W = r['W'];
    const b = r['b'];
    if (!s || !isNums(W) || (b !== undefined && !isNums(b))) {
      dropped.push(id);
      continue;
    }
    const cand = { W, ...(b !== undefined ? { b } : {}) };
    if (!matches(cand, s)) {
      dropped.push(id);
      continue;
    }
    params[id] =
      b !== undefined
        ? { W: Float32Array.from(W), b: Float32Array.from(b) }
        : { W: Float32Array.from(W) };
  }
  return { params, dropped };
}
