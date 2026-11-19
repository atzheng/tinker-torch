/** Graph format v2 (CONTRACTS.md section 2): types, zod schema, parse (with v1 migration) and serialise, port helpers. */
import { z } from 'zod';
import { CORE_PIECES } from './registry.js';
import { parseGraph as parseGraphV1 } from './legacy/graph.js';
import { migrateV1 } from './migrate.js';
import {
  ACTIVATION_FNS,
  DATASET_NAMES,
  OPTIMIZERS,
  bool,
  dataSchema,
  edgeSchema,
  int,
  isRecord,
  nodeId,
  num,
  oneOf,
  parsePort,
  trainSchema,
  type ActivationFn,
  type DataConfig,
  type DataInfo,
  type DatasetName,
  type Edge,
  type Endpoint,
  type NodeId,
  type PortRef,
  type TrainConfig,
} from './common.js';

export {
  ACTIVATION_FNS,
  DATASET_NAMES,
  OPTIMIZERS,
  parsePort,
  type ActivationFn,
  type DataConfig,
  type DataInfo,
  type DatasetName,
  type Edge,
  type Endpoint,
  type NodeId,
  type PortRef,
  type TrainConfig,
};

export const LINK_FNS = ['sigmoid', 'softmax'] as const;
export const NORM_KINDS = ['l2', 'l1'] as const;
export const NODE_TYPES = [
  'input',
  'target',
  'feature',
  'linear',
  'activation',
  'dropout',
  'add',
  'link',
  'subtract',
  'square',
  'abs',
  'huber',
  'log_loss',
  'cross_entropy',
  'mean',
  'scale',
  'weights',
  'norm',
] as const;

const nodeSchema = z.discriminatedUnion('type', [
  z.strictObject({
    id: nodeId,
    type: z.literal('input'),
    columns: z.array(z.string({ error: 'must be a list of column names' }), {
      error: 'must be a list of column names',
    }),
  }),
  z.strictObject({ id: nodeId, type: z.literal('target') }),
  z.strictObject({
    id: nodeId,
    type: z.literal('feature'),
    degree: int(1, 5).default(1),
    sin: bool.default(false),
    cos: bool.default(false),
    products: bool.default(false),
  }),
  z.strictObject({
    id: nodeId,
    type: z.literal('linear'),
    out: int(1, 64),
    bias: bool.default(true),
  }),
  z.strictObject({
    id: nodeId,
    type: z.literal('activation'),
    fn: z.enum(ACTIVATION_FNS, { error: oneOf(ACTIVATION_FNS) }),
  }),
  z.strictObject({
    id: nodeId,
    type: z.literal('dropout'),
    p: num('must be a number from 0 up to but not including 1', (n) => n >= 0 && n < 1),
  }),
  z.strictObject({ id: nodeId, type: z.literal('add') }),
  z.strictObject({
    id: nodeId,
    type: z.literal('link'),
    fn: z.enum(LINK_FNS, { error: oneOf(LINK_FNS) }),
  }),
  z.strictObject({ id: nodeId, type: z.literal('subtract') }),
  z.strictObject({ id: nodeId, type: z.literal('square') }),
  z.strictObject({ id: nodeId, type: z.literal('abs') }),
  z.strictObject({
    id: nodeId,
    type: z.literal('huber'),
    delta: num('must be a number above 0', (n) => n > 0).default(1),
  }),
  z.strictObject({ id: nodeId, type: z.literal('log_loss') }),
  z.strictObject({ id: nodeId, type: z.literal('cross_entropy') }),
  z.strictObject({ id: nodeId, type: z.literal('mean') }),
  z.strictObject({
    id: nodeId,
    type: z.literal('scale'),
    c: num('must be a number', () => true),
  }),
  z.strictObject({ id: nodeId, type: z.literal('weights'), of: nodeId }),
  z.strictObject({
    id: nodeId,
    type: z.literal('norm'),
    kind: z.enum(NORM_KINDS, { error: oneOf(NORM_KINDS) }),
  }),
]);

const GRAPH_VERSION = 2 as const;
const NODE_REF = 'must be a node name (text)';

const graphSchema = z.strictObject({
  version: z.literal(GRAPH_VERSION, { error: 'must be 2' }),
  nodes: z.array(nodeSchema, { error: 'must be a list of nodes' }),
  edges: z.array(edgeSchema, { error: 'must be a list of edges' }),
  predictions: z
    .array(z.string({ error: NODE_REF }), { error: 'must be a list of node names' })
    .default([]),
  loss: z.string({ error: NODE_REF }).nullable().default(null),
  data: dataSchema,
  train: trainSchema,
});

export type Node = z.infer<typeof nodeSchema>;
export type PieceType = Node['type'];
export type NodeOf<T extends PieceType> = Extract<Node, { type: T }>;
/** A node without its id (distributes over the union, unlike `Omit<Node, 'id'>`). */
export type NodeBody = Node extends infer N ? (N extends Node ? Omit<N, 'id'> : never) : never;
export type LinkFn = (typeof LINK_FNS)[number];
export type NormKind = (typeof NORM_KINDS)[number];
export type Graph = {
  version: 2;
  nodes: Node[];
  edges: Edge[];
  /** The ŷ pins: output ports whose values predict() returns, in order. May be empty. */
  predictions: PortRef[];
  /** The L pin: the output port training minimises, or null when none is chosen yet. */
  loss: PortRef | null;
  data: DataConfig;
  train: TrainConfig;
};

// ---------------------------------------------------------------- ports

export type ResolvedEdge =
  { ok: true; from: Endpoint; to: Endpoint } | { ok: false; error: string };
export type ResolvedPin = { ok: true; at: Endpoint } | { ok: false; error: string };

/** Input port names of a node type (CONTRACTS.md section 2). */
export function inputPortsOf(type: PieceType): string[] {
  return CORE_PIECES[type].inputPorts;
}

/** Output port names of a node type: every piece has exactly `out`. */
export function outputPortsOf(_type: PieceType): string[] {
  return ['out'];
}

/**
 * Resolves an edge's two port references against the nodes: a bare source means `out`, a bare
 * destination means the single input port. Gives one plain sentence when a node or port is wrong.
 */
export function resolveEdge(nodes: readonly Node[], edge: Edge): ResolvedEdge {
  const [fromRef, toRef] = edge;
  const text = `The edge from "${fromRef}" to "${toRef}"`;
  const src = parsePort(fromRef);
  const dst = parsePort(toRef);
  const from = nodes.find((n) => n.id === src.node);
  const to = nodes.find((n) => n.id === dst.node);
  if (!from) return { ok: false, error: `${text} starts at "${src.node}", which is not a node.` };
  if (!to) return { ok: false, error: `${text} ends at "${dst.node}", which is not a node.` };
  if (src.port !== null && src.port !== 'out') {
    return {
      ok: false,
      error: `${text} starts at "${fromRef}", but "${src.node}" only has the output "out".`,
    };
  }
  const ports = inputPortsOf(to.type);
  if (ports.length === 0) {
    return { ok: false, error: `${text} ends at "${dst.node}", which takes no inputs.` };
  }
  let toPort: string;
  if (dst.port === null) {
    const only = ports[0];
    if (ports.length !== 1 || only === undefined) {
      return {
        ok: false,
        error: `${text} does not say which input of "${dst.node}" to use (${ports
          .map((p) => `${dst.node}.${p}`)
          .join(' or ')}).`,
      };
    }
    toPort = only;
  } else {
    if (!ports.includes(dst.port)) {
      return {
        ok: false,
        error: `${text} ends at "${toRef}", but "${dst.node}" has the inputs ${ports.join(', ')}.`,
      };
    }
    toPort = dst.port;
  }
  return { ok: true, from: { node: from.id, port: 'out' }, to: { node: to.id, port: toPort } };
}

/** Resolves a role pin (a ŷ or the L): a node, or "node.out". One sentence when it is wrong. */
export function resolvePin(nodes: readonly Node[], ref: PortRef): ResolvedPin {
  const { node, port } = parsePort(ref);
  if (!nodes.some((n) => n.id === node)) {
    return { ok: false, error: `The pin "${ref}" points at "${node}", which is not a node.` };
  }
  if (port !== null && port !== 'out') {
    return { ok: false, error: `The pin "${ref}" must point at the output "out" of "${node}".` };
  }
  return { ok: true, at: { node, port: 'out' } };
}

// ---------------------------------------------------------------- parse

export type ParseResult = { ok: true; graph: Graph } | { ok: false; error: string };

function describeIssue(issue: z.core.$ZodIssue, json: unknown): string {
  const path = issue.path;
  const [head, idx] = path;
  let where: string;
  let rest: PropertyKey[];
  if (head === 'nodes' && typeof idx === 'number') {
    const raw = isRecord(json) && Array.isArray(json['nodes']) ? json['nodes'][idx] : undefined;
    const id = isRecord(raw) && typeof raw['id'] === 'string' ? raw['id'] : undefined;
    where = id !== undefined ? `Node "${id}"` : `Node number ${idx + 1}`;
    rest = path.slice(2);
  } else if (head === 'edges' && typeof idx === 'number') {
    where = `Edge number ${idx + 1}`;
    rest = path.slice(2);
  } else if (head === 'data') {
    where = 'The data settings';
    rest = path.slice(1);
  } else if (head === 'train') {
    where = 'The training settings';
    rest = path.slice(1);
  } else {
    where = 'The graph';
    rest = path;
  }
  const field = rest.map(String).join('.');
  if (issue.code === 'unrecognized_keys') {
    const keys = issue.keys.map((k) => `"${k}"`).join(', ');
    const prefix = field ? `${where} field "${field}"` : where;
    return `${prefix} has an unknown field ${keys}.`;
  }
  const message = issue.message.replace(/\.$/, '');
  if (!field) return `${where} ${message}.`;
  return `${where} field "${field}" ${message}.`;
}

/**
 * Parses and validates a graph. Accepts version 2, and version 1 which it migrates to identical math
 * (CONTRACTS.md section 2). Fills defaults, rejects unknown fields, gives one plain sentence on failure.
 */
export function parseGraph(json: unknown): ParseResult {
  if (!isRecord(json)) return { ok: false, error: 'The graph must be a JSON object.' };
  if (json['version'] === 1) {
    const v1 = parseGraphV1(json);
    return v1.ok ? { ok: true, graph: migrateV1(v1.graph) } : v1;
  }
  // Node types are checked first so that an unknown type gets a clear sentence.
  if (Array.isArray(json['nodes'])) {
    for (const [i, raw] of json['nodes'].entries()) {
      if (!isRecord(raw)) return { ok: false, error: `Node number ${i + 1} must be an object.` };
      const type = raw['type'];
      if (typeof type !== 'string' || !(NODE_TYPES as readonly string[]).includes(type)) {
        const id = typeof raw['id'] === 'string' ? `"${raw['id']}"` : `number ${i + 1}`;
        if (type === undefined) {
          return {
            ok: false,
            error: `Node ${id} has no type; the types are ${NODE_TYPES.join(', ')}.`,
          };
        }
        const old =
          type === 'output' || type === 'loss'
            ? ` (the "${type}" piece only exists in version 1 files)`
            : '';
        return {
          ok: false,
          error: `Node ${id} has an unknown type ${JSON.stringify(type)}${old}; the types are ${NODE_TYPES.join(', ')}.`,
        };
      }
    }
  }
  const parsed = graphSchema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    return { ok: false, error: issue ? describeIssue(issue, json) : 'The graph is not valid.' };
  }
  const g = parsed.data;
  const nodes: Node[] = g.nodes;
  const seen = new Set<string>();
  for (const n of nodes) {
    if (seen.has(n.id)) return { ok: false, error: `Two nodes are both called "${n.id}".` };
    seen.add(n.id);
  }
  for (const n of nodes) {
    if (n.type !== 'weights') continue;
    const target = nodes.find((m) => m.id === n.of);
    if (target?.type !== 'linear') {
      return {
        ok: false,
        error: `Node "${n.id}" field "of" must name a Linear piece, but "${n.of}" is not one.`,
      };
    }
  }
  for (const edge of g.edges) {
    const r = resolveEdge(nodes, edge);
    if (!r.ok) return { ok: false, error: r.error };
  }
  const pinned = new Set<string>();
  for (const ref of g.predictions) {
    const r = resolvePin(nodes, ref);
    if (!r.ok) return { ok: false, error: r.error };
    if (pinned.has(r.at.node)) {
      return { ok: false, error: `"${r.at.node}" is pinned as a prediction twice.` };
    }
    pinned.add(r.at.node);
  }
  if (g.loss !== null) {
    const r = resolvePin(nodes, g.loss);
    if (!r.ok) return { ok: false, error: r.error };
  }
  return {
    ok: true,
    graph: {
      version: GRAPH_VERSION,
      nodes,
      edges: g.edges,
      predictions: g.predictions,
      loss: g.loss,
      data: g.data,
      train: g.train,
    },
  };
}

// ---------------------------------------------------------------- serialise

const NODE_KEY_ORDER: Record<PieceType, string[]> = {
  input: ['id', 'type', 'columns'],
  target: ['id', 'type'],
  feature: ['id', 'type', 'degree', 'sin', 'cos', 'products'],
  linear: ['id', 'type', 'out', 'bias'],
  activation: ['id', 'type', 'fn'],
  dropout: ['id', 'type', 'p'],
  add: ['id', 'type'],
  link: ['id', 'type', 'fn'],
  subtract: ['id', 'type'],
  square: ['id', 'type'],
  abs: ['id', 'type'],
  huber: ['id', 'type', 'delta'],
  log_loss: ['id', 'type'],
  cross_entropy: ['id', 'type'],
  mean: ['id', 'type'],
  scale: ['id', 'type', 'c'],
  weights: ['id', 'type', 'of'],
  norm: ['id', 'type', 'kind'],
};

function pick(obj: object, keys: string[]): Record<string, unknown> {
  const src = obj as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of keys) if (src[k] !== undefined) out[k] = src[k];
  return out;
}

/** Two-space JSON with a stable key order. Always writes version 2. */
export function serializeGraph(graph: Graph): string {
  return JSON.stringify(
    {
      version: GRAPH_VERSION,
      nodes: graph.nodes.map((n) => pick(n, NODE_KEY_ORDER[n.type])),
      edges: graph.edges,
      predictions: graph.predictions,
      loss: graph.loss,
      data: pick(graph.data, ['set', 'n', 'noise', 'val', 'seed', 'standardize']),
      train: pick(graph.train, ['optimizer', 'lr', 'batch', 'seed']),
    },
    null,
    2,
  );
}
