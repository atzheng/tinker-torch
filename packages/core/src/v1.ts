/**
 * Graph format version 1, read only (CONTRACTS.md section 2, "Version 1 migration"). This is the one place
 * version 1 documents are parsed; `migrate.ts` turns the result into a version 2 graph. Nothing else
 * in core, the engine or the web app knows version 1.
 */
import { z } from 'zod';
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

export const OUTPUT_LINKS = ['identity', 'sigmoid', 'softmax'] as const;
export const LOSS_FNS = [
  'squared_error',
  'absolute_error',
  'huber',
  'log_loss',
  'cross_entropy',
] as const;
export const NODE_TYPES = [
  'input',
  'feature',
  'linear',
  'activation',
  'dropout',
  'add',
  'output',
  'target',
  'loss',
] as const;

const penaltySchema = z.strictObject({
  kind: z.enum(['l2', 'l1'], { error: oneOf(['l2', 'l1']) }),
  lambda: num('must be a number from 0 to 100', (n) => n >= 0 && n <= 100),
});

const nodeSchema = z.discriminatedUnion('type', [
  z.strictObject({
    id: nodeId,
    type: z.literal('input'),
    columns: z.array(z.string({ error: 'must be a list of column names' }), {
      error: 'must be a list of column names',
    }),
  }),
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
    type: z.literal('output'),
    link: z.enum(OUTPUT_LINKS, { error: oneOf(OUTPUT_LINKS) }).default('identity'),
  }),
  z.strictObject({ id: nodeId, type: z.literal('target') }),
  z
    .strictObject({
      id: nodeId,
      type: z.literal('loss'),
      fn: z.enum(LOSS_FNS, { error: oneOf(LOSS_FNS) }),
      delta: num('must be a number above 0', (n) => n > 0).optional(),
      penalty: penaltySchema.optional(),
    })
    .refine((n) => n.delta === undefined || n.fn === 'huber', {
      path: ['delta'],
      error: 'can only be set on the Huber loss',
    }),
]);

const graphSchema = z.strictObject({
  version: z.literal(1, { error: 'must be 1' }),
  nodes: z.array(nodeSchema, { error: 'must be a list of nodes' }),
  edges: z.array(edgeSchema, { error: 'must be a list of edges' }),
  data: dataSchema,
  train: trainSchema,
});

export type Node = z.infer<typeof nodeSchema>;
export type PieceType = Node['type'];
export type NodeOf<T extends PieceType> = Extract<Node, { type: T }>;
/** A node without its id (distributes over the union, unlike `Omit<Node, 'id'>`). */
export type NodeBody = Node extends infer N ? (N extends Node ? Omit<N, 'id'> : never) : never;
export type Penalty = z.infer<typeof penaltySchema>;
export type OutputLink = (typeof OUTPUT_LINKS)[number];
export type LossFn = (typeof LOSS_FNS)[number];
export type Graph = {
  version: 1;
  nodes: Node[];
  edges: Edge[];
  data: DataConfig;
  train: TrainConfig;
};

// ---------------------------------------------------------------- ports

export type ResolvedEdge =
  { ok: true; from: Endpoint; to: Endpoint } | { ok: false; error: string };

/** Input port names of a node type (CONTRACTS.md section 2). */
export function inputPortsOf(type: PieceType): string[] {
  return type === 'input' || type === 'target'
    ? []
    : type === 'loss'
      ? ['pred', 'target']
      : type === 'add'
        ? ['a', 'b']
        : ['in'];
}

/** Whether nodes of this type have the output port `out`. */
export function hasOutput(type: PieceType): boolean {
  return type !== 'loss';
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
  if (!hasOutput(from.type) || (src.port !== null && src.port !== 'out')) {
    return {
      ok: false,
      error: hasOutput(from.type)
        ? `${text} starts at "${fromRef}", but "${src.node}" only has the output "out".`
        : `${text} starts at "${src.node}", which has no output.`,
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

/** Parses and validates a graph. Fills defaults, rejects unknown fields, gives one plain sentence on failure. */
export function parseGraph(json: unknown): ParseResult {
  if (!isRecord(json)) return { ok: false, error: 'The graph must be a JSON object.' };
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
        return {
          ok: false,
          error: `Node ${id} has an unknown type ${JSON.stringify(type)}; the types are ${NODE_TYPES.join(', ')}.`,
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
  const nodes: Node[] = g.nodes.map((n) =>
    n.type === 'loss' && n.fn === 'huber' && n.delta === undefined ? { ...n, delta: 1 } : n,
  );
  const seen = new Set<string>();
  for (const n of nodes) {
    if (seen.has(n.id)) return { ok: false, error: `Two nodes are both called "${n.id}".` };
    seen.add(n.id);
  }
  for (const edge of g.edges) {
    const r = resolveEdge(nodes, edge);
    if (!r.ok) return { ok: false, error: r.error };
  }
  return {
    ok: true,
    graph: { version: 1, nodes, edges: g.edges, data: g.data, train: g.train },
  };
}
