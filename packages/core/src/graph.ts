/** Graph format (CONTRACTS.md section 2): types, zod schema, parse and serialise, port helpers. */
import { z } from 'zod';
import { CORE_PIECES } from './registry.js';

export const DATASET_NAMES = [
  'line',
  'quadratic',
  'sine',
  'step',
  'wiggle',
  'plane',
  'bowl',
  'saddle',
  'blobs',
  'circle',
  'xor',
  'moons',
] as const;
export const ACTIVATION_FNS = ['sigmoid', 'tanh', 'relu', 'gelu'] as const;
export const OUTPUT_LINKS = ['identity', 'sigmoid', 'softmax'] as const;
export const LOSS_FNS = [
  'squared_error',
  'absolute_error',
  'huber',
  'log_loss',
  'cross_entropy',
] as const;
export const OPTIMIZERS = ['sgd', 'momentum', 'adam'] as const;
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

const oneOf = (values: readonly string[]) => `must be one of ${values.join(', ')}`;
const NODE_NAME = 'must be a node name (text)';

function num(message: string, ok: (n: number) => boolean) {
  return z
    .number({ error: message })
    .refine((n) => Number.isFinite(n) && ok(n), { error: message });
}
const int = (lo: number, hi: number) =>
  num(
    `must be a whole number from ${lo} to ${hi}`,
    (n) => Number.isInteger(n) && n >= lo && n <= hi,
  );
const bool = z.boolean({ error: 'must be true or false' });

const nodeId = z.string({ error: 'must be text' }).regex(/^[a-z][a-z0-9_]*$/, {
  error: 'must start with a lowercase letter and use only a-z, 0-9 and _',
});

const penaltySchema = z.strictObject({
  kind: z.enum(['l2', 'l1'], { error: oneOf(['l2', 'l1']) }),
  lambda: num('must be a number of at least 0', (n) => n >= 0),
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

const edgeSchema = z.tuple([z.string({ error: NODE_NAME }), z.string({ error: NODE_NAME })], {
  error: 'must be a pair of node names',
});

const dataSchema = z.strictObject({
  set: z.enum(DATASET_NAMES, { error: oneOf(DATASET_NAMES) }),
  n: int(10, 500),
  noise: num('must be a number from 0 to 1', (n) => n >= 0 && n <= 1),
  val: num('must be a number from 0 to 0.5', (n) => n >= 0 && n <= 0.5),
  seed: int(-2147483648, 4294967295),
  standardize: bool.default(false),
});

const trainSchema = z.strictObject({
  optimizer: z.enum(OPTIMIZERS, { error: oneOf(OPTIMIZERS) }),
  lr: num('must be a number above 0', (n) => n > 0),
  batch: z.union([z.literal('full'), int(1, 1_000_000)], {
    error: 'must be "full" or a whole number of at least 1',
  }),
  seed: int(-2147483648, 4294967295),
});

const graphSchema = z.strictObject({
  version: z.literal(1, { error: 'must be 1' }),
  nodes: z.array(nodeSchema, { error: 'must be a list of nodes' }),
  edges: z.array(edgeSchema, { error: 'must be a list of edges' }),
  data: dataSchema,
  train: trainSchema,
});

export type NodeId = string;
export type Node = z.infer<typeof nodeSchema>;
export type PieceType = Node['type'];
export type NodeOf<T extends PieceType> = Extract<Node, { type: T }>;
/** A node without its id (distributes over the union, unlike `Omit<Node, 'id'>`). */
export type NodeBody = Node extends infer N ? (N extends Node ? Omit<N, 'id'> : never) : never;
export type Penalty = z.infer<typeof penaltySchema>;
export type ActivationFn = (typeof ACTIVATION_FNS)[number];
export type OutputLink = (typeof OUTPUT_LINKS)[number];
export type LossFn = (typeof LOSS_FNS)[number];
export type DatasetName = (typeof DATASET_NAMES)[number];
export type PortRef = string;
export type Edge = [from: PortRef, to: PortRef];
export type DataConfig = z.infer<typeof dataSchema>;
export type TrainConfig = z.infer<typeof trainSchema>;
export type Graph = {
  version: 1;
  nodes: Node[];
  edges: Edge[];
  data: DataConfig;
  train: TrainConfig;
};

// ---------------------------------------------------------------- ports

/** Splits "node" or "node.port". `port` is null for a bare node id. */
export function parsePort(ref: PortRef): { node: NodeId; port: string | null } {
  const dot = ref.indexOf('.');
  if (dot < 0) return { node: ref, port: null };
  return { node: ref.slice(0, dot), port: ref.slice(dot + 1) };
}

export type Endpoint = { node: NodeId; port: string };
export type ResolvedEdge =
  { ok: true; from: Endpoint; to: Endpoint } | { ok: false; error: string };

/** Input port names of a node type (CONTRACTS.md section 2). */
export function inputPortsOf(type: PieceType): string[] {
  return CORE_PIECES[type].inputPorts;
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

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

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

// ---------------------------------------------------------------- serialise

const NODE_KEY_ORDER: Record<PieceType, string[]> = {
  input: ['id', 'type', 'columns'],
  feature: ['id', 'type', 'degree', 'sin', 'cos', 'products'],
  linear: ['id', 'type', 'out', 'bias'],
  activation: ['id', 'type', 'fn'],
  dropout: ['id', 'type', 'p'],
  add: ['id', 'type'],
  output: ['id', 'type', 'link'],
  target: ['id', 'type'],
  loss: ['id', 'type', 'fn', 'delta', 'penalty'],
};

function pick(obj: object, keys: string[]): Record<string, unknown> {
  const src = obj as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of keys) if (src[k] !== undefined) out[k] = src[k];
  return out;
}

/** Two-space JSON with a stable key order. */
export function serializeGraph(graph: Graph): string {
  const nodes = graph.nodes.map((n) => {
    const o = pick(n, NODE_KEY_ORDER[n.type]);
    if (n.type === 'loss' && n.penalty) o['penalty'] = pick(n.penalty, ['kind', 'lambda']);
    return o;
  });
  return JSON.stringify(
    {
      version: graph.version,
      nodes,
      edges: graph.edges,
      data: pick(graph.data, ['set', 'n', 'noise', 'val', 'seed', 'standardize']),
      train: pick(graph.train, ['optimizer', 'lr', 'batch', 'seed']),
    },
    null,
    2,
  );
}
