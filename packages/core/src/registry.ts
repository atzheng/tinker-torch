/** Piece registry v2, core half (CONTRACTS.md section 3): ports, defaults, shape rules, param shapes, torch strings. */
import type { DataInfo, Node, NodeBody, NodeOf, PieceType } from './graph.js';
import type { NodeId } from './common.js';
import {
  lossSubgraph,
  weightPenaltySubgraph,
  type LossKind,
  type Namer,
  type Subgraph,
} from './macros.js';

export type { PieceType } from './graph.js';

export type Family =
  | 'inputs'
  | 'features'
  | 'linear'
  | 'activations'
  | 'structure'
  | 'links'
  | 'errors'
  | 'losses'
  | 'reductions'
  | 'math'
  | 'parameters'
  | 'scalars';

/**
 * What flows along a wire. `example`: one row of `width` numbers per training example (batch rows
 * are implicit). `scalar`: a single number (what a Mean gives, and what L must be). `param`: a
 * parameter matrix read by a Weights tap, never batched.
 */
export type Shape =
  | { kind: 'example'; width: number }
  | { kind: 'scalar' }
  | { kind: 'param'; rows: number; cols: number };
export type ShapeError = { error: string };

export type ShapeCtx = {
  dataColumns: number;
  numClasses: number;
  /** [rows, cols] of a Linear's W, or undefined when that Linear's input width is unknown. */
  weightShape(linearId: NodeId): [rows: number, cols: number] | undefined;
};

/** Named parameter tensors of one node (CONTRACTS.md section 5): Linear has W [in, out] and b [out]. */
export type ParamShapes = Record<string, number[]>;
/** How a node's tensors start: uniform on +/- bound, or every value `value`. Plain data, one spec per node. */
export type ParamInit = { kind: 'uniform'; bound: number } | { kind: 'const'; value: number };

export type CorePiece<N extends Node> = {
  /** Drop-to-swap is allowed only within a family. */
  family: Family;
  inputPorts: string[];
  /**
   * Input ports (a subset of `inputPorts`) that take any number of wires, at least one. Absent or
   * empty: every port takes exactly one wire. In `outShape`'s `inShapes`, a variadic port holds the
   * shape every wire into it has (analyse checks they agree before calling `outShape`; a missing
   * entry means no wire yet). The wires' order is their order in `graph.edges`.
   */
  variadicPorts?: readonly string[];
  defaults: () => Omit<N, 'id'>;
  /**
   * `inShapes` is keyed by input port. A missing entry means that port is not connected (or its
   * shape is unknown), which gives an error sentence. Every piece has the one output port `out`.
   */
  outShape(node: N, inShapes: Record<string, Shape>, ctx: ShapeCtx): Shape | ShapeError;
  /**
   * The node's named parameter tensors (empty for a piece with none). Linear: W [in, out] and b [out].
   * `inWidth` is the example width entering the node (0 for a piece with no input port). A node owns
   * parameters when this is non-empty (see `ownsParams`).
   */
  paramShapes(node: N, inWidth: number): ParamShapes;
  /** How the node's tensors start. Called only when `paramShapes` is non-empty. */
  paramInit(node: N, inWidth: number, data: DataInfo): ParamInit;
  /** Torch counterpart for the later code view. `inWidth` is the width entering the node, when known. */
  torch(node: N, inWidth?: number): string;
};

export type PaletteEntry =
  | {
      kind: 'piece';
      key: string;
      label: string;
      family: Family;
      make: () => NodeBody;
    }
  | {
      /** A ready-made group: inserts primitive nodes and edges (CONTRACTS.md section 3). */
      kind: 'macro';
      key: string;
      label: string;
      family: 'losses' | 'parameters';
      make: (name: Namer, opts?: { linears?: readonly NodeId[] }) => Subgraph;
    };

const py = {
  bool: (b: boolean) => (b ? 'True' : 'False'),
  float: (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x)),
};

export const describeShape = (s: Shape): string =>
  s.kind === 'example'
    ? `${s.width} per example`
    : s.kind === 'scalar'
      ? 'a single number'
      : `a ${s.rows} by ${s.cols} weight matrix`;

/** Whether two wire shapes are the same kind and size. */
export const sameShape = (a: Shape, b: Shape): boolean =>
  a.kind === b.kind &&
  (a.kind !== 'example' || a.width === (b as typeof a).width) &&
  (a.kind !== 'param' || (a.rows === (b as typeof a).rows && a.cols === (b as typeof a).cols));

function needInput(label: string, s: Shape | undefined): Shape | ShapeError {
  return s === undefined ? { error: `${label} needs an input.` } : s;
}

/** The single input `in`, which must be a per-example value; gives its width. */
function exampleIn(
  label: string,
  inShapes: Record<string, Shape>,
  port = 'in',
): number | ShapeError {
  const s = needInput(label, inShapes[port]);
  if ('error' in s) return s;
  if (s.kind !== 'example') {
    return { error: `${label} needs a per-example input, but this one is ${describeShape(s)}.` };
  }
  return s.width;
}

/** Elementwise piece: the shape of `in` passes through unchanged. */
function elementwise(label: string) {
  return (_node: unknown, inShapes: Record<string, Shape>): Shape | ShapeError =>
    needInput(label, inShapes['in']);
}

/** Per-example width of a pass-through piece. */
function exampleThrough(label: string) {
  return (_node: unknown, inShapes: Record<string, Shape>): Shape | ShapeError => {
    const w = exampleIn(label, inShapes);
    return typeof w === 'number' ? { kind: 'example', width: w } : w;
  };
}

/** a and b must have the same shape; the result has it too. */
function pairwise(label: string) {
  return (_node: unknown, inShapes: Record<string, Shape>): Shape | ShapeError => {
    const a = inShapes['a'];
    const b = inShapes['b'];
    if (a === undefined || b === undefined) {
      return { error: `${label} needs both of its inputs connected.` };
    }
    if (!sameShape(a, b)) {
      if (a.kind === 'example' && b.kind === 'example') {
        return {
          error: `${label} needs both inputs to have the same width: ${a.width} and ${b.width}.`,
        };
      }
      return {
        error: `${label} needs both inputs to have the same shape: ${describeShape(a)} and ${describeShape(b)}.`,
      };
    }
    return a;
  };
}

const noParams = () => ({});
const noInit = (): ParamInit => ({ kind: 'const', value: 0 });

function columnIndex(name: string, position: number): number {
  const m = /^x(\d+)$/.exec(name);
  return m?.[1] !== undefined ? Number(m[1]) - 1 : position;
}

/** Output width of a feature expansion of d columns (CONTRACTS.md section 3). */
export function featureWidth(
  d: number,
  f: { degree: number; sin: boolean; cos: boolean; products: boolean },
): number {
  return d * f.degree + (f.sin ? d : 0) + (f.cos ? d : 0) + (f.products ? (d * (d - 1)) / 2 : 0);
}

export const CORE_PIECES: { [T in PieceType]: CorePiece<NodeOf<T>> } = {
  input: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'input', columns: ['x1'] }),
    outShape: (node) =>
      node.columns.length === 0
        ? { error: 'Input needs at least one data column switched on.' }
        : { kind: 'example', width: node.columns.length },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => `x[:, [${node.columns.map((c, i) => columnIndex(c, i)).join(', ')}]]`,
  },
  column: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'column', name: 'x1' }),
    outShape: () => ({ kind: 'example', width: 1 }),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => `x[:, [${columnIndex(node.name, 0)}]]`,
  },
  scalew: {
    family: 'scalars',
    inputPorts: ['in'],
    defaults: () => ({ type: 'scalew' }),
    outShape: exampleThrough('A weight'),
    paramShapes: () => ({ W: [1, 1] }),
    // A scalar model starts at the scale of nn.Linear(d, 1): uniform on +/- 1/sqrt(d).
    paramInit: (_node, _inWidth, data) => ({
      kind: 'uniform',
      bound: 1 / Math.sqrt(Math.max(1, data.columns.length)),
    }),
    torch: () => 'w * x',
  },
  bias: {
    family: 'scalars',
    inputPorts: ['in'],
    defaults: () => ({ type: 'bias' }),
    outShape: exampleThrough('A bias'),
    paramShapes: () => ({ b: [1] }),
    paramInit: (_node, _inWidth, data) => ({
      kind: 'uniform',
      bound: 1 / Math.sqrt(Math.max(1, data.columns.length)),
    }),
    torch: () => 'x + b',
  },
  sum: {
    family: 'structure',
    inputPorts: ['in'],
    variadicPorts: ['in'],
    defaults: () => ({ type: 'sum' }),
    // Analyse has already checked every wire into `in` has the same shape.
    outShape: (_node, inShapes) => needInput('A sum', inShapes['in']),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'torch.stack(xs).sum(0)',
  },
  target: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'target' }),
    // 1 for regression and binary sets (a class index for softmax classification).
    outShape: () => ({ kind: 'example', width: 1 }),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'y',
  },
  feature: {
    family: 'features',
    inputPorts: ['in'],
    defaults: () => ({ type: 'feature', degree: 2, sin: false, cos: false, products: false }),
    outShape: (node, inShapes) => {
      const d = exampleIn('Feature', inShapes);
      return typeof d === 'number' ? { kind: 'example', width: featureWidth(d, node) } : d;
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, inWidth) => {
      const terms = ['x'];
      for (let k = 2; k <= node.degree; k++) terms.push(`x**${k}`);
      if (node.sin) terms.push('torch.sin(x)');
      if (node.cos) terms.push('torch.cos(x)');
      if (node.products) {
        if (inWidth === undefined) terms.push('x[:, [?]] * x[:, [?]]');
        else
          for (let i = 0; i < inWidth; i++)
            for (let j = i + 1; j < inWidth; j++) terms.push(`x[:, [${i}]] * x[:, [${j}]]`);
      }
      return `torch.cat([${terms.join(', ')}], dim=-1)`;
    },
  },
  linear: {
    family: 'linear',
    inputPorts: ['in'],
    defaults: () => ({ type: 'linear', out: 4, bias: true }),
    outShape: (node, inShapes) => {
      const d = exampleIn('Linear', inShapes);
      return typeof d === 'number' ? { kind: 'example', width: node.out } : d;
    },
    paramShapes: (node, inWidth) => ({
      W: [inWidth, node.out],
      ...(node.bias ? { b: [node.out] } : {}),
    }),
    // nn.Linear default: W and b uniform on +/- 1/sqrt(in).
    paramInit: (_node, inWidth) => ({ kind: 'uniform', bound: 1 / Math.sqrt(inWidth) }),
    torch: (node, inWidth) =>
      `nn.Linear(${inWidth ?? '?'}, ${node.out}, bias=${py.bool(node.bias)})`,
  },
  activation: {
    family: 'activations',
    inputPorts: ['in'],
    defaults: () => ({ type: 'activation', fn: 'relu' }),
    outShape: exampleThrough('An activation'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) =>
      ({ sigmoid: 'nn.Sigmoid()', tanh: 'nn.Tanh()', relu: 'nn.ReLU()', gelu: 'nn.GELU()' })[
        node.fn
      ],
  },
  dropout: {
    family: 'structure',
    inputPorts: ['in'],
    defaults: () => ({ type: 'dropout', p: 0.2 }),
    outShape: exampleThrough('Dropout'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => `nn.Dropout(p=${py.float(node.p)})`,
  },
  add: {
    family: 'structure',
    inputPorts: ['a', 'b'],
    defaults: () => ({ type: 'add' }),
    outShape: pairwise('Add'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'a + b',
  },
  link: {
    family: 'links',
    inputPorts: ['in'],
    defaults: () => ({ type: 'link', fn: 'sigmoid' }),
    outShape: exampleThrough('A link'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => (node.fn === 'sigmoid' ? 'nn.Sigmoid()' : 'nn.Softmax(dim=-1)'),
  },
  subtract: {
    family: 'math',
    inputPorts: ['a', 'b'],
    defaults: () => ({ type: 'subtract' }),
    outShape: pairwise('Subtract'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'a - b',
  },
  square: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'square' }),
    outShape: elementwise('Square'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'x ** 2',
  },
  abs: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'abs' }),
    outShape: elementwise('Absolute value'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'x.abs()',
  },
  huber: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'huber', delta: 1 }),
    outShape: elementwise('Huber'),
    paramShapes: noParams,
    paramInit: noInit,
    // Per-element Huber of a difference d (0.5 d^2 inside delta, delta (|d| - 0.5 delta) outside).
    torch: (node) =>
      `F.huber_loss(d, torch.zeros_like(d), delta=${py.float(node.delta)}, reduction='none')`,
  },
  log_loss: {
    family: 'losses',
    inputPorts: ['pred', 'target'],
    defaults: () => ({ type: 'log_loss' }),
    outShape: (_node, inShapes) => {
      const p = exampleIn('Log loss', inShapes, 'pred');
      if (typeof p !== 'number') return p;
      const t = exampleIn('Log loss', inShapes, 'target');
      if (typeof t !== 'number') return t;
      if (p !== 1)
        return { error: `Log loss needs a prediction of width 1, but this one is ${p}.` };
      return { kind: 'example', width: 1 };
    },
    paramShapes: noParams,
    paramInit: noInit,
    // The engine fuses a sigmoid link into this (BCE with logits).
    torch: () => "F.binary_cross_entropy(p, y, reduction='none')",
  },
  cross_entropy: {
    family: 'losses',
    inputPorts: ['pred', 'target'],
    defaults: () => ({ type: 'cross_entropy' }),
    outShape: (_node, inShapes) => {
      const p = exampleIn('Cross-entropy', inShapes, 'pred');
      if (typeof p !== 'number') return p;
      const t = exampleIn('Cross-entropy', inShapes, 'target');
      if (typeof t !== 'number') return t;
      return { kind: 'example', width: 1 };
    },
    paramShapes: noParams,
    paramInit: noInit,
    // The engine fuses a softmax link into this (log-softmax, then pick the target class).
    torch: () => "F.nll_loss(torch.log(p), y.long(), reduction='none')",
  },
  mean: {
    family: 'reductions',
    inputPorts: ['in'],
    defaults: () => ({ type: 'mean' }),
    outShape: (_node, inShapes) => {
      const w = exampleIn('Mean', inShapes);
      return typeof w === 'number' ? { kind: 'scalar' } : w;
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => 'x.mean()',
  },
  scale: {
    family: 'math',
    inputPorts: ['in'],
    defaults: () => ({ type: 'scale', c: 1 }),
    outShape: elementwise('Scale'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => `${py.float(node.c)} * x`,
  },
  weights: {
    family: 'parameters',
    inputPorts: [],
    defaults: () => ({ type: 'weights', of: '' }),
    outShape: (node, _in, ctx) => {
      const s = ctx.weightShape(node.of);
      return s
        ? { kind: 'param', rows: s[0], cols: s[1] }
        : { error: `Weights reads the Linear piece "${node.of}", whose size is not known yet.` };
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => `${node.of}.weight`,
  },
  norm: {
    family: 'parameters',
    inputPorts: ['in'],
    defaults: () => ({ type: 'norm', kind: 'l2' }),
    outShape: (_node, inShapes) => {
      const s = needInput('A norm', inShapes['in']);
      if ('error' in s) return s;
      if (s.kind !== 'param') {
        return {
          error: `A norm needs a weight matrix input, but this one is ${describeShape(s)}.`,
        };
      }
      return { kind: 'scalar' };
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node) => (node.kind === 'l2' ? 'w.pow(2).sum()' : 'w.abs().sum()'),
  },
};

const piece = (
  key: string,
  label: string,
  type: PieceType,
  make: () => NodeBody = () => CORE_PIECES[type].defaults() as NodeBody,
): PaletteEntry => ({ kind: 'piece', key, label, family: CORE_PIECES[type].family, make });

const loss = (key: LossKind, label: string): PaletteEntry => ({
  kind: 'macro',
  key: `loss_${key}`,
  label,
  family: 'losses',
  make: (name) => lossSubgraph(key, { name }),
});

/**
 * The palette, in display order: single pieces first, then the ready-made groups. A macro inserts
 * primitive nodes and edges, never a node of its own (CONTRACTS.md section 3).
 */
/** True when the node has parameter tensors of its own (Linear; later the scalar-weight and bias pieces). */
export function ownsParams(node: Node): boolean {
  const piece = CORE_PIECES[node.type] as unknown as CorePiece<Node>;
  return Object.keys(piece.paramShapes(node, 1)).length > 0;
}

export const PALETTE: PaletteEntry[] = [
  piece('input', 'Input', 'input'),
  piece('target', 'Target', 'target'),
  piece('column', 'Column', 'column'),
  piece('feature', 'Feature', 'feature'),
  piece('linear', 'Linear', 'linear'),
  piece('activation', 'Activation', 'activation'),
  piece('dropout', 'Dropout', 'dropout'),
  piece('add', 'Add', 'add'),
  piece('sum', 'Sum', 'sum'),
  piece('scalew', 'Weight', 'scalew'),
  piece('bias', 'Bias', 'bias'),
  piece('sigmoid', 'Sigmoid', 'link', () => ({ type: 'link', fn: 'sigmoid' })),
  piece('softmax', 'Softmax', 'link', () => ({ type: 'link', fn: 'softmax' })),
  piece('subtract', 'Subtract', 'subtract'),
  piece('square', 'Square', 'square'),
  piece('abs', 'Absolute value', 'abs'),
  piece('huber', 'Huber', 'huber'),
  piece('log_loss', 'Log loss', 'log_loss'),
  piece('cross_entropy', 'Cross-entropy', 'cross_entropy'),
  piece('mean', 'Mean', 'mean'),
  piece('scale', 'Scale', 'scale'),
  piece('weights', 'Weights', 'weights'),
  piece('l2', 'L2 norm', 'norm', () => ({ type: 'norm', kind: 'l2' })),
  piece('l1', 'L1 norm', 'norm', () => ({ type: 'norm', kind: 'l1' })),
  loss('squared_error', 'Squared error'),
  loss('absolute_error', 'Absolute error'),
  loss('huber', 'Huber loss'),
  loss('log_loss', 'Log loss (mean)'),
  loss('cross_entropy', 'Cross-entropy (mean)'),
  {
    kind: 'macro',
    key: 'weight_penalty',
    label: 'Weight penalty',
    family: 'parameters',
    // The caller passes the Linear pieces to read; with none, the group is empty.
    make: (name, opts) =>
      weightPenaltySubgraph({ kind: 'l2', lambda: 0.01, linears: opts?.linears ?? [], name }),
  },
];
