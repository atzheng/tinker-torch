/** Piece registry v2, core half (CONTRACTS.md section 3): ports, defaults, shape rules, param shapes, torch emitters. */
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
  | 'scalars'
  | 'categories';

/**
 * What flows along a wire. `example`: one row of `width` numbers per training example (batch rows
 * are implicit). `scalar`: a single number (what a Mean gives, and what L must be). `param`: a
 * parameter matrix read by a Weights tap, never batched.
 */
export type Shape =
  | {
      kind: 'example';
      width: number;
      /**
       * Set only on a category column (an Input or Column of one categorical dataset column): the
       * value is the category's index, width 1, `count` categories. Only pieces that list the port in
       * `categoryPorts` accept it (One-hot, Lookup); analyse refuses it anywhere else.
       */
      category?: { column: string; count: number };
    }
  | { kind: 'scalar' }
  | { kind: 'param'; rows: number; cols: number };
export type ShapeError = { error: string };

export type ShapeCtx = {
  dataColumns: number;
  /** Number of categories by column id, for the categorical columns of the dataset (absent: all numeric). */
  categories?: Record<string, number>;
  numClasses: number;
  /**
   * Fitted leaf count by Tree node id (CONTRACTS.md section 3, "Tree"). A Tree's output width is its
   * entry; absent entry: the tree has not been fitted yet. Absent object: no tree is fitted.
   */
  treeLeaves?: Record<NodeId, number>;
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
  /**
   * Input ports (a subset of `inputPorts`) that may stay unwired. A missing wire there is not an
   * issue and `outShape` gets no entry for the port. Absent or empty: every port is required.
   */
  optionalPorts?: readonly string[];
  /**
   * Input ports (a subset of `inputPorts`) that take a category column. Every other port refuses
   * one (analyse gives the sentence); these ports need one (the piece's `outShape` says so).
   */
  categoryPorts?: readonly string[];
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
  paramShapes(node: N, inWidth: number, inShape?: Shape): ParamShapes;
  /** How the node's tensors start. Called only when `paramShapes` is non-empty. */
  paramInit(node: N, inWidth: number, data: DataInfo): ParamInit;
  /**
   * Torch counterpart for the generated script (CONTRACTS.md section 14). Pure text: the statement
   * `<name> = <expr>` and the module line `self.<name> = <init>` are put together by `torchScript`.
   */
  torch(node: N, ctx: TorchCtx): TorchEmit;
};

/** What a piece's torch emitter may read (CONTRACTS.md section 3, "Torch emitters"). */
export type TorchCtx = {
  /**
   * Python variable names of the wires into each input port, in `analysis.inputs` order. A plain
   * port has one entry, a variadic port one per wire, an unwired optional port none.
   */
  ins: Record<string, string[]>;
  /** Example width entering the node (0 when nothing is wired in). */
  inWidth: number;
  /** The dataset's column ids (`DataInfo.columns`): an Input or Column indexes `x` by position here. */
  dataColumns: readonly string[];
  /** Category count of the node's category input (OneHot, Lookup); 0 otherwise. */
  categories: number;
  /** Uniform init bound of the node's own parameters (`analysis.paramInit`); 1 when it owns none. */
  bound: number;
  /** Set when a scalar parameter must start inside (0, 1) (it feeds an unfused log loss as a probability): `uniform_(lo, hi)` replaces +/- bound. */
  initRange?: [lo: number, hi: number];
  /** `self` in `Model.forward`, `model` in `objective`: what a parameter owner is reached through. */
  owner: 'self' | 'model';
  /** `<owner>.training`. */
  training: string;
  /** `<owner>.<variable of this node>`: where this node's module or parameter lives. */
  self: string;
  /** Python variable name of any node (a Weights tap reads `<owner>.<name(of)>.weight`). */
  name(id: NodeId): string;
  /** Tree only: its fitted estimator, and the node ids of its leaves, as module-level variables. */
  tree?: { estimator: string; leaves: string; columns: number[] };
};

/**
 * `expr`: the right-hand side of the node's statement. `init`: the right-hand side of the
 * `self.<name> = ...` line in `Model.__init__`, for a piece that owns a module or parameter. `pre`:
 * statements run once before the Model class (a Tree's fit); lines separated by `\n`.
 */
export type TorchEmit = { init?: string; pre?: string; expr: string };

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
    ? s.category
      ? `the category "${s.category.column}"`
      : `${s.width} per example`
    : s.kind === 'scalar'
      ? 'a single number'
      : `a ${s.rows} by ${s.cols} weight matrix`;

/** Whether two wire shapes are the same kind and size. */
export const sameShape = (a: Shape, b: Shape): boolean =>
  a.kind === b.kind &&
  (a.kind !== 'example' ||
    (a.width === (b as typeof a).width &&
      a.category?.column === (b as typeof a).category?.column &&
      a.category?.count === (b as typeof a).category?.count)) &&
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

/** Position of a dataset column in `x` (columns are `x1`, `x2`, ...; `position` is the fallback). */
function dataIndex(ctx: TorchCtx, name: string, position: number): number {
  const i = ctx.dataColumns.indexOf(name);
  if (i >= 0) return i;
  const m = /^x(\d+)$/.exec(name);
  return m?.[1] !== undefined ? Number(m[1]) - 1 : position;
}

/** The `lo, hi` of a scalar parameter's uniform init. */
const scalarRange = (ctx: TorchCtx): string => {
  const [lo, hi] = ctx.initRange ?? [-ctx.bound, ctx.bound];
  return `${py.float(lo)}, ${py.float(hi)}`;
};

/** The variable of the one wire into a plain port (a placeholder name while unwired: `h` for `in`, else the port name). */
const one = (ctx: TorchCtx, port: string): string =>
  ctx.ins[port]?.[0] ?? (port === 'in' ? 'h' : port);

/** Output width of a feature expansion of d columns (CONTRACTS.md section 3). */
export function featureWidth(
  d: number,
  f: { degree: number; sin: boolean; cos: boolean; products: boolean },
): number {
  return d * f.degree + (f.sin ? d : 0) + (f.cos ? d : 0) + (f.products ? (d * (d - 1)) / 2 : 0);
}

/**
 * The shape of an Input or Column over these dataset columns: their count, or, for one category
 * column (a categorical dataset column, stored as its index), a width-1 shape that carries the
 * category count. A category column cannot share an Input with other columns.
 */
function columnsShape(columns: readonly string[], ctx: ShapeCtx): Shape | ShapeError {
  const cats = columns.filter((c) => (ctx.categories?.[c] ?? 0) > 0);
  const first = cats[0];
  if (first === undefined) return { kind: 'example', width: columns.length };
  if (columns.length > 1) {
    return {
      error: `Input mixes the category column "${first}" with other columns; give the category its own Input.`,
    };
  }
  return {
    kind: 'example',
    width: 1,
    category: { column: first, count: ctx.categories?.[first] ?? 0 },
  };
}

/** The `in` shape of One-hot and Lookup: a category column, as its category count. */
function categoryIn(label: string, inShapes: Record<string, Shape>): number | ShapeError {
  const s = needInput(label, inShapes['in']);
  if ('error' in s) return s;
  if (s.kind === 'example' && s.category) return s.category.count;
  return {
    error: `${label} needs a category column, but this input is ${describeShape(s)}; switch on a category column in an Input.`,
  };
}

export const CORE_PIECES: { [T in PieceType]: CorePiece<NodeOf<T>> } = {
  input: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'input', columns: ['x1'] }),
    outShape: (node, _in, ctx) => {
      if (node.columns.length === 0) {
        return { error: 'Input needs at least one data column switched on.' };
      }
      return columnsShape(node.columns, ctx);
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({
      expr: `x[:, [${node.columns.map((c, i) => dataIndex(ctx, c, i)).join(', ')}]]`,
    }),
  },
  column: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'column', name: 'x1' }),
    outShape: (node, _in, ctx) => columnsShape([node.name], ctx),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({ expr: `x[:, [${dataIndex(ctx, node.name, 0)}]]` }),
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
    torch: (_node, ctx) => ({
      init: `nn.Parameter(torch.empty(1, 1).uniform_(${scalarRange(ctx)}))`,
      expr: `${ctx.self} * ${one(ctx, 'in')}`,
    }),
  },
  bias: {
    family: 'scalars',
    inputPorts: ['in'],
    optionalPorts: ['in'],
    defaults: () => ({ type: 'bias' }),
    // Unwired, a Bias is one learnable value for every example: the mean-as-a-weight model.
    outShape: (_node, inShapes) =>
      inShapes['in'] === undefined
        ? { kind: 'example', width: 1 }
        : exampleThrough('A bias')(_node, inShapes),
    paramShapes: () => ({ b: [1] }),
    paramInit: (_node, _inWidth, data) => ({
      kind: 'uniform',
      bound: 1 / Math.sqrt(Math.max(1, data.columns.length)),
    }),
    torch: (_node, ctx) => ({
      init: `nn.Parameter(torch.empty(1, 1).uniform_(${scalarRange(ctx)}))`,
      expr:
        ctx.ins['in']?.[0] === undefined
          ? `${ctx.self}.expand(x.shape[0], 1)`
          : `${one(ctx, 'in')} + ${ctx.self}`,
    }),
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
    torch: (_node, ctx) => ({ expr: `torch.stack([${(ctx.ins['in'] ?? []).join(', ')}]).sum(0)` }),
  },
  onehot: {
    family: 'categories',
    inputPorts: ['in'],
    categoryPorts: ['in'],
    defaults: () => ({ type: 'onehot' }),
    // One column per category, no category dropped (with a bias the columns are redundant, which gradient descent tolerates).
    outShape: (_node, inShapes) => {
      const k = categoryIn('One-hot', inShapes);
      return typeof k === 'number' ? { kind: 'example', width: k } : k;
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (_node, ctx) => ({
      expr: `F.one_hot(${one(ctx, 'in')}[:, 0].long(), ${ctx.categories}).float()`,
    }),
  },
  lookup: {
    family: 'categories',
    inputPorts: ['in'],
    categoryPorts: ['in'],
    defaults: () => ({ type: 'lookup', width: 1 }),
    outShape: (node, inShapes) => {
      const k = categoryIn('Lookup', inShapes);
      return typeof k === 'number' ? { kind: 'example', width: node.width } : k;
    },
    // One row of `width` numbers per category (width 1: one learned number per category).
    paramShapes: (node, _inWidth, inShape) => ({
      W: [inShape?.kind === 'example' ? (inShape.category?.count ?? 1) : 1, node.width],
    }),
    paramInit: () => ({ kind: 'uniform', bound: 1 }),
    torch: (node, ctx) => ({
      init: `nn.Embedding(${ctx.categories}, ${node.width})`,
      expr: `${ctx.self}(${one(ctx, 'in')}[:, 0].long())`,
    }),
  },
  tree: {
    family: 'features',
    inputPorts: ['in'],
    defaults: () => ({
      type: 'tree',
      criterion: 'squared_error',
      depth: 3,
      minSplit: 2,
      minLeaf: 1,
      minDecrease: 0,
    }),
    // One output column per leaf of the fitted tree (the host fits it before analysis).
    outShape: (node, inShapes, ctx) => {
      const d = exampleIn('Tree', inShapes);
      if (typeof d !== 'number') return d;
      const leaves = ctx.treeLeaves?.[node.id];
      return leaves === undefined
        ? { error: 'Tree has not been fitted to data yet.' }
        : { kind: 'example', width: leaves };
    },
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => {
      const cls =
        node.criterion === 'squared_error' || node.criterion === 'absolute_error'
          ? 'Regressor'
          : 'Classifier';
      const t = ctx.tree;
      const cols = (t?.columns ?? []).join(', ');
      const est = t?.estimator ?? 'tree';
      const leaves = t?.leaves ?? 'leaves';
      return {
        // Fitted once on the training rows, like the app does; the one-hot is the leaf each row falls in.
        pre: [
          `${est} = DecisionTree${cls}(criterion='${node.criterion}', max_depth=${node.depth}, min_samples_split=${node.minSplit}, min_samples_leaf=${node.minLeaf}, min_impurity_decrease=${py.float(node.minDecrease)}).fit(x_train[:, [${cols}]].numpy(), y_train.numpy().ravel())`,
          `${leaves} = torch.as_tensor(${est}.tree_.children_left == -1).nonzero().squeeze(1)`,
        ].join('\n'),
        expr: `(torch.as_tensor(${est}.apply(${one(ctx, 'in')}.numpy())).unsqueeze(1) == ${leaves}).float()`,
      };
    },
  },
  target: {
    family: 'inputs',
    inputPorts: [],
    defaults: () => ({ type: 'target' }),
    // 1 for regression and binary sets (a class index for softmax classification).
    outShape: () => ({ kind: 'example', width: 1 }),
    paramShapes: noParams,
    paramInit: noInit,
    torch: () => ({ expr: 'y' }),
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
    torch: (node, ctx) => {
      const h = one(ctx, 'in');
      const terms = [h];
      for (let k = 2; k <= node.degree; k++) terms.push(`${h} ** ${k}`);
      if (node.sin) terms.push(`torch.sin(${h})`);
      if (node.cos) terms.push(`torch.cos(${h})`);
      if (node.products) {
        for (let i = 0; i < ctx.inWidth; i++)
          for (let j = i + 1; j < ctx.inWidth; j++) terms.push(`${h}[:, [${i}]] * ${h}[:, [${j}]]`);
      }
      return { expr: `torch.cat([${terms.join(', ')}], dim=-1)` };
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
    torch: (node, ctx) => ({
      init: `nn.Linear(${ctx.inWidth}, ${node.out}, bias=${py.bool(node.bias)})`,
      expr: `${ctx.self}(${one(ctx, 'in')})`,
    }),
  },
  activation: {
    family: 'activations',
    inputPorts: ['in'],
    defaults: () => ({ type: 'activation', fn: 'relu' }),
    outShape: exampleThrough('An activation'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({
      expr: `${{ sigmoid: 'torch.sigmoid', tanh: 'torch.tanh', relu: 'torch.relu', gelu: 'F.gelu' }[node.fn]}(${one(ctx, 'in')})`,
    }),
  },
  dropout: {
    family: 'structure',
    inputPorts: ['in'],
    defaults: () => ({ type: 'dropout', p: 0.2 }),
    outShape: exampleThrough('Dropout'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({
      expr: `F.dropout(${one(ctx, 'in')}, p=${py.float(node.p)}, training=${ctx.training})`,
    }),
  },
  add: {
    family: 'structure',
    inputPorts: ['a', 'b'],
    defaults: () => ({ type: 'add' }),
    outShape: pairwise('Add'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (_node, ctx) => ({ expr: `${one(ctx, 'a')} + ${one(ctx, 'b')}` }),
  },
  link: {
    family: 'links',
    inputPorts: ['in'],
    defaults: () => ({ type: 'link', fn: 'sigmoid' }),
    outShape: exampleThrough('A link'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({
      expr:
        node.fn === 'sigmoid'
          ? `torch.sigmoid(${one(ctx, 'in')})`
          : `torch.softmax(${one(ctx, 'in')}, dim=-1)`,
    }),
  },
  subtract: {
    family: 'math',
    inputPorts: ['a', 'b'],
    defaults: () => ({ type: 'subtract' }),
    outShape: pairwise('Subtract'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (_node, ctx) => ({ expr: `${one(ctx, 'a')} - ${one(ctx, 'b')}` }),
  },
  square: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'square' }),
    outShape: elementwise('Square'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (_node, ctx) => ({ expr: `${one(ctx, 'in')} ** 2` }),
  },
  abs: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'abs' }),
    outShape: elementwise('Absolute value'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (_node, ctx) => ({ expr: `torch.abs(${one(ctx, 'in')})` }),
  },
  huber: {
    family: 'errors',
    inputPorts: ['in'],
    defaults: () => ({ type: 'huber', delta: 1 }),
    outShape: elementwise('Huber'),
    paramShapes: noParams,
    paramInit: noInit,
    // Per-element Huber of a difference d (0.5 d^2 inside delta, delta (|d| - 0.5 delta) outside).
    torch: (node, ctx) => ({
      expr: `F.huber_loss(${one(ctx, 'in')}, torch.zeros_like(${one(ctx, 'in')}), delta=${py.float(node.delta)}, reduction='none')`,
    }),
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
    // The literal formula (NaN outside (0, 1), where F.binary_cross_entropy would raise); torchScript writes the with-logits idiom when a sigmoid link is fused in front.
    torch: (_node, ctx) => ({
      expr: `-(${one(ctx, 'target')} * torch.log(${one(ctx, 'pred')}) + (1 - ${one(ctx, 'target')}) * torch.log(1 - ${one(ctx, 'pred')}))`,
    }),
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
    // The literal form; torchScript writes F.cross_entropy on the logits when a softmax link is fused in front.
    torch: (_node, ctx) => ({
      expr: `F.nll_loss(torch.log(${one(ctx, 'pred')}), ${one(ctx, 'target')}.long().squeeze(1), reduction='none').unsqueeze(1)`,
    }),
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
    torch: (_node, ctx) => ({ expr: `${one(ctx, 'in')}.mean()` }),
  },
  scale: {
    family: 'math',
    inputPorts: ['in'],
    defaults: () => ({ type: 'scale', c: 1 }),
    outShape: elementwise('Scale'),
    paramShapes: noParams,
    paramInit: noInit,
    torch: (node, ctx) => ({ expr: `${py.float(node.c)} * ${one(ctx, 'in')}` }),
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
    torch: (node, ctx) => ({ expr: `${ctx.owner}.${ctx.name(node.of)}.weight.T` }),
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
    torch: (node, ctx) => ({
      expr: `${one(ctx, 'in')}${node.kind === 'l2' ? '.pow(2).sum()' : '.abs().sum()'}`,
    }),
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
  piece('tree', 'Tree', 'tree'),
  piece('linear', 'Linear', 'linear'),
  piece('activation', 'Activation', 'activation'),
  piece('dropout', 'Dropout', 'dropout'),
  piece('add', 'Add', 'add'),
  piece('sum', 'Sum', 'sum'),
  piece('onehot', 'One-hot', 'onehot'),
  piece('lookup', 'Lookup', 'lookup'),
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
