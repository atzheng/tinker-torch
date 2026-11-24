/**
 * Piece reference cards (CONTRACTS.md section 15, tinker-32v). `PIECE_DOCS` holds the drafted text
 * for every palette entry (reference voice only: what a piece computes, never why or when).
 * Shapes, parameter counts and the torch line are derived from `CORE_PIECES`, never written here.
 * Pure text and data: no DOM, no jax-js. `docView` and `docViewFor` never throw.
 */
import type { Analysis } from './analyse.js';
import { optionalPortsOf, type DataInfo, type Graph, type Node, type NodeBody } from './graph.js';
import {
  CORE_PIECES,
  PALETTE,
  type CorePiece,
  type PaletteEntry,
  type Shape,
  type ShapeCtx,
  type TorchCtx,
} from './registry.js';
import { nodeTorch } from './torchgen.js';

export type PieceDoc = {
  summary: string;
  equation: string;
  settings?: Record<string, string>;
  variants?: Record<string, { equation: string; summary?: string }>;
  notes?: string[];
  seeAlso?: string[];
};

export type DocView = {
  key: string;
  label: string;
  summary: string;
  equation: string;
  shapes?: { in: Record<string, string>; out: string };
  params?: string;
  settings: { field: string; text: string }[];
  torch?: string;
  notes: string[];
  seeAlso: { key: string; label: string }[];
};

export const PIECE_DOCS: Record<string, PieceDoc> = {
  input: {
    summary: 'The data columns that are switched on, one row of numbers per example.',
    equation: 'x',
    settings: { columns: 'The data columns the piece outputs.' },
    notes: ['Takes no input.', 'A category column needs an Input of its own.'],
    seeAlso: ['column', 'target'],
  },
  target: {
    summary: 'The true answers of the dataset, one number per example.',
    equation: 'y',
    notes: ['In a classification set the number is the class index.'],
    seeAlso: ['input'],
  },
  column: {
    summary: 'One named data column, output as a single number per example.',
    equation: 'x_{j}',
    settings: { name: 'The data column the piece outputs.' },
    notes: ['A category column goes only to One-hot or Lookup.'],
    seeAlso: ['input', 'onehot', 'lookup'],
  },
  feature: {
    summary: 'New columns computed from the input columns: powers, sin, cos and pairwise products.',
    equation: '\\phi(h) = [\\,h,\\ h^2,\\ \\dots,\\ h^k,\\ \\sin h,\\ \\cos h,\\ h_i h_j\\,]',
    settings: {
      degree: 'The highest power k; the columns h through h^k are included.',
      sin: 'Adds the column sin(h) for every input column.',
      cos: 'Adds the column cos(h) for every input column.',
      products: 'Adds the column h_i * h_j for every pair of input columns i < j.',
    },
    notes: [
      'Columns are ordered by power, then sin, then cos, then products.',
      'sin and cos apply to the raw column values with no frequency scaling.',
    ],
    seeAlso: ['linear'],
  },
  tree: {
    summary: 'A decision tree fitted to the training rows; outputs one 0/1 column per leaf.',
    equation: '\\mathrm{leaf}(h)',
    settings: {
      criterion: 'The impurity measure a split is chosen by.',
      depth: 'The largest number of splits from the root to a leaf.',
      minSplit: 'The fewest examples a node needs to be split.',
      minLeaf: 'The fewest examples a leaf may hold.',
      minDecrease: 'The smallest weighted impurity decrease a split must achieve.',
    },
    notes: [
      'Fitted on the training rows only, whenever the data or a tree setting changes.',
      'Reads data columns directly, so its input comes from an Input.',
      'Owns no learned parameters.',
    ],
    seeAlso: ['input', 'linear'],
  },
  linear: {
    summary: 'A weighted sum of the input columns for each of m outputs, plus an optional bias.',
    equation: '\\mathbf{W}h + \\mathbf{b}',
    settings: {
      out: 'The output width m, the number of weighted sums.',
      bias: 'Whether the vector b is added.',
    },
    notes: ['W and b start uniform between minus and plus 1 over the square root of d.'],
    seeAlso: ['activation', 'scalew', 'weights'],
  },
  activation: {
    summary: 'Applies a function to each number separately.',
    equation: 'f(h)',
    settings: { fn: 'The function f: sigmoid, tanh, ReLU or GELU.' },
    variants: {
      sigmoid: {
        equation: '\\sigma(h) = \\frac{1}{1 + e^{-h}}',
        summary: 'Maps each number into (0, 1) as 1 / (1 + e^-h).',
      },
      tanh: {
        equation: '\\tanh(h) = \\frac{e^{h} - e^{-h}}{e^{h} + e^{-h}}',
        summary: 'Maps each number into (-1, 1) as the hyperbolic tangent.',
      },
      relu: {
        equation: '\\mathrm{ReLU}(h) = \\max(0,\\, h)',
        summary: 'Replaces each negative number with 0 and keeps the rest.',
      },
      gelu: {
        equation: '\\mathrm{GELU}(h) = h\\,\\Phi(h)',
        summary: 'Multiplies each number by the standard normal distribution function of it.',
      },
    },
    notes: ['The width does not change.', 'Owns no parameters.'],
    seeAlso: ['linear', 'sigmoid'],
  },
  dropout: {
    summary: 'In training, sets each number to 0 with probability p and divides the rest by 1 - p.',
    equation:
      '\\mathrm{drop}_{p}(h) = \\frac{r \\odot h}{1 - p},\\quad r_j \\sim \\mathrm{Bernoulli}(1 - p)',
    settings: { p: 'The probability that a number is set to 0.' },
    notes: ['Off during evaluation and prediction: the output equals the input.'],
    seeAlso: ['linear'],
  },
  add: {
    summary: 'The elementwise sum of two inputs of the same shape.',
    equation: 'a + b',
    notes: ['Both inputs have the same width, and so does the output.'],
    seeAlso: ['sum', 'subtract'],
  },
  sum: {
    summary: 'The elementwise sum of any number of inputs of the same shape.',
    equation: 'h_1 + h_2 + \\dots + h_k',
    notes: ['Takes any number of wires on its one input.'],
    seeAlso: ['add'],
  },
  onehot: {
    summary: 'Turns a category column into one 0/1 column per category.',
    equation: '\\mathrm{onehot}(h)_j = [\\,h = j\\,]',
    notes: [
      'Needs a category column as input.',
      'Columns follow the dataset category order and no category is left out.',
    ],
    seeAlso: ['lookup', 'column'],
  },
  lookup: {
    summary: 'Maps each category to its own learned row of m numbers.',
    equation: '\\mathrm{lookup}(h) = \\mathbf{W}_{h}',
    settings: { width: 'The numbers learned per category, the output width m.' },
    notes: ['Needs a category column as input.', 'W starts uniform between -1 and 1.'],
    seeAlso: ['onehot', 'column'],
  },
  scalew: {
    summary: 'Multiplies its input by one learned number.',
    equation: 'w\\,h',
    notes: ['The one number is shared by every column and every example.'],
    seeAlso: ['bias', 'linear'],
  },
  bias: {
    summary:
      'Adds one learned number to its input, or outputs that number when nothing is wired in.',
    equation: 'h + b',
    notes: ['With no input the output is b for every example.'],
    seeAlso: ['scalew', 'linear'],
  },
  sigmoid: {
    summary: 'Maps each number into (0, 1) as 1 / (1 + e^-h).',
    equation: '\\sigma(h) = \\frac{1}{1 + e^{-h}}',
    notes: ['A log loss directly after it is computed in one combined, numerically stable form.'],
    seeAlso: ['softmax', 'log_loss', 'activation'],
  },
  softmax: {
    summary: 'Turns each row of numbers into positive probabilities that sum to 1.',
    equation: '\\mathrm{softmax}(h)_j = \\frac{e^{h_j}}{\\sum_k e^{h_k}}',
    notes: [
      'A cross-entropy directly after it is computed in one combined, numerically stable form.',
    ],
    seeAlso: ['sigmoid', 'cross_entropy'],
  },
  subtract: {
    summary: 'The elementwise difference a - b of two inputs of the same shape.',
    equation: 'a - b',
    notes: ['Both inputs have the same width, and so does the output.'],
    seeAlso: ['add', 'square', 'abs'],
  },
  square: {
    summary: 'Squares each number.',
    equation: 'h^2',
    notes: ['Works on any shape and keeps it.'],
    seeAlso: ['abs', 'huber', 'mean'],
  },
  abs: {
    summary: 'The absolute value of each number.',
    equation: '\\left|h\\right|',
    notes: ['Works on any shape and keeps it.'],
    seeAlso: ['square', 'huber', 'mean'],
  },
  huber: {
    summary: 'Squared for differences up to delta, linear beyond it.',
    equation:
      'H_{\\delta}(h) = \\begin{cases} \\tfrac{1}{2} h^2 & |h| \\le \\delta \\\\ \\delta\\,(|h| - \\tfrac{1}{2}\\delta) & |h| > \\delta \\end{cases}',
    settings: { delta: 'The size of the difference at which squared turns into linear.' },
    notes: ['Works on any shape and keeps it.'],
    seeAlso: ['square', 'abs', 'mean'],
  },
  log_loss: {
    summary: 'Per example, the binary log loss of a probability p against a 0/1 target t.',
    equation: '-\\left(t \\log p + (1 - t)\\log(1 - p)\\right)',
    notes: [
      'The prediction has width 1.',
      'A prediction outside (0, 1) gives NaN, which stops the run as diverged.',
    ],
    seeAlso: ['sigmoid', 'mean', 'loss_log_loss'],
  },
  cross_entropy: {
    summary: 'Per example, minus the log of the probability given to the true class.',
    equation: '-\\log p_{t}',
    notes: ['The target is a class index and the prediction holds one probability per class.'],
    seeAlso: ['softmax', 'mean', 'loss_cross_entropy'],
  },
  mean: {
    summary: 'Averages over the examples (and elements) into one number.',
    equation: '\\frac{1}{n}\\sum_i h_i',
    notes: ['The output is a single number.'],
    seeAlso: ['square', 'abs', 'huber'],
  },
  scale: {
    summary: 'Multiplies every number by the constant c.',
    equation: 'c\\,h',
    settings: { c: 'The constant factor.' },
    notes: ['Works on any shape and keeps it.'],
    seeAlso: ['add'],
  },
  weights: {
    summary: 'Reads the weight matrix W of one Linear piece.',
    equation: '\\mathbf{W}',
    settings: { of: 'The Linear piece whose weight matrix is read.' },
    notes: ['The output is a matrix, which a norm accepts.', 'Owns no parameters.'],
    seeAlso: ['l2', 'l1', 'linear'],
  },
  l2: {
    summary: 'The sum of the squared entries of a weight matrix.',
    equation: '\\|\\mathbf{W}\\|_2^2 = \\sum_{ij} W_{ij}^2',
    notes: ['Needs a weight matrix as input and outputs a single number.'],
    seeAlso: ['l1', 'weights', 'weight_penalty'],
  },
  l1: {
    summary: 'The sum of the absolute values of the entries of a weight matrix.',
    equation: '\\|\\mathbf{W}\\|_1 = \\sum_{ij} |W_{ij}|',
    notes: ['Needs a weight matrix as input and outputs a single number.'],
    seeAlso: ['l2', 'weights', 'weight_penalty'],
  },
  loss_squared_error: {
    summary: 'Squared error averaged over the examples.',
    equation: '\\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2',
    notes: ['Inserts Subtract, Square and Mean pieces wired to the prediction and the target.'],
    seeAlso: ['subtract', 'square', 'mean'],
  },
  loss_absolute_error: {
    summary: 'Absolute error averaged over the examples.',
    equation: '\\frac{1}{n}\\sum_i \\left|\\hat{y}_i - y_i\\right|',
    notes: [
      'Inserts Subtract, Absolute value and Mean pieces wired to the prediction and the target.',
    ],
    seeAlso: ['subtract', 'abs', 'mean'],
  },
  loss_huber: {
    summary: 'Huber loss averaged over the examples.',
    equation: '\\frac{1}{n}\\sum_i H_{\\delta}(\\hat{y}_i - y_i)',
    notes: ['Inserts Subtract, Huber and Mean pieces wired to the prediction and the target.'],
    seeAlso: ['subtract', 'huber', 'mean'],
  },
  loss_log_loss: {
    summary: 'Binary log loss averaged over the examples.',
    equation: '-\\frac{1}{n}\\sum_i \\left(y_i \\log p_i + (1 - y_i)\\log(1 - p_i)\\right)',
    notes: ['Inserts Log loss and Mean pieces wired to the prediction and the target.'],
    seeAlso: ['log_loss', 'mean', 'sigmoid'],
  },
  loss_cross_entropy: {
    summary: 'Cross-entropy averaged over the examples.',
    equation: '-\\frac{1}{n}\\sum_i \\log p_{i,\\,y_i}',
    notes: ['Inserts Cross-entropy and Mean pieces wired to the prediction and the target.'],
    seeAlso: ['cross_entropy', 'mean', 'softmax'],
  },
  weight_penalty: {
    summary: "Adds c times the L2 or L1 norm of each Linear piece's weights to the loss.",
    equation: 'L + c \\sum_k \\|\\mathbf{W}_k\\|_2^2',
    notes: [
      'Inserts Weights, Norm, Scale and Add pieces, one Weights and Norm per Linear piece.',
      'Biases are not penalised.',
    ],
    seeAlso: ['weights', 'l2', 'l1'],
  },
};

// ---------------------------------------------------------------- keys

/** The palette key of a node: a link by its function, a norm by its kind, otherwise the type. */
export function paletteKeyOf(node: Node): string {
  if (node.type === 'link') return node.fn;
  if (node.type === 'norm') return node.kind;
  return node.type;
}

const entryOf = (key: string): PaletteEntry | undefined => PALETTE.find((e) => e.key === key);
const labelOf = (key: string): string => entryOf(key)?.label ?? key;

/** Macro cards: shapes and the section 14 idiom (palette only, there is no node). */
const MACROS: Record<string, { in: Record<string, string>; torch: string }> = {
  loss_squared_error: {
    in: { prediction: 'n \\times m', target: 'n \\times m' },
    torch: 'F.mse_loss(y_hat, y)',
  },
  loss_absolute_error: {
    in: { prediction: 'n \\times m', target: 'n \\times m' },
    torch: 'F.l1_loss(y_hat, y)',
  },
  loss_huber: {
    in: { prediction: 'n \\times m', target: 'n \\times m' },
    torch: 'F.huber_loss(y_hat, y, delta=1.0)',
  },
  loss_log_loss: {
    in: { prediction: 'n \\times 1', target: 'n \\times 1' },
    torch: 'F.binary_cross_entropy_with_logits(z, y)',
  },
  loss_cross_entropy: {
    in: { prediction: 'n \\times K', target: 'n \\times 1' },
    torch: 'F.cross_entropy(z, y.long().squeeze(1))',
  },
  weight_penalty: { in: {}, torch: 'c * model.linear.weight.T.pow(2).sum()' },
};

// ---------------------------------------------------------------- symbolic and live pieces

// Sentinel widths: the emitters and shape rules take numbers, so the symbolic card runs them on
// these and prints the symbol back for any of them.
const D = 1_000_001; // input width d
const M = 1_000_002; // output width m
const K = 1_000_003; // categories
const LEAVES = 1_000_004;
const SYMBOLS = new Map<number, string>([
  [D, 'd'],
  [M, 'm'],
  [K, 'K'],
  [LEAVES, '\\ell'],
]);
const dim = (n: number): string => SYMBOLS.get(n) ?? String(n);
/** Output widths the sentinels cannot carry through the shape rule. */
const OUT_WIDTH: Record<string, string> = { input: 'd', feature: "d'" };

const fmtShape = (s: Shape): string =>
  s.kind === 'scalar'
    ? '1'
    : s.kind === 'param'
      ? `${dim(s.rows)} \\times ${dim(s.cols)}`
      : `n \\times ${dim(s.width)}`;

const wordSyms = (s: string): string =>
  s
    .replace(new RegExp(`\\b${D}\\b`, 'g'), 'd')
    .replace(new RegExp(`\\b${M}\\b`, 'g'), 'm')
    .replace(new RegExp(`\\b${K}\\b`, 'g'), 'K')
    .replace(new RegExp(`\\b${LEAVES}\\b`, 'g'), 'leaves');

const hasSentinel = (shapes: number[][]): boolean => shapes.flat().some((n) => SYMBOLS.has(n));

/** `3 \cdot 8 + 8 = 32`, or `d \cdot m + m` while a width is a symbol. */
function fmtParams(shapes: Record<string, number[]>): string | undefined {
  const tensors = Object.values(shapes);
  if (tensors.length === 0) return undefined;
  const terms = tensors.map((t) =>
    tensors.length === 1 && t.every((n) => n === 1) ? '1' : t.map(dim).join(' \\cdot '),
  );
  const expr = terms.join(' + ');
  if (hasSentinel(tensors)) return expr;
  const total = tensors.reduce((s, t) => s + t.reduce((p, n) => p * n, 1), 0);
  return expr === String(total) ? expr : `${expr} = ${total}`;
}

type Parts = {
  shapes?: { in: Record<string, string>; out: string };
  params?: string;
  torch?: string;
};

const typed = (node: Node): CorePiece<Node> => CORE_PIECES[node.type] as unknown as CorePiece<Node>;

/** The torch line: the emitter's `init` when present, else its `expr`. */
const torchLine = (node: Node, ctx: TorchCtx): string => {
  const e = typed(node).torch(node, ctx);
  return e.init ?? e.expr;
};

function baseCtx(over: Partial<TorchCtx>): TorchCtx {
  return {
    ins: {},
    inWidth: D,
    dataColumns: ['x1'],
    categories: 0,
    bound: 1,
    owner: 'self',
    training: 'self.training',
    self: 'self.node',
    name: (id) => id,
    ...over,
  };
}

/** The symbolic node of a palette piece: widths and categories are the sentinels. */
function symbolicNode(key: string): Node | null {
  const e = entryOf(key);
  if (e?.kind !== 'piece') return null;
  const body: NodeBody = e.make();
  const node = { ...body, id: 'h' } as Node;
  if (node.type === 'linear') node.out = M;
  if (node.type === 'lookup') node.width = M;
  if (node.type === 'weights') node.of = 'linear';
  return node;
}

function symbolicIn(node: Node, port: string): Shape {
  if (port === 'target') return { kind: 'example', width: 1 };
  if (node.type === 'log_loss' && port === 'pred') return { kind: 'example', width: 1 };
  if (node.type === 'cross_entropy' && port === 'pred') return { kind: 'example', width: K };
  if (node.type === 'norm') return { kind: 'param', rows: D, cols: M };
  if (node.type === 'onehot' || node.type === 'lookup') {
    return { kind: 'example', width: 1, category: { column: 'c', count: K } };
  }
  return { kind: 'example', width: D };
}

/** A scalar piece's init bound is 1/sqrt(d) (the emitter prints the number). */
const symInit = (node: Node, line: string): string =>
  node.type === 'scalew' || node.type === 'bias'
    ? line.replace('uniform_(-1.0, 1.0)', 'uniform_(-1/sqrt(d), 1/sqrt(d))')
    : line;

function symbolicParts(key: string): Parts {
  const macro = MACROS[key];
  if (macro) return { shapes: { in: macro.in, out: '1' }, torch: macro.torch };
  const node = symbolicNode(key);
  if (node === null) return {};
  const piece = typed(node);
  const ins: Record<string, Shape> = {};
  const insNames: Record<string, string[]> = {};
  for (const port of piece.inputPorts) {
    ins[port] = symbolicIn(node, port);
    insNames[port] = [port === 'in' ? 'h' : port];
  }
  const ctx: ShapeCtx = {
    dataColumns: D,
    numClasses: K,
    weightShape: () => [D, M],
    treeLeaves: { h: LEAVES },
  };
  const out = piece.outShape(node, ins, ctx);
  const outWidth = OUT_WIDTH[node.type];
  const outText =
    outWidth !== undefined
      ? `n \\times ${outWidth}`
      : 'error' in out
        ? 'n \\times d'
        : fmtShape(out);
  const inText: Record<string, string> = {};
  for (const port of piece.inputPorts) inText[port] = fmtShape(ins[port] as Shape);
  const first = ins[piece.inputPorts[0] ?? ''];
  const params = fmtParams(piece.paramShapes(node, D, first));
  const torch = symInit(
    node,
    wordSyms(
      torchLine(
        node,
        baseCtx({
          ins: insNames,
          categories: K,
          tree: { estimator: 'tree', leaves: 'leaves', columns: [] },
        }),
      ),
    ),
  );
  return {
    shapes: { in: inText, out: outText },
    ...(params !== undefined ? { params } : {}),
    torch,
  };
}

function liveParts(node: Node, analysis: Analysis, data: DataInfo, nodes: readonly Node[]): Parts {
  const piece = typed(node);
  const sym = symbolicParts(paletteKeyOf(node));
  const wiresOf = (port: string) => analysis.inputs[node.id]?.[port] ?? [];
  const optional = optionalPortsOf(node.type);

  // Known widths come from the analysis, then from the node's own settings; the rest are symbols.
  const inShapes: Record<string, Shape> = {};
  const inText: Record<string, string> = {};
  let firstShape: Shape | undefined;
  piece.inputPorts.forEach((port, i) => {
    const from = wiresOf(port)[0]?.from;
    const s = from === undefined ? undefined : analysis.shape[from];
    if (i === 0) firstShape = s;
    if (s !== undefined) {
      inShapes[port] = s;
      inText[port] = fmtShape(s);
    } else if (from === undefined && optional.includes(port)) {
      // an unwired optional port has no shape
    } else {
      inShapes[port] = symbolicIn(node, port);
      inText[port] = sym.shapes?.in[port] ?? 'n \\times d';
    }
  });
  const known = analysis.shape[node.id];
  let out: string;
  if (known !== undefined) out = fmtShape(known);
  else if (OUT_WIDTH[node.type] !== undefined) out = sym.shapes?.out ?? 'n \\times d';
  else {
    const r = piece.outShape(node, inShapes, {
      dataColumns: D,
      numClasses: K,
      weightShape: () => [D, M],
      treeLeaves: { [node.id]: LEAVES },
    });
    out = 'error' in r ? (sym.shapes?.out ?? 'n \\times d') : fmtShape(r);
  }

  const inWidth = analysis.inWidth[node.id] ?? D;
  const first = piece.inputPorts[0];
  const tensors =
    analysis.paramShapes[node.id] ??
    piece.paramShapes(
      node,
      inWidth,
      firstShape ?? (first === undefined ? undefined : symbolicIn(node, first)),
    );
  const params = fmtParams(tensors);

  // The script's own emitter (names, init rule); widths it was not given stay symbols.
  const patched: Analysis = {
    ...analysis,
    inWidth:
      node.type === 'feature' ? analysis.inWidth : { ...analysis.inWidth, [node.id]: inWidth },
  };
  const emitted = nodeTorch(nodes, patched, data, node.id);
  let torch = emitted === null ? sym.torch : wordSyms(emitted.init ?? emitted.expr);
  if (emitted !== null && torch !== undefined) {
    if (node.type === 'lookup') torch = torch.replace('Embedding(0,', 'Embedding(K,');
    if (node.type === 'onehot') torch = torch.replace(', 0).float()', ', K).float()');
    if (analysis.paramInit[node.id] === undefined) torch = symInit(node, torch);
  }
  return {
    shapes: { in: inText, out },
    ...(params !== undefined ? { params } : {}),
    ...(torch !== undefined ? { torch } : {}),
  };
}

// ---------------------------------------------------------------- views

function assemble(key: string, parts: Parts, variant?: string): DocView {
  const doc = PIECE_DOCS[key];
  const v = variant === undefined ? undefined : doc?.variants?.[variant];
  return {
    key,
    label: labelOf(key),
    summary: v?.summary ?? doc?.summary ?? labelOf(key),
    equation: v?.equation ?? doc?.equation ?? '',
    ...(parts.shapes !== undefined
      ? { shapes: { in: { ...parts.shapes.in }, out: parts.shapes.out } }
      : {}),
    ...(parts.params !== undefined ? { params: parts.params } : {}),
    settings: Object.entries(doc?.settings ?? {}).map(([field, text]) => ({ field, text })),
    ...(parts.torch !== undefined ? { torch: parts.torch } : {}),
    notes: [...(doc?.notes ?? [])],
    seeAlso: (doc?.seeAlso ?? []).map((k) => ({ key: k, label: labelOf(k) })),
  };
}

/** The symbolic card of a palette key (d input width, m output width, K categories, n examples). Never throws. */
export function docView(key: string): DocView {
  let parts: Parts = {};
  try {
    parts = symbolicParts(key);
  } catch {
    /* the text still shows */
  }
  return assemble(key, parts);
}

/** The live card of a node on the canvas, from its analysis. Never throws: unknown widths stay symbols. */
export function docViewFor(node: Node, analysis: Analysis, data: DataInfo, graph?: Graph): DocView {
  const key = paletteKeyOf(node);
  const variant = node.type === 'activation' ? node.fn : undefined;
  try {
    return assemble(key, liveParts(node, analysis, data, graph?.nodes ?? [node]), variant);
  } catch {
    try {
      return assemble(key, symbolicParts(key), variant);
    } catch {
      return assemble(key, {}, variant);
    }
  }
}
