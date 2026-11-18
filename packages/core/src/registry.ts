/** Piece registry, core half (CONTRACTS.md section 3): ports, defaults, width rules, shapes, torch strings. */
import type { Node, NodeBody, NodeOf, PieceType, Penalty } from './graph.js';

export type { PieceType } from './graph.js';

export type Family =
  'inputs' | 'features' | 'linear' | 'activations' | 'structure' | 'outputs' | 'losses';

export type WidthCtx = { dataColumns: number; numClasses: number };
export type WidthError = { error: string };

export type CorePiece<N extends Node> = {
  /** Drop-to-swap is allowed only within a family. */
  family: Family;
  inputPorts: string[];
  defaults: () => Omit<N, 'id'>;
  /**
   * `inWidths` is keyed by input port. A missing entry means that port is not connected, which
   * gives an error sentence. A loss has no output; its width is 0.
   */
  outWidth(node: N, inWidths: Record<string, number>, ctx: WidthCtx): number | WidthError;
  /** W is [in, out]. */
  paramShapes(node: N, inWidth: number): { W?: [rows: number, cols: number]; b?: [number] };
  /** Torch counterpart for the later code view. `inWidth` is the width entering the node, when known. */
  torch(node: N, inWidth?: number): string;
};

export type PaletteEntry = {
  key: string;
  label: string;
  family: Family | 'penalties';
  make: () => NodeBody | Penalty;
};

const py = {
  bool: (b: boolean) => (b ? 'True' : 'False'),
  float: (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x)),
};

function needInput(label: string, w: number | undefined): number | WidthError {
  return w === undefined ? { error: `${label} needs an input.` } : w;
}

/** Width of the single input `in`, passed through unchanged. */
function passThrough(label: string) {
  return (_node: unknown, inWidths: Record<string, number>): number | WidthError =>
    needInput(label, inWidths['in']);
}

const noParams = () => ({});

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
    outWidth: (node) =>
      node.columns.length === 0
        ? { error: 'Input needs at least one data column switched on.' }
        : node.columns.length,
    paramShapes: noParams,
    torch: (node) => `x[:, [${node.columns.map((c, i) => columnIndex(c, i)).join(', ')}]]`,
  },
  feature: {
    family: 'features',
    inputPorts: ['in'],
    defaults: () => ({ type: 'feature', degree: 2, sin: false, cos: false, products: false }),
    outWidth: (node, inWidths) => {
      const d = needInput('Feature', inWidths['in']);
      return typeof d === 'number' ? featureWidth(d, node) : d;
    },
    paramShapes: noParams,
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
    outWidth: (node, inWidths) => {
      const d = needInput('Linear', inWidths['in']);
      return typeof d === 'number' ? node.out : d;
    },
    paramShapes: (node, inWidth) => ({
      W: [inWidth, node.out],
      ...(node.bias ? { b: [node.out] as [number] } : {}),
    }),
    torch: (node, inWidth) =>
      `nn.Linear(${inWidth ?? '?'}, ${node.out}, bias=${py.bool(node.bias)})`,
  },
  activation: {
    family: 'activations',
    inputPorts: ['in'],
    defaults: () => ({ type: 'activation', fn: 'relu' }),
    outWidth: passThrough('An activation'),
    paramShapes: noParams,
    torch: (node) =>
      ({ sigmoid: 'nn.Sigmoid()', tanh: 'nn.Tanh()', relu: 'nn.ReLU()', gelu: 'nn.GELU()' })[
        node.fn
      ],
  },
  dropout: {
    family: 'structure',
    inputPorts: ['in'],
    defaults: () => ({ type: 'dropout', p: 0.2 }),
    outWidth: passThrough('Dropout'),
    paramShapes: noParams,
    torch: (node) => `nn.Dropout(p=${py.float(node.p)})`,
  },
  add: {
    family: 'structure',
    inputPorts: ['a', 'b'],
    defaults: () => ({ type: 'add' }),
    outWidth: (_node, inWidths) => {
      const a = inWidths['a'];
      const b = inWidths['b'];
      if (a === undefined || b === undefined)
        return { error: 'Add needs both of its inputs connected.' };
      if (a !== b) {
        return { error: `Add needs both inputs to have the same width: ${a} and ${b}.` };
      }
      return a;
    },
    paramShapes: noParams,
    torch: () => 'a + b',
  },
  output: {
    family: 'outputs',
    inputPorts: ['in'],
    defaults: () => ({ type: 'output', link: 'identity' }),
    outWidth: passThrough('The output'),
    paramShapes: noParams,
    torch: (node) =>
      ({ identity: 'nn.Identity()', sigmoid: 'nn.Sigmoid()', softmax: 'nn.Softmax(dim=-1)' })[
        node.link
      ],
  },
  target: {
    family: 'outputs',
    inputPorts: [],
    defaults: () => ({ type: 'target' }),
    // 1 for regression and binary sets; the loss checks the prediction against the classes.
    outWidth: () => 1,
    paramShapes: noParams,
    torch: () => 'y',
  },
  loss: {
    family: 'losses',
    inputPorts: ['pred', 'target'],
    defaults: () => ({ type: 'loss', fn: 'squared_error' }),
    outWidth: () => 0,
    paramShapes: noParams,
    torch: (node) => {
      switch (node.fn) {
        case 'squared_error':
          return 'nn.MSELoss()';
        case 'absolute_error':
          return 'nn.L1Loss()';
        case 'huber':
          return `nn.HuberLoss(delta=${py.float(node.delta ?? 1)})`;
        case 'log_loss':
          return 'nn.BCEWithLogitsLoss()';
        case 'cross_entropy':
          return 'nn.CrossEntropyLoss()';
      }
    },
  },
};

/** The 14 palette entries, one per row of the SPEC pieces table, in SPEC order. */
export const PALETTE: PaletteEntry[] = [
  { key: 'input', label: 'Input', family: 'inputs', make: () => CORE_PIECES.input.defaults() },
  {
    key: 'feature',
    label: 'Feature',
    family: 'features',
    make: () => CORE_PIECES.feature.defaults(),
  },
  { key: 'linear', label: 'Linear', family: 'linear', make: () => CORE_PIECES.linear.defaults() },
  {
    key: 'activation',
    label: 'Activation',
    family: 'activations',
    make: () => CORE_PIECES.activation.defaults(),
  },
  {
    key: 'dropout',
    label: 'Dropout',
    family: 'structure',
    make: () => CORE_PIECES.dropout.defaults(),
  },
  { key: 'add', label: 'Add', family: 'structure', make: () => CORE_PIECES.add.defaults() },
  { key: 'output', label: 'Output', family: 'outputs', make: () => CORE_PIECES.output.defaults() },
  { key: 'target', label: 'Target', family: 'outputs', make: () => CORE_PIECES.target.defaults() },
  {
    key: 'squared_error',
    label: 'Squared error',
    family: 'losses',
    make: () => ({ type: 'loss', fn: 'squared_error' }),
  },
  {
    key: 'absolute_error',
    label: 'Absolute error',
    family: 'losses',
    make: () => ({ type: 'loss', fn: 'absolute_error' }),
  },
  {
    key: 'huber',
    label: 'Huber',
    family: 'losses',
    make: () => ({ type: 'loss', fn: 'huber', delta: 1 }),
  },
  {
    key: 'log_loss',
    label: 'Log loss',
    family: 'losses',
    make: () => ({ type: 'loss', fn: 'log_loss' }),
  },
  {
    key: 'cross_entropy',
    label: 'Cross-entropy',
    family: 'losses',
    make: () => ({ type: 'loss', fn: 'cross_entropy' }),
  },
  {
    key: 'weight_penalty',
    label: 'Weight penalty',
    family: 'penalties',
    make: () => ({ kind: 'l2', lambda: 0.01 }),
  },
];
