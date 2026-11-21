/** modelEquations (CONTRACTS.md section 3, "Equations"). */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  modelEquations,
  type DataInfo,
  type Edge,
  type Graph,
  type HostParams,
  type Node,
} from '../src/index.js';

const info = (d: number): DataInfo => ({
  columns: Array.from({ length: d }, (_, i) => `x${i + 1}`),
  kind: 'regression',
  numClasses: 2,
});

function g(nodes: Node[], edges: Edge[], predictions: string[], loss: string | null): Graph {
  return {
    version: 2,
    nodes,
    edges,
    predictions,
    loss,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
}

const eq = (graph: Graph, d = 1, host?: HostParams) =>
  modelEquations(graph, analyse(graph, info(d)), info(d), host);
const f32 = (...v: number[]) => Float32Array.from(v);
const squaredLoss = (pred: string): { nodes: Node[]; edges: Edge[] } => ({
  nodes: [
    { id: 'y', type: 'target' },
    { id: 'sub', type: 'subtract' },
    { id: 'sq', type: 'square' },
    { id: 'mean', type: 'mean' },
  ],
  edges: [
    [pred, 'sub.a'],
    ['y', 'sub.b'],
    ['sub', 'sq'],
    ['sq', 'mean'],
  ],
});

describe('modelEquations', () => {
  test('linear regression', () => {
    const l = squaredLoss('lin');
    const e = eq(
      g(
        [
          { id: 'in', type: 'input', columns: ['x1', 'x2', 'x3'] },
          { id: 'lin', type: 'linear', out: 1, bias: true },
          ...l.nodes,
        ],
        [['in', 'lin'], ...l.edges],
        ['lin'],
        'mean',
      ),
      3,
    );
    expect(e.lines).toEqual([
      { lhs: '\\hat{y}', rhs: 'W_{1}\\,x + b_{1}', kind: 'pred' },
      { lhs: 'L', rhs: '\\frac{1}{n}\\sum_i \\left(\\hat{y}_{i} - y_{i}\\right)^2', kind: 'loss' },
    ]);
    expect(e.numeric).toBeUndefined();
  });

  test('scalar model with numeric line and folded signs', () => {
    const l = squaredLoss('b');
    const graph = g(
      [
        { id: 'c1', type: 'column', name: 'x1' },
        { id: 'c2', type: 'column', name: 'x2' },
        { id: 'w1', type: 'scalew' },
        { id: 'w2', type: 'scalew' },
        { id: 's', type: 'sum' },
        { id: 'b', type: 'bias' },
        ...l.nodes,
      ],
      [['c1', 'w1'], ['c2', 'w2'], ['w1', 's'], ['w2', 's'], ['s', 'b'], ...l.edges],
      ['b'],
      'mean',
    );
    const host: HostParams = { w1: { W: f32(0.5) }, w2: { W: f32(-1.25) }, b: { b: f32(-0.3) } };
    const e = eq(graph, 2, host);
    expect(e.lines[0]?.rhs).toBe('w_{1}\\,x_{1} + w_{2}\\,x_{2} + b_{1}');
    expect(e.numeric?.[0]?.rhs).toBe('0.50\\,x_{1} - 1.25\\,x_{2} - 0.30');
    expect(e.numeric?.length).toBe(e.lines.length);
  });

  test('unwired bias gives the mean model', () => {
    const l = squaredLoss('b');
    const e = eq(g([{ id: 'b', type: 'bias' }, ...l.nodes], l.edges, ['b'], 'mean'), 1, {
      b: { b: f32(2.5) },
    });
    expect(e.lines[0]).toEqual({ lhs: '\\hat{y}', rhs: 'b_{1}', kind: 'pred' });
    expect(e.numeric?.[0]?.rhs).toBe('2.50');
  });

  test('tanh MLP nests', () => {
    const e = eq(
      g(
        [
          { id: 'in', type: 'input', columns: ['x1'] },
          { id: 'l1', type: 'linear', out: 4, bias: true },
          { id: 'a', type: 'activation', fn: 'tanh' },
          { id: 'l2', type: 'linear', out: 1, bias: true },
        ],
        [
          ['in', 'l1'],
          ['l1', 'a'],
          ['a', 'l2'],
        ],
        ['l2'],
        null,
      ),
    );
    expect(e.lines).toEqual([
      {
        lhs: '\\hat{y}',
        rhs: 'W_{2}\\,\\tanh\\left(W_{1}\\,x + b_{1}\\right) + b_{2}',
        kind: 'pred',
      },
    ]);
  });

  test('skip connection gets a definition line', () => {
    const e = eq(
      g(
        [
          { id: 'in', type: 'input', columns: ['x1'] },
          { id: 'l1', type: 'linear', out: 1, bias: true },
          { id: 'a', type: 'activation', fn: 'relu' },
          { id: 'add', type: 'add' },
        ],
        [
          ['in', 'l1'],
          ['l1', 'a'],
          ['l1', 'add.a'],
          ['a', 'add.b'],
        ],
        ['add'],
        null,
      ),
    );
    expect(e.lines).toEqual([
      { lhs: 'h_{1}', rhs: 'W_{1}\\,x + b_{1}', kind: 'def' },
      { lhs: '\\hat{y}', rhs: 'h_{1} + \\mathrm{ReLU}\\left(h_{1}\\right)', kind: 'pred' },
    ]);
  });

  test('weight penalty shows the norm', () => {
    const e = eq(
      g(
        [
          { id: 'in', type: 'input', columns: ['x1'] },
          { id: 'l1', type: 'linear', out: 1, bias: true },
          { id: 'wt', type: 'weights', of: 'l1' },
          { id: 'n', type: 'norm', kind: 'l2' },
          { id: 'sc', type: 'scale', c: 0.01 },
          { id: 'y', type: 'target' },
          { id: 'sub', type: 'subtract' },
          { id: 'sq', type: 'square' },
          { id: 'mean', type: 'mean' },
          { id: 'tot', type: 'add' },
        ],
        [
          ['in', 'l1'],
          ['l1', 'sub.a'],
          ['y', 'sub.b'],
          ['sub', 'sq'],
          ['sq', 'mean'],
          ['wt', 'n'],
          ['n', 'sc'],
          ['mean', 'tot.a'],
          ['sc', 'tot.b'],
        ],
        ['l1'],
        'tot',
      ),
    );
    const loss = e.lines.find((l) => l.kind === 'loss');
    expect(loss?.rhs).toContain('+ 0.01\\,\\|W_{1}\\|_2^2');
  });

  test('invalid or unpinned graphs omit lines and never throw', () => {
    const l = squaredLoss('lin');
    const dangling = g(
      [{ id: 'lin', type: 'linear', out: 1, bias: true }, ...l.nodes],
      l.edges,
      ['lin', 'nope'],
      'mean',
    );
    expect(() => eq(dangling)).not.toThrow();
    expect(eq(dangling).lines.filter((l) => l.kind === 'pred')).toEqual([]);
    expect(eq(g([], [], [], null)).lines).toEqual([]);
    const cyc = g(
      [
        { id: 'a', type: 'activation', fn: 'tanh' },
        { id: 'b', type: 'activation', fn: 'tanh' },
      ],
      [
        ['a', 'b'],
        ['b', 'a'],
      ],
      ['a'],
      null,
    );
    expect(() => eq(cyc)).not.toThrow();
  });
});
