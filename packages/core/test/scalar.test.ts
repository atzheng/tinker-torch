/** Scalar pieces (CONTRACTS.md sections 2 to 5): column, scalew, bias and sum. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  CORE_PIECES,
  initParams,
  isOptionalPort,
  reconcileParams,
  insertOnWire,
  withFreshColumn,
  PALETTE,
  swapPiece,
  type DataInfo,
  type Edge,
  type Graph,
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

/** column x_i -> w_i -> sum -> bias -> yhat, squared error against the target. */
function scalarGraph(d: number): Graph {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  for (let i = 1; i <= d; i++) {
    nodes.push({ id: `c${i}`, type: 'column', name: `x${i}` }, { id: `w${i}`, type: 'scalew' });
    edges.push([`c${i}`, `w${i}`], [`w${i}`, 'sum']);
  }
  nodes.push(
    { id: 'sum', type: 'sum' },
    { id: 'b', type: 'bias' },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'sq', type: 'square' },
    { id: 'm', type: 'mean' },
  );
  edges.push(['sum', 'b'], ['b', 'diff.a'], ['y', 'diff.b'], ['diff', 'sq'], ['sq', 'm']);
  return g(nodes, edges, ['b'], 'm');
}

describe('scalar regression', () => {
  test('is trainable with d + 1 parameters', () => {
    for (const d of [1, 3]) {
      const a = analyse(scalarGraph(d), info(d));
      expect(a.issues).toEqual([]);
      expect(a.totalParams).toBe(d + 1);
      expect(a.paramNodes).toHaveLength(d + 1);
      expect(a.paramShapes['w1']).toEqual({ W: [1, 1] });
      expect(a.paramShapes['b']).toEqual({ b: [1] });
    }
  });

  test('init bound is 1/sqrt(d) for weights and bias', () => {
    const a = analyse(scalarGraph(4), info(4));
    expect(a.paramInit['w1']).toEqual({ kind: 'uniform', bound: 0.5 });
    expect(a.paramInit['b']).toEqual({ kind: 'uniform', bound: 0.5 });
    const host = initParams(a, 7);
    for (const v of host['w2']?.['W'] ?? []) expect(Math.abs(v)).toBeLessThanOrEqual(0.5);
  });

  test('the parameter limit counts them', () => {
    const d = 10001;
    const cols = Array.from({ length: d }, (_, i) => `x${i + 1}`);
    const a = analyse(scalarGraph(d), { columns: cols, kind: 'regression', numClasses: 2 });
    expect(a.totalParams).toBe(d + 1);
    expect(a.issues.some((i) => /10,?000/.test(i.message))).toBe(true);
    // Builds and analyses a 20k-node graph (about 3 s alone), so allow for a loaded machine.
  }, 30_000);

  test('a column that is not in the dataset is an issue', () => {
    const a = analyse(scalarGraph(2), info(1));
    expect(a.issues.map((i) => i.message)).toContain(
      'Column uses "x2", which this dataset does not have.',
    );
  });

  test('a loss that depends only on a weight is valid', () => {
    const a = analyse(
      g(
        [
          { id: 'c', type: 'column', name: 'x1' },
          { id: 'w', type: 'scalew' },
          { id: 'm', type: 'mean' },
        ],
        [
          ['c', 'w'],
          ['w', 'm'],
        ],
        [],
        'm',
      ),
      info(1),
    );
    expect(a.issues).toEqual([]);
    expect(a.totalParams).toBe(1);
  });

  test('a sum with no wire needs an input', () => {
    const a = analyse(g([{ id: 's', type: 'sum' }], [], [], null), info(1));
    expect(a.issues.some((i) => i.message === 'A sum needs an input.')).toBe(true);
  });
});

describe('scalar pieces: registry', () => {
  test('ports, variadic flags, palette', () => {
    expect(CORE_PIECES.sum.inputPorts).toEqual(['in']);
    expect(CORE_PIECES.sum.variadicPorts).toEqual(['in']);
    expect(CORE_PIECES.column.inputPorts).toEqual([]);
    expect(CORE_PIECES.column.family).toBe('inputs');
    for (const key of ['column', 'scalew', 'bias', 'sum']) {
      expect(PALETTE.some((p) => p.key === key)).toBe(true);
    }
    expect(PALETTE.find((p) => p.key === 'scalew')?.label).toBe('Weight');
  });

  test('torch strings', () => {
    expect(CORE_PIECES.scalew.torch({ id: 'w', type: 'scalew' })).toBe('w * x');
    expect(CORE_PIECES.bias.torch({ id: 'b', type: 'bias' })).toBe('x + b');
    expect(CORE_PIECES.column.torch({ id: 'c', type: 'column', name: 'x3' })).toBe('x[:, [2]]');
  });

  test('swap rules', () => {
    const graph = scalarGraph(2);
    // Bias has an optional input and Weight does not, so they no longer swap
    expect(swapPiece(graph, 'w1', { type: 'bias' }).ok).toBe(false);
    expect(swapPiece(graph, 'c1', { type: 'target' }).ok).toBe(false);
    expect(swapPiece(graph, 'c1', { type: 'input', columns: ['x1'] }).ok).toBe(false);
    expect(swapPiece(graph, 'y', { type: 'column', name: 'x1' }).ok).toBe(false);
    expect(swapPiece(graph, 'sum', { type: 'add' }).ok).toBe(false);
    expect(swapPiece(graph, 'sum', { type: 'scalew' }).ok).toBe(false);
    expect(swapPiece(graph, 'w1', { type: 'linear', out: 2, bias: true }).ok).toBe(false);
  });

  test('new weights get w1.. ids', () => {
    const graph = g(
      [
        { id: 'a', type: 'target' },
        { id: 'b', type: 'mean' },
      ],
      [['a', 'b']],
      [],
      null,
    );
    const r = insertOnWire(graph, 0, { type: 'scalew' });
    expect(r.ok && r.select).toBe('w1');
    if (!r.ok) return;
    const r2 = insertOnWire(r.graph, 0, { type: 'scalew' });
    expect(r2.ok && r2.select).toBe('w2');
  });
});

describe('new Column defaults', () => {
  const plane = (...a: Parameters<typeof g>): Graph => {
    const gr = g(...a);
    return { ...gr, data: { ...gr.data, set: 'plane' } };
  };
  test('takes the first dataset column not used by another Column', () => {
    const graph = plane([{ id: 'x1', type: 'column', name: 'x1' }], [], [], null);
    expect(withFreshColumn(graph, { type: 'column', name: 'x1' })).toEqual({
      type: 'column',
      name: 'x2',
    });
    const both = plane(
      [
        { id: 'x1', type: 'column', name: 'x1' },
        { id: 'x2', type: 'column', name: 'x2' },
      ],
      [],
      [],
      null,
    );
    // the 'plane' set has two columns, so all are used and it falls back to x1
    expect(withFreshColumn(both, { type: 'column', name: 'x2' })).toEqual({
      type: 'column',
      name: 'x1',
    });
    expect(withFreshColumn(graph, { type: 'scalew' })).toEqual({ type: 'scalew' });
  });
});

describe('column ids', () => {
  test('palette-inserted columns get x1, x2, ...', () => {
    const x = (id: string): Node => ({ id, type: 'column', name: 'x1' });
    const graph = g([x('x1'), { id: 'm', type: 'mean' }], [['x1', 'm']], [], null);
    // a Column has no input port, so it cannot go on a wire; ids come from the same allocator
    const r = insertOnWire(graph, 0, { type: 'scalew' });
    expect(r.ok && r.select).toBe('w1');
  });
});

describe('optional input (tinker-7ab): an unwired Bias is the mean as a weight', () => {
  const intercept = (): Graph =>
    g(
      [
        { id: 'b', type: 'bias' },
        { id: 'y', type: 'target' },
        { id: 'diff', type: 'subtract' },
        { id: 'sq', type: 'square' },
        { id: 'm', type: 'mean' },
      ],
      [
        ['b', 'diff.a'],
        ['y', 'diff.b'],
        ['diff', 'sq'],
        ['sq', 'm'],
      ],
      ['b'],
      'm',
    );

  test('has no issues, one parameter and can train', () => {
    const a = analyse(intercept(), info(1));
    expect(a.issues).toEqual([]);
    expect(a.totalParams).toBe(1);
    expect(a.trainable).toBe(true);
    expect(a.paramShapes['b']).toEqual({ b: [1] });
    expect(a.shape['b']).toEqual({ kind: 'example', width: 1 });
  });

  test('torch strings and registry flags', () => {
    expect(CORE_PIECES.bias.optionalPorts).toEqual(['in']);
    expect(isOptionalPort('bias', 'in')).toBe(true);
    expect(isOptionalPort('scalew', 'in')).toBe(false);
    expect(CORE_PIECES.bias.torch({ id: 'b', type: 'bias' }, 0)).toBe('b.expand(n, 1)');
    expect(CORE_PIECES.bias.torch({ id: 'b', type: 'bias' }, 1)).toBe('x + b');
  });

  test('wiring and unwiring keep b (reconcileParams)', () => {
    const g1 = intercept();
    const a1 = analyse(g1, info(1));
    const h1 = initParams(a1, 1);
    const hb = h1['b']?.b;
    if (hb) hb[0] = 7;
    const g2: Graph = {
      ...g1,
      nodes: [...g1.nodes, { id: 'c', type: 'column', name: 'x1' }],
      edges: [...g1.edges, ['c', 'b']],
    };
    const a2 = analyse(g2, info(1));
    expect(a2.issues).toEqual([]);
    const h2 = reconcileParams(h1, a1, a2, 1);
    expect(h2['b']?.b?.[0]).toBe(7);
    const h3 = reconcileParams(h2, a2, a1, 1);
    expect(h3['b']?.b?.[0]).toBe(7);
  });

  test('a wired Bias still rejects scalar inputs; other pieces still need their input', () => {
    const g1 = intercept();
    const bad: Graph = { ...g1, edges: [...g1.edges, ['m', 'b']] };
    expect(analyse(bad, info(1)).issues.length).toBeGreaterThan(0);
    const dangling: Graph = {
      ...g1,
      nodes: [...g1.nodes, { id: 'w', type: 'scalew' }],
      edges: [...g1.edges, ['w', 'diff.a']],
    };
    expect(analyse(dangling, info(1)).issues.length).toBeGreaterThan(0);
  });
});
