/** Group into Linear / Expand to scalars (core/scalar.ts). The loss check is in the engine package. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  expandToScalars,
  findLinearGroups,
  groupIntoLinear,
  parseGraph,
  type DataInfo,
  type Edge,
  type Graph,
  type HostParams,
  type Node,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 };

function mk(nodes: Node[], edges: Edge[], extra: Partial<Graph> = {}): Graph {
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: ['b'],
    loss: 'm',
    data: { set: 'plane', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
    ...extra,
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const tail: Node[] = [
  { id: 'y', type: 'target' },
  { id: 'diff', type: 'subtract' },
  { id: 'sq', type: 'square' },
  { id: 'm', type: 'mean' },
];
const rest: Edge[] = [
  ['y', 'diff.b'],
  ['diff', 'sq'],
  ['sq', 'm'],
];

/** Columns named `names`, Weights w1.., Sum, Bias b (unless `withBias` is false): scalar regression. */
function scalar(names: string[], withBias = true): Graph {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  names.forEach((nm, i) => {
    nodes.push({ id: `c${i + 1}`, type: 'column', name: nm }, { id: `w${i + 1}`, type: 'scalew' });
    edges.push([`c${i + 1}`, `w${i + 1}`], [`w${i + 1}`, 'sum']);
  });
  nodes.push({ id: 'sum', type: 'sum' });
  if (!withBias) {
    return mk([...nodes, ...tail], [...edges, ['sum', 'diff.a'], ...rest], {
      predictions: ['sum'],
    });
  }
  nodes.push({ id: 'b', type: 'bias' });
  return mk([...nodes, ...tail], [...edges, ['sum', 'b'], ['b', 'diff.a'], ...rest]);
}

const host = (n: number): HostParams => {
  const h = Object.create(null) as HostParams;
  for (let i = 1; i <= n; i++) h[`w${i}`] = { W: Float32Array.of(10 * i + 0.25) };
  h['b'] = { b: Float32Array.of(-0.5) };
  return h;
};

const an = (g: Graph) => analyse(g, info);
const nums = (a: Float32Array | undefined) => Array.from(a ?? [], (v) => v);
const IDS = ['c1', 'w1', 'c2', 'w2', 'sum', 'b'];

describe('findLinearGroups', () => {
  test('finds the scalar regression pattern', () => {
    const g = scalar(['x1', 'x2']);
    expect(findLinearGroups(g, an(g))).toEqual([IDS]);
  });
  test('without a Bias the group is Columns, Weights and Sum', () => {
    const g = scalar(['x1', 'x2'], false);
    expect(findLinearGroups(g, an(g))).toEqual([['c1', 'w1', 'c2', 'w2', 'sum']]);
  });
  test('no group when a Weight feeds something else too', () => {
    const g = scalar(['x1', 'x2']);
    const h = mk([...g.nodes, { id: 'extra', type: 'square' }], [...g.edges, ['w1', 'extra']]);
    expect(findLinearGroups(h, an(h))).toEqual([]);
  });
  test('no group when a Column name is not a dataset column or is repeated', () => {
    const a = scalar(['x1', 'x3']);
    expect(findLinearGroups(a, an(a))).toEqual([]);
    const b = scalar(['x1', 'x1']);
    expect(findLinearGroups(b, an(b))).toEqual([]);
  });
  test('a Bias that is not the only consumer of the Sum is left out', () => {
    const g = scalar(['x1', 'x2']);
    const h = mk([...g.nodes, { id: 'extra', type: 'square' }], [...g.edges, ['sum', 'extra']]);
    expect(findLinearGroups(h, an(h))).toEqual([['c1', 'w1', 'c2', 'w2', 'sum']]);
  });
  test('a Sum fed by something that is not a Weight is not a group', () => {
    const g = mk(
      [
        { id: 'c', type: 'column', name: 'x1' },
        { id: 's', type: 'sum' },
      ],
      [['c', 's']],
      { predictions: [], loss: null },
    );
    expect(findLinearGroups(g, an(g))).toEqual([]);
  });
});

describe('groupIntoLinear', () => {
  test('builds Input -> Linear with W and b copied, consumers rewired, pin moved', () => {
    const g = scalar(['x1', 'x2']);
    const h = host(2);
    const r = groupIntoLinear(g, h, IDS);
    if (!r.ok) throw new Error(r.error);
    expect(r.select).toBe('linear1');
    expect(r.graph.nodes.map((n) => n.id)).toEqual(['y', 'diff', 'sq', 'm', 'input1', 'linear1']);
    expect(r.graph.nodes.find((n) => n.id === 'input1')).toEqual({
      id: 'input1',
      type: 'input',
      columns: ['x1', 'x2'],
    });
    expect(r.graph.edges).toContainEqual(['linear1', 'diff.a']);
    expect(r.graph.edges).toContainEqual(['input1', 'linear1']);
    expect(r.graph.predictions).toEqual(['linear1']);
    expect(nums(r.host['linear1']?.['W'])).toEqual([10.25, 20.25]);
    expect(nums(r.host['linear1']?.['b'])).toEqual([-0.5]);
    expect(Object.keys(r.host)).toEqual(['linear1']);
    const a = an(r.graph);
    expect(a.issues).toEqual([]);
    expect(a.paramShapes['linear1']).toEqual({ W: [2, 1], b: [1] });
    expect(h['w1']).toBeDefined(); // the input host is not touched
  });

  test('columns wired in non-dataset order: weights follow their columns', () => {
    const g = scalar(['x2', 'x1']); // w1 reads x2 (10.25), w2 reads x1 (20.25)
    const r = groupIntoLinear(g, host(2), IDS);
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.nodes.find((n) => n.id === 'input1')).toMatchObject({ columns: ['x1', 'x2'] });
    expect(nums(r.host['linear1']?.['W'])).toEqual([20.25, 10.25]);
  });

  test('no Bias gives a Linear without bias', () => {
    const g = scalar(['x1', 'x2'], false);
    const r = groupIntoLinear(g, host(2), ['c1', 'w1', 'c2', 'w2', 'sum']);
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.nodes.find((n) => n.id === 'linear1')).toMatchObject({ out: 1, bias: false });
    expect(r.host['linear1']?.['b']).toBeUndefined();
    expect(r.graph.predictions).toEqual(['linear1']);
  });

  test('a Column with another consumer stays', () => {
    const g = scalar(['x1', 'x2']);
    const h = mk([...g.nodes, { id: 'extra', type: 'square' }], [...g.edges, ['c1', 'extra']]);
    const r = groupIntoLinear(h, host(2), IDS);
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.nodes.some((n) => n.id === 'c1')).toBe(true);
    expect(r.graph.nodes.some((n) => n.id === 'c2')).toBe(false);
    expect(r.graph.edges).toContainEqual(['c1', 'extra']);
  });

  test('a locked Column that stays does not block', () => {
    const g = scalar(['x1', 'x2']);
    const h = mk([...g.nodes, { id: 'extra', type: 'square' }], [...g.edges, ['c1', 'extra']]);
    expect(groupIntoLinear(h, host(2), IDS, { locked: ['c1'] }).ok).toBe(true);
  });

  const g0 = scalar(['x1', 'x2']);
  const refusals: [string, () => ReturnType<typeof groupIntoLinear>][] = [
    ['ids that are not the whole group', () => groupIntoLinear(g0, host(2), ['w1', 'sum'])],
    ['a pin on a Weight', () => groupIntoLinear({ ...g0, predictions: ['w1'] }, host(2), IDS)],
    [
      'a pin on the Sum when a Bias follows',
      () => groupIntoLinear({ ...g0, loss: 'sum' }, host(2), IDS),
    ],
    ['a locked piece', () => groupIntoLinear(g0, host(2), IDS, { locked: ['w2'] })],
    ['a locked Column that goes', () => groupIntoLinear(g0, host(2), IDS, { locked: ['c1'] })],
    ['missing values', () => groupIntoLinear(g0, Object.create(null) as HostParams, IDS)],
  ];
  test.each(refusals)('refuses %s with one sentence', (_n, run) => {
    const r = run();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/^[^.]+\.$/);
  });
});

describe('expandToScalars', () => {
  function linearGraph(opts: { out?: number; bias?: boolean; from?: Node } = {}): Graph {
    return mk(
      [
        opts.from ?? { id: 'x', type: 'input', columns: ['x2', 'x1'] },
        { id: 'lin', type: 'linear', out: opts.out ?? 1, bias: opts.bias ?? true },
        ...tail,
      ],
      [['x', 'lin'], ['lin', 'diff.a'], ...rest],
      { predictions: ['lin'] },
    );
  }
  const lh = (): HostParams => {
    const h = Object.create(null) as HostParams;
    h['lin'] = { W: Float32Array.of(1.5, -2.5), b: Float32Array.of(0.125) };
    return h;
  };

  test('makes Columns, Weights, Sum and Bias in the Input column order', () => {
    const r = expandToScalars(linearGraph(), lh(), 'lin');
    if (!r.ok) throw new Error(r.error);
    const g = r.graph;
    expect(g.nodes.map((n) => `${n.id}:${n.type}`)).toEqual([
      'y:target',
      'diff:subtract',
      'sq:square',
      'm:mean',
      'x1:column',
      'x2:column',
      'w1:scalew',
      'w2:scalew',
      'sum1:sum',
      'bias1:bias',
    ]);
    expect(g.nodes.find((n) => n.id === 'x1')).toMatchObject({ name: 'x2' });
    expect(g.predictions).toEqual(['bias1']);
    expect(g.edges).toContainEqual(['bias1', 'diff.a']);
    expect(nums(r.host['w1']?.['W'])).toEqual([1.5]);
    expect(nums(r.host['w2']?.['W'])).toEqual([-2.5]);
    expect(nums(r.host['bias1']?.['b'])).toEqual([0.125]);
    expect(r.host['lin']).toBeUndefined();
    expect(r.select).toBe('bias1');
    const a = an(g);
    expect(a.issues).toEqual([]);
    expect(a.totalParams).toBe(3);
  });

  test('a Linear without bias gives no Bias and the Sum takes the pin', () => {
    const h = Object.create(null) as HostParams;
    h['lin'] = { W: Float32Array.of(1, 2) };
    const r = expandToScalars(linearGraph({ bias: false }), h, 'lin');
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.nodes.some((n) => n.type === 'bias')).toBe(false);
    expect(r.graph.predictions).toEqual(['sum1']);
  });

  test('keeps the Input when something else reads it; fresh ids avoid existing ones', () => {
    const g = linearGraph();
    const h = mk(
      [
        ...g.nodes,
        { id: 'x1', type: 'square' },
        { id: 'other', type: 'linear', out: 2, bias: true },
      ],
      [...g.edges, ['x', 'other']],
      { predictions: ['lin'] },
    );
    const r = expandToScalars(h, lh(), 'lin');
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.nodes.some((n) => n.id === 'x')).toBe(true);
    expect(r.graph.nodes.find((n) => n.id === 'x1')).toMatchObject({ type: 'square' });
    expect(r.graph.nodes.filter((n) => n.type === 'column').map((n) => n.id)).toEqual(['x2', 'x3']);
  });

  const feature: Node = {
    id: 'x',
    type: 'feature',
    degree: 1,
    sin: false,
    cos: false,
    products: false,
  };
  const cases: [string, () => ReturnType<typeof expandToScalars>][] = [
    ['out > 1', () => expandToScalars(linearGraph({ out: 2 }), lh(), 'lin')],
    [
      'a feeder that is not an Input',
      () => expandToScalars(linearGraph({ from: feature }), lh(), 'lin'),
    ],
    ['a locked Linear', () => expandToScalars(linearGraph(), lh(), 'lin', { locked: ['lin'] })],
    [
      'a locked Input that would go',
      () => expandToScalars(linearGraph(), lh(), 'lin', { locked: ['x'] }),
    ],
    ['a missing piece', () => expandToScalars(linearGraph(), lh(), 'nope')],
    [
      'missing values',
      () => expandToScalars(linearGraph(), Object.create(null) as HostParams, 'lin'),
    ],
  ];
  test.each(cases)('refuses %s with one sentence', (_n, run) => {
    const r = run();
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/^[^.]+\.$/);
  });
});

describe('round trip', () => {
  test('group, expand, group gives identical params (bitwise), graph and shapes', () => {
    const g0 = scalar(['x2', 'x1']);
    const h0 = host(2);
    h0['w1'] = { W: Float32Array.of(0.1) }; // not exactly representable: compare bits
    h0['b'] = { b: Float32Array.of(1 / 3) };
    const a = groupIntoLinear(g0, h0, IDS);
    if (!a.ok) throw new Error(a.error);
    const e = expandToScalars(a.graph, a.host, 'linear1');
    if (!e.ok) throw new Error(e.error);
    const found = findLinearGroups(e.graph, an(e.graph));
    expect(found).toHaveLength(1);
    const b = groupIntoLinear(e.graph, e.host, found[0] ?? []);
    if (!b.ok) throw new Error(b.error);
    expect(nums(b.host['linear1']?.['W'])).toEqual(nums(a.host['linear1']?.['W']));
    expect(nums(b.host['linear1']?.['b'])).toEqual(nums(a.host['linear1']?.['b']));
    expect(b.graph).toEqual(a.graph);
    expect(an(e.graph).totalParams).toBe(an(g0).totalParams);
    expect(an(e.graph).shape['bias1']).toEqual(an(g0).shape['b']);
  });
});

describe('positions', () => {
  test('removed pieces lose theirs, the rest keep theirs, new pieces get none', () => {
    const pos: Record<string, [number, number]> = { w1: [1, 2], m: [3, 4] };
    const a = groupIntoLinear({ ...scalar(['x1', 'x2']), positions: pos }, host(2), IDS);
    if (!a.ok) throw new Error(a.error);
    expect(a.graph.positions).toEqual({ m: [3, 4] });
    const lh = Object.create(null) as HostParams;
    lh['linear1'] = a.host['linear1'] ?? {};
    const withPos = { ...a.graph, positions: { linear1: [5, 6], m: [3, 4] } as typeof pos };
    const e = expandToScalars(withPos, lh, 'linear1');
    if (!e.ok) throw new Error(e.error);
    expect(e.graph.positions).toEqual({ m: [3, 4] });
  });
});
