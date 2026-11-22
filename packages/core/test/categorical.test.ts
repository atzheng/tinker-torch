import { describe, expect, it } from 'vitest';
import {
  analyse,
  buildCsv,
  initParams,
  inputKind,
  parseCsv,
  parseGraph,
  registerCsv,
  sample,
  usableInputs,
  type DataConfig,
  type Graph,
} from '../src/index.js';

const cfg = (set: DataConfig['set'], over: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 100,
  noise: 0.3,
  val: 0.3,
  seed: 2,
  standardize: false,
  ...over,
});

const ZONE = sample(cfg('delivery_late')).info;

function graph(nodes: object[], edges: [string, string][], pred: string, set = 'delivery_late') {
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: [pred],
    loss: 'loss',
    data: cfg(set as DataConfig['set']),
    train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const tail = [
  { id: 'y', type: 'target' },
  { id: 'err', type: 'log_loss' },
  { id: 'loss', type: 'mean' },
];
const tailEdges: [string, string][] = [
  ['p', 'err.pred'],
  ['y', 'err.target'],
  ['err', 'loss'],
];

function issues(g: Graph): string[] {
  return analyse(g, ZONE).issues.map((i) => i.message);
}

describe('categorical columns in analyse', () => {
  it('a Lookup after a category Input is valid and owns [K, width]', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x4'] },
        { id: 'look', type: 'lookup', width: 1 },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'look'], ['look', 'p'], ...tailEdges],
      'p',
    );
    const a = analyse(g, ZONE);
    expect(a.issues).toEqual([]);
    expect(a.paramShapes['look']).toEqual({ W: [4, 1] });
    expect(a.totalParams).toBe(4);
    expect(a.shape['look']).toEqual({ kind: 'example', width: 1 });
    expect(initParams(a, 1)['look']?.['W']).toHaveLength(4);
  });

  it('One-hot gives K columns and feeds a Linear', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x4'] },
        { id: 'oh', type: 'onehot' },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'oh'], ['oh', 'lin'], ['lin', 'p'], ...tailEdges],
      'p',
    );
    const a = analyse(g, ZONE);
    expect(a.issues).toEqual([]);
    expect(a.outWidth['oh']).toBe(4);
    expect(a.paramShapes['lin']).toEqual({ W: [4, 1], b: [1] });
  });

  it('a category column feeding a numeric piece is one plain sentence at that piece', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x4'] },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'lin'], ['lin', 'p'], ...tailEdges],
      'p',
    );
    const a = analyse(g, ZONE);
    expect(a.issues).toHaveLength(1);
    expect(a.issues[0]?.message).toBe(
      '"lin" needs numbers, but the category "x4" is a category; send it through One-hot or Lookup first.',
    );
    expect(a.issues[0]?.at).toEqual({ node: 'lin', edge: 0 });
  });

  it('numbers feeding One-hot or Lookup are refused', () => {
    for (const type of ['onehot', 'lookup']) {
      const g = graph(
        [
          { id: 'n', type: 'input', columns: ['x1'] },
          { id: 'c', type, ...(type === 'lookup' ? { width: 1 } : {}) },
          { id: 'lin', type: 'linear', out: 1, bias: true },
          { id: 'p', type: 'link', fn: 'sigmoid' },
          ...tail,
        ],
        [['n', 'c'], ['c', 'lin'], ['lin', 'p'], ...tailEdges],
        'p',
      );
      expect(issues(g).join(' ')).toMatch(
        /needs a category column, but this input is 1 per example/,
      );
    }
  });

  it('an Input cannot mix a category with numeric columns', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x1', 'x4'] },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'lin'], ['lin', 'p'], ...tailEdges],
      'p',
    );
    expect(issues(g)).toEqual([
      'Input mixes the category column "x4" with other columns; give the category its own Input.',
    ]);
  });

  it('a category cannot be pinned as a prediction or added to a number', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x4'] },
        { id: 'n', type: 'input', columns: ['x1'] },
        { id: 'add', type: 'add' },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'add.a'], ['n', 'add.b'], ['add', 'p'], ...tailEdges],
      'z',
    );
    const msgs = issues(g);
    expect(msgs.some((m) => /"add" needs numbers/.test(m))).toBe(true);
    expect(msgs.some((m) => /A prediction must be a number, but "z" is a category/.test(m))).toBe(
      true,
    );
  });

  it('a dataset with no categories treats every column as numbers (the same graph is fine)', () => {
    const g = graph(
      [
        { id: 'z', type: 'input', columns: ['x4'] },
        { id: 'lin', type: 'linear', out: 1, bias: true },
        { id: 'p', type: 'link', fn: 'sigmoid' },
        ...tail,
      ],
      [['z', 'lin'], ['lin', 'p'], ...tailEdges],
      'p',
    );
    const { categories: _c, columnKinds: _k, ...numeric } = ZONE;
    expect(analyse(g, numeric).issues).toEqual([]);
  });
});

const CATS = ['red', 'green', 'blue'];
const csvText = [
  'size,colour,label,id',
  ...Array.from({ length: 40 }, (_, i) => `${i % 5},${CATS[(i * 7) % 3]},${i % 2},r${i}`),
].join('\n');

describe('CSV text input columns', () => {
  it('keeps a text column with few values as a category, sorted over the whole file', () => {
    const p = parseCsv(csvText);
    if (!p.ok) throw new Error(p.error);
    const { csv } = p;
    expect(inputKind(csv, 0)).toBe('numeric');
    expect(inputKind(csv, 1)).toBe('categorical');
    expect(inputKind(csv, 3)).toBe(null); // 40 distinct ids
    expect(usableInputs(csv, 2)).toEqual([0, 1]);
    const b = buildCsv(csv, { target: 2, inputs: [0, 1, 3], kind: 'binary' });
    if (!b.ok) throw new Error(b.error);
    expect(b.notes[0]).toMatch(/Left out 1 text column with too many distinct values: id/);
    expect(b.spec.categories).toEqual({ 1: ['blue', 'green', 'red'] });
    registerCsv(b);
    // every row (so train and validation alike) holds an index that exists in the list
    for (const seed of [1, 2, 3]) {
      const ds = sample(cfg('csv', { seed, val: 0.5 }));
      expect(ds.info.columnKinds).toEqual(['numeric', 'categorical']);
      expect(ds.info.categories).toEqual({ x2: ['blue', 'green', 'red'] });
      for (const xs of [ds.xTrain, ds.xVal]) {
        for (let i = 0; i < xs.length / 2; i++) {
          const v = xs[i * 2 + 1] ?? NaN;
          expect(Number.isInteger(v) && v >= 0 && v < 3).toBe(true);
        }
      }
      // Standardise leaves the category index alone
      const st = sample(cfg('csv', { seed, standardize: true }));
      expect(st.standardization?.std[1]).toBe(1);
      expect(
        Array.from(st.xTrain)
          .filter((_, i) => i % 2 === 1)
          .every(Number.isInteger),
      ).toBe(true);
    }
  });
});
