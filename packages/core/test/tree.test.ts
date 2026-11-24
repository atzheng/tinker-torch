import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  analyse,
  fitTree,
  fitTrees,
  parseGraph,
  sample,
  serializeGraph,
  treeLeafCounts,
  treeLeafIndex,
  treeOneHot,
  modelEquations,
  CORE_PIECES,
  type DataConfig,
  type TreeFit,
  type TreeSettings,
} from '../src/index.js';

type Case = {
  name: string;
  kind: 'regression' | 'binary' | 'multiclass';
  numClasses: number;
  n: number;
  d: number;
  x: number[];
  y: number[];
  settings: TreeSettings;
  partitionStable: boolean;
  splitsStable: boolean;
  expect: { leaves: number; leaf: number[]; splits: [number, number][] };
};
const fixtures = JSON.parse(
  readFileSync(new URL('./fixtures/tree_sklearn.json', import.meta.url), 'utf8'),
) as { sklearn: string; cases: Case[] };

const S: TreeSettings = {
  criterion: 'squared_error',
  depth: 3,
  minSplit: 2,
  minLeaf: 1,
  minDecrease: 0,
};
const splitsOf = (f: TreeFit): [number, number][] =>
  f.nodes.flatMap((nd) =>
    nd.kind === 'split' ? [[nd.feature, nd.threshold] as [number, number]] : [],
  );
const REG = { numClasses: 1 };

/** Brute-force impurity of a set of targets (independent of core/tree.ts). */
function impurityOf(criterion: TreeSettings['criterion'], v: number[]): number {
  const m = v.length;
  if (criterion === 'squared_error') {
    const mean = v.reduce((a, b) => a + b, 0) / m;
    return v.reduce((a, b) => a + (b - mean) ** 2, 0) / m;
  }
  if (criterion === 'absolute_error') {
    const t = [...v].sort((a, b) => a - b);
    const med =
      m % 2 ? (t[(m - 1) / 2] as number) : ((t[m / 2 - 1] as number) + (t[m / 2] as number)) / 2;
    return v.reduce((a, b) => a + Math.abs(b - med), 0) / m;
  }
  const counts = new Map<number, number>();
  for (const y of v) counts.set(y, (counts.get(y) ?? 0) + 1);
  let s = 0;
  for (const c of counts.values()) {
    const q = c / m;
    s += criterion === 'gini' ? -q * q : -q * Math.log2(q);
  }
  return criterion === 'gini' ? 1 + s : s;
}

/** Every split is the brute-force best at its node; every leaf is justified by a stopping rule. */
function expectOptimal(c: Case, fit: TreeFit): void {
  const { settings: st, x, y, n, d } = c;
  const rec = (id: number, rows: number[], depth: number): void => {
    const nd = fit.nodes[id] as TreeFit['nodes'][number];
    const m = rows.length;
    const imp = impurityOf(
      st.criterion,
      rows.map((r) => y[r] as number),
    );
    // Best gain over every candidate (feature, threshold) that keeps minLeaf on each side.
    let best = 0;
    for (let f = 0; f < d; f++) {
      const order = [...rows].sort((a, b) => (x[a * d + f] as number) - (x[b * d + f] as number));
      for (let p = 1; p < m; p++) {
        const lo = x[(order[p - 1] as number) * d + f] as number;
        const hi = x[(order[p] as number) * d + f] as number;
        if (hi <= lo + 1e-7 || p < st.minLeaf || m - p < st.minLeaf) continue;
        const g =
          m * imp -
          p *
            impurityOf(
              st.criterion,
              order.slice(0, p).map((r) => y[r] as number),
            ) -
          (m - p) *
            impurityOf(
              st.criterion,
              order.slice(p).map((r) => y[r] as number),
            );
        best = Math.max(best, g);
      }
    }
    const tol = 1e-9 * Math.max(m * imp, 1e-12);
    if (nd.kind === 'split') {
      expect(depth).toBeLessThan(st.depth);
      expect(m).toBeGreaterThanOrEqual(st.minSplit);
      const lr = rows.filter((r) => (x[r * d + nd.feature] as number) <= nd.threshold);
      const rr = rows.filter((r) => !((x[r * d + nd.feature] as number) <= nd.threshold));
      expect(Math.min(lr.length, rr.length)).toBeGreaterThanOrEqual(st.minLeaf);
      const g =
        m * imp -
        lr.length *
          impurityOf(
            st.criterion,
            lr.map((r) => y[r] as number),
          ) -
        rr.length *
          impurityOf(
            st.criterion,
            rr.map((r) => y[r] as number),
          );
      expect(Math.abs(g - best)).toBeLessThanOrEqual(tol);
      expect(g / n).toBeGreaterThanOrEqual(st.minDecrease - 1e-12);
      rec(nd.left, lr, depth + 1);
      rec(nd.right, rr, depth + 1);
    } else {
      const reason =
        depth >= st.depth ||
        m < st.minSplit ||
        m < 2 * st.minLeaf ||
        imp <= 1e-12 ||
        best <= tol ||
        best / n < st.minDecrease + 1e-12;
      expect(reason).toBe(true);
    }
  };
  rec(
    0,
    Array.from({ length: n }, (_, i) => i),
    0,
  );
}

describe('fitTree against scikit-learn fixtures', () => {
  it('has cases for every criterion', () => {
    const crits = new Set(fixtures.cases.map((c) => c.settings.criterion));
    expect([...crits].sort()).toEqual(['absolute_error', 'entropy', 'gini', 'squared_error']);
    expect(fixtures.cases.length).toBeGreaterThan(40);
  });
  for (const c of fixtures.cases) {
    it(c.name, () => {
      const fit = fitTree(c.x, c.n, c.d, c.y, c.settings, { numClasses: c.numClasses });
      const idx = [...treeLeafIndex(fit, c.x, c.n, c.d)];
      const got = splitsOf(fit);
      const [rf, rt] = c.expect.splits[0] as [number, number];
      // Whatever the ties, the root split is unambiguous in these data.
      expect(got[0]?.[0]).toBe(rf);
      expect(got[0]?.[1]).toBeCloseTo(rt, 9);
      expectOptimal(c, fit);
      if (!c.partitionStable) {
        // Equally good splits that cut the rows differently: sklearn's own answer depends on its
        // random feature order, so only the invariants are checked.
        expect(fit.leaves).toBeLessThanOrEqual(2 ** c.settings.depth);
        const sizes = new Map<number, number>();
        for (const l of idx) sizes.set(l, (sizes.get(l) ?? 0) + 1);
        expect(Math.min(...sizes.values())).toBeGreaterThanOrEqual(c.settings.minLeaf);
        return;
      }
      // The partition of the rows into leaves is the same (leaf ids may be numbered differently).
      expect(fit.leaves).toBe(c.expect.leaves);
      const map = new Map<number, number>();
      idx.forEach((l, i) => {
        const want = c.expect.leaf[i] as number;
        expect(map.get(l) ?? want).toBe(want);
        map.set(l, want);
      });
      expect(new Set(map.values()).size).toBe(c.expect.leaves);
      if (!c.splitsStable) return; // a tie between two features (e.g. a two-row node) picks either
      expect(idx).toEqual(c.expect.leaf);
      expect(got.length).toBe(c.expect.splits.length);
      got.forEach(([f, t], i) => {
        expect(f).toBe((c.expect.splits[i] as [number, number])[0]);
        expect(t).toBeCloseTo((c.expect.splits[i] as [number, number])[1], 9);
      });
    });
  }
  it('most cases are tie-free (exact splits compared)', () => {
    expect(fixtures.cases.filter((c) => c.splitsStable).length).toBeGreaterThanOrEqual(50);
    for (const crit of ['squared_error', 'absolute_error', 'gini', 'entropy']) {
      const n = fixtures.cases.filter(
        (c) => c.settings.criterion === crit && c.splitsStable,
      ).length;
      expect(n).toBeGreaterThanOrEqual(3);
    }
  });
});

describe('fitTree rules', () => {
  it('splits at the midpoint between distinct values and sends x <= t left', () => {
    const fit = fitTree([0, 1, 2, 3], 4, 1, [0, 0, 10, 10], { ...S, depth: 1 }, REG);
    expect(splitsOf(fit)).toEqual([[0, 1.5]]);
    expect([...treeLeafIndex(fit, [1.5, 1.5001, -9, 9], 4, 1)]).toEqual([0, 1, 0, 1]);
  });

  it('ignores duplicate x values: no threshold falls between equal values', () => {
    const x = [1, 1, 1, 2, 2, 3];
    const y = [0, 5, 0, 5, 0, 9];
    const fit = fitTree(x, 6, 1, y, { ...S, depth: 8 }, REG);
    const idx = treeLeafIndex(fit, x, 6, 1);
    // equal x always share a leaf
    expect(idx[0]).toBe(idx[1]);
    expect(idx[1]).toBe(idx[2]);
    expect(idx[3]).toBe(idx[4]);
    for (const [, t] of splitsOf(fit)) expect([1.5, 2.5]).toContain(t);
  });

  it('is a single leaf for pure targets, one row, or constant x', () => {
    expect(fitTree([0, 1, 2], 3, 1, [4, 4, 4], S, REG).leaves).toBe(1);
    expect(fitTree([5], 1, 1, [1], S, REG).leaves).toBe(1);
    expect(fitTree([2, 2, 2, 2], 4, 1, [0, 1, 2, 3], S, REG).leaves).toBe(1);
  });

  it('does not split when the improvement is zero (xor)', () => {
    const x = [0, 0, 0, 1, 1, 0, 1, 1];
    const y = [0, 1, 1, 0];
    const fit = fitTree(x, 4, 2, y, { ...S, criterion: 'gini', depth: 4 }, { numClasses: 2 });
    expect(fit.leaves).toBe(1);
  });

  it('breaks ties by the lowest feature, then the lowest threshold', () => {
    // Both columns separate the classes equally well.
    const x = [0, 0, 1, 1, 2, 2, 3, 3];
    const y = [0, 0, 1, 1];
    const fit = fitTree(x, 4, 2, y, { ...S, criterion: 'gini', depth: 1 }, { numClasses: 2 });
    expect(splitsOf(fit)).toEqual([[0, 1.5]]);
    // Within one column, two equally good thresholds: the lower wins.
    const f2 = fitTree([0, 1, 2, 3, 4, 5], 6, 1, [0, 1, 1, 1, 1, 0], { ...S, depth: 1 }, REG);
    expect(splitsOf(f2)).toEqual([[0, 0.5]]);
  });

  it('honours depth 1..8: leaf count is bounded by 2^depth and depth is reached on rich data', () => {
    const n = 300;
    const x = Array.from({ length: n }, (_, i) => i / n);
    const y = x.map((v, i) => Math.sin(40 * v) + 0.01 * ((i * 7919) % 13));
    for (let depth = 1; depth <= 8; depth++) {
      const fit = fitTree(x, n, 1, y, { ...S, depth }, REG);
      expect(fit.leaves).toBeLessThanOrEqual(2 ** depth);
      expect(fit.leaves).toBeGreaterThan(depth);
      const maxDepth = (i: number): number => {
        const nd = fit.nodes[i];
        return nd?.kind === 'split' ? 1 + Math.max(maxDepth(nd.left), maxDepth(nd.right)) : 0;
      };
      expect(maxDepth(0)).toBe(depth);
    }
  });

  it('honours minLeaf, minSplit and minDecrease', () => {
    const x = Array.from({ length: 40 }, (_, i) => i);
    const y = x.map((v) => (v < 3 ? 100 : v % 5));
    const a = fitTree(x, 40, 1, y, { ...S, depth: 8, minLeaf: 1 }, REG);
    const b = fitTree(x, 40, 1, y, { ...S, depth: 8, minLeaf: 6 }, REG);
    const counts = (f: TreeFit) => {
      const c = new Map<number, number>();
      for (const l of treeLeafIndex(f, x, 40, 1)) c.set(l, (c.get(l) ?? 0) + 1);
      return [...c.values()];
    };
    expect(Math.min(...counts(a))).toBeLessThan(6);
    expect(Math.min(...counts(b))).toBeGreaterThanOrEqual(6);
    expect(fitTree(x, 40, 1, y, { ...S, depth: 8, minSplit: 41 }, REG).leaves).toBe(1);
    expect(fitTree(x, 40, 1, y, { ...S, depth: 8, minDecrease: 1e6 }, REG).leaves).toBe(1);
  });

  it('numbers leaves left to right and one-hot rows sum to 1', () => {
    const x = [0, 1, 2, 3, 4, 5, 6, 7];
    const y = [0, 0, 5, 5, 1, 1, 9, 9];
    const fit = fitTree(x, 8, 1, y, { ...S, depth: 2 }, REG);
    const idx = [...treeLeafIndex(fit, x, 8, 1)];
    // x is sorted, so left-to-right numbering makes the leaf ids non-decreasing and consecutive
    expect(idx).toEqual([...idx].sort((a, b) => a - b));
    expect(new Set(idx).size).toBe(fit.leaves);
    expect(idx[0]).toBe(0);
    expect(idx[7]).toBe(fit.leaves - 1);
    const oh = treeOneHot(fit, x, 8, 1);
    expect(oh.length).toBe(8 * fit.leaves);
    for (let i = 0; i < 8; i++) {
      let s = 0;
      for (let k = 0; k < fit.leaves; k++) s += oh[i * fit.leaves + k] ?? 0;
      expect(s).toBe(1);
    }
  });
});

const cfg = (set: DataConfig['set'], over: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 100,
  noise: 0.3,
  val: 0.3,
  seed: 3,
  standardize: false,
  ...over,
});

function graphOf(set: DataConfig['set'], nodes: object[], edges: [string, string][]) {
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: ['lin'],
    loss: 'loss',
    data: cfg(set),
    train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const regNodes = (tree: object = {}) => [
  { id: 'x', type: 'input', columns: ['x1'] },
  { id: 'tree', type: 'tree', ...tree },
  { id: 'lin', type: 'linear', out: 1 },
  { id: 'y', type: 'target' },
  { id: 'diff', type: 'subtract' },
  { id: 'sq', type: 'square' },
  { id: 'loss', type: 'mean' },
];
const regEdges: [string, string][] = [
  ['x', 'tree'],
  ['tree', 'lin'],
  ['lin', 'diff.a'],
  ['y', 'diff.b'],
  ['diff', 'sq'],
  ['sq', 'loss'],
];

describe('Tree node: parse, analyse, fitTrees', () => {
  it('parseGraph accepts the node and fills the defaults', () => {
    const g = graphOf('sine', regNodes(), regEdges);
    expect(g.nodes.find((n) => n.id === 'tree')).toEqual({
      id: 'tree',
      type: 'tree',
      criterion: 'squared_error',
      depth: 3,
      minSplit: 2,
      minLeaf: 1,
      minDecrease: 0,
    });
    expect(CORE_PIECES.tree.defaults()).toMatchObject({ depth: 3, minSplit: 2, minLeaf: 1 });
    // round trip
    const again = parseGraph(JSON.parse(serializeGraph(g)));
    expect(again.ok && again.graph.nodes).toEqual(g.nodes);
  });

  it('rejects out-of-range settings', () => {
    for (const bad of [
      { depth: 0 },
      { depth: 9 },
      { minSplit: 1 },
      { minLeaf: 0 },
      { minDecrease: -1 },
      { criterion: 'mse' },
    ]) {
      const r = parseGraph({
        version: 2,
        nodes: [{ id: 't', type: 'tree', ...bad }],
        edges: [],
        data: cfg('sine'),
        train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
      });
      expect(r.ok).toBe(false);
    }
  });

  it('Input -> Tree -> Linear(1): Linear reads its width from the leaf count; params = leaves + 1', () => {
    const g = graphOf('sine', regNodes({ depth: 3 }), regEdges);
    const ds = sample(g.data);
    const fits = fitTrees(g, ds);
    const leaves = fits['tree']?.leaves ?? 0;
    expect(leaves).toBeGreaterThan(1);
    expect(leaves).toBeLessThanOrEqual(8);
    const a = analyse(g, ds.info, treeLeafCounts(fits));
    expect(a.issues).toEqual([]);
    expect(a.outWidth['tree']).toBe(leaves);
    expect(a.inWidth['lin']).toBe(leaves);
    expect(a.params['lin']).toBe(leaves + 1);
    expect(a.params['tree']).toBe(0);
    expect(a.totalParams).toBe(leaves + 1);
    // one-hot of the training rows has the analysed width
    const fit = fits['tree'] as NonNullable<(typeof fits)[string]>;
    const oh = treeOneHot(fit, ds.xTrain, ds.nTrain, ds.info.columns.length);
    expect(oh.length).toBe(ds.nTrain * leaves);
  });

  it('fits on the training rows only', () => {
    const g = graphOf('sine', regNodes({ depth: 8 }), regEdges);
    const ds = sample(g.data);
    const fit = fitTrees(g, ds)['tree'];
    const direct = fitTree(
      Float64Array.from(ds.xTrain),
      ds.nTrain,
      1,
      ds.yTrain,
      g.nodes[1] as never,
      ds.info,
    );
    expect(fit?.leaves).toBe(direct.leaves);
    const roots = fit?.nodes.filter((nd) => nd.kind === 'leaf').reduce((s, nd) => s + nd.n, 0);
    expect(roots).toBe(ds.nTrain);
  });

  it('maps feature indices to dataset columns', () => {
    const nodes = regNodes();
    (nodes[0] as { columns: string[] }).columns = ['x2'];
    const g = graphOf('plane', nodes, regEdges);
    const ds = sample(g.data);
    const fit = fitTrees(g, ds)['tree'];
    expect(fit?.columns).toEqual([1]);
    for (const nd of fit?.nodes ?? []) if (nd.kind === 'split') expect(nd.feature).toBe(1);
  });

  it('says so when the tree is not fitted', () => {
    const g = graphOf('sine', regNodes(), regEdges);
    const a = analyse(g, sample(g.data).info);
    expect(a.issues.map((i) => i.message)).toContain('Tree has not been fitted to data yet.');
    expect(a.trainable).toBe(false);
  });

  it('wants an Input wired straight in', () => {
    const nodes = [
      ...regNodes(),
      { id: 'feat', type: 'feature', degree: 2, sin: false, cos: false, products: false },
    ];
    const edges: [string, string][] = regEdges.map((e) => (e[0] === 'x' ? e : e));
    edges[0] = ['x', 'feat'];
    edges.push(['feat', 'tree']);
    const g = graphOf('sine', nodes, edges);
    const ds = sample(g.data);
    const a = analyse(g, ds.info, treeLeafCounts(fitTrees(g, ds)));
    expect(a.issues.map((i) => i.message)).toContain(
      'A Tree reads data columns directly; wire an Input into it.',
    );
    expect(a.trainable).toBe(false);
  });

  it('the criterion must match the kind of data', () => {
    const g = graphOf('sine', regNodes({ criterion: 'gini' }), regEdges);
    const ds = sample(g.data);
    expect(fitTrees(g, ds)['tree']).toBeUndefined();
    const a = analyse(g, ds.info, treeLeafCounts(fitTrees(g, ds)));
    expect(
      a.issues.some((i) => /needs the squared error or absolute error criterion/.test(i.message)),
    ).toBe(true);

    const c = graphOf('blobs', regNodes({ criterion: 'squared_error' }), regEdges);
    const dc = sample(c.data);
    const ac = analyse(c, dc.info, treeLeafCounts(fitTrees(c, dc)));
    expect(ac.issues.some((i) => /needs the gini or entropy criterion/.test(i.message))).toBe(true);
    // and the right one works
    const ok = graphOf('blobs', regNodes({ criterion: 'entropy' }), regEdges);
    const dok = sample(ok.data);
    expect(fitTrees(ok, dok)['tree']?.leaves).toBeGreaterThan(1);
  });

  it('refuses a category column', () => {
    const nodes = regNodes();
    (nodes[0] as { columns: string[] }).columns = ['x4'];
    const g = graphOf('delivery_time', nodes, regEdges);
    const ds = sample(g.data);
    const fits = fitTrees(g, ds);
    expect(fits['tree']).toBeUndefined();
    const a = analyse(g, ds.info, treeLeafCounts(fits));
    expect(a.trainable).toBe(false);
    expect(a.issues.some((i) => /category/.test(i.message))).toBe(true);
  });

  it('has an equation and a torch string', () => {
    const g = graphOf('sine', regNodes(), regEdges);
    const ds = sample(g.data);
    const a = analyse(g, ds.info, treeLeafCounts(fitTrees(g, ds)));
    const eq = modelEquations(g, a, ds.info);
    expect(JSON.stringify(eq)).toContain('\\mathrm{leaf}');
    const emitted = CORE_PIECES.tree.torch(g.nodes[1] as never, {
      ins: { in: ['x'] },
      inWidth: 1,
      dataColumns: ['x1'],
      categories: 0,
      bound: 1,
      owner: 'self',
      training: 'self.training',
      self: 'self.t',
      name: (id) => id,
      tree: { estimator: 't_tree', leaves: 't_leaves', columns: [0] },
    });
    expect(emitted.pre).toContain('DecisionTreeRegressor(criterion=');
  });
});
