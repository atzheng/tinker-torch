// The resampling overlay with a Tree model (tinker-6h1.3): every copy is trained on its own
// resample and predicted with ITS OWN tree. A depth-1 tree has two leaves, so a trained copy is the
// two leaf means of its own training set around its own threshold: curves, bias and variance are
// computed by hand here and compared with the overlay's.
import { installProbe } from '@tinker/engine/probe';
import { expect, test, vi } from 'vitest';
import {
  biasVariance,
  fitTrees,
  grid1d,
  makeRng,
  parseGraph,
  sample,
  treeLeafIndex,
  truth,
  type Graph,
} from '@tinker/core';
import { runResample } from '../src/results/resampleRun.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('@tinker/engine');
const engine = await createEngine();

function treeGraph(depth: number): Graph {
  const g = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      {
        id: 'tree',
        type: 'tree',
        criterion: 'squared_error',
        depth,
        minSplit: 2,
        minLeaf: 1,
        minDecrease: 0,
      },
      { id: 'lin1', type: 'linear', out: 1, bias: true },
      { id: 'y', type: 'target' },
      { id: 'diff', type: 'subtract' },
      { id: 'err', type: 'square' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['x', 'tree'],
      ['tree', 'lin1'],
      ['lin1', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    predictions: ['lin1'],
    loss: 'loss',
    data: { set: 'sine', n: 40, noise: 0.3, val: 0, seed: 1, standardize: false },
    train: { optimizer: 'sgd', lr: 0.2, batch: 'full', seed: 1 },
  });
  if (!g.ok) throw new Error(g.error);
  return g.graph;
}

test('each copy is drawn with its own depth-1 tree; bias and variance equal the hand computation', () => {
  const graph = treeGraph(1);
  const k = 8;
  const seed = 11;
  const gridN = 100;
  const r = runResample(engine, { graph, steps: 1500, k, seed, gridN });
  if (!r.ok) throw new Error(r.message);
  expect(r.diverged).toBe(0);
  expect(r.curves).toHaveLength(k);

  // by hand: the same resamples as runResample, the leaf means of each copy's own tree
  const base = makeRng(seed);
  const expected: Float32Array[] = [];
  const thresholds = new Set<number>();
  for (let i = 0; i < k; i++) {
    const ds = sample({ ...graph.data, seed: base.fork('data' + i).int(2 ** 31) });
    const fit = fitTrees(graph, ds)['tree'];
    if (!fit) throw new Error('no fit');
    expect(fit.leaves).toBe(2);
    const root = fit.nodes[0];
    if (root?.kind !== 'split') throw new Error('expected a split');
    thresholds.add(root.threshold);
    const leaf = treeLeafIndex(fit, ds.xTrain, ds.nTrain, 1);
    const sum = [0, 0];
    const cnt = [0, 0];
    leaf.forEach((l, j) => {
      sum[l] = (sum[l] ?? 0) + (ds.yTrain[j] ?? 0);
      cnt[l] = (cnt[l] ?? 0) + 1;
    });
    const grid = grid1d(ds.domain, gridN, ds.standardization);
    const gl = treeLeafIndex(fit, grid.xStd, grid.n, 1);
    expected.push(Float32Array.from(gl, (l) => (sum[l] ?? 0) / (cnt[l] ?? 1)));
  }
  expect(thresholds.size).toBeGreaterThan(1); // the copies really have different trees
  r.curves.forEach((c, i) => {
    const want = expected[i] as Float32Array;
    let worst = 0;
    c.forEach((v, j) => (worst = Math.max(worst, Math.abs(v - (want[j] ?? NaN)))));
    expect(worst, `copy ${i}`).toBeLessThan(2e-3);
  });

  const grid = grid1d(sample(graph.data).domain, gridN);
  const t = truth(graph.data.set, grid.xRaw);
  const got = biasVariance(r.curves, t);
  const hand = biasVariance(expected, t);
  expect(got.variance).toBeCloseTo(hand.variance, 3);
  expect(got.bias2).toBeCloseTo(hand.bias2, 3);
  expect(hand.variance).toBeGreaterThan(0);
});

test('copies with different leaf counts (depth 8) are all drawn and leave no live buffers behind', () => {
  const graph = treeGraph(8);
  const run = () => runResample(engine, { graph, steps: 30, k: 5, seed: 4, gridN: 60 });
  const first = run();
  expect(first.ok).toBe(true);
  if (!first.ok) return;
  expect(first.curves).toHaveLength(5);
  const before = engine.liveArrays();
  expect(run().ok).toBe(true);
  expect(engine.liveArrays()).toBe(before);
});
