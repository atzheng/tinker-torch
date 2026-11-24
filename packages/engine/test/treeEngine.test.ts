// The Tree piece in the engine (tinker-6h1.3): the one-hot is computed on the host and passed to the
// jitted functions as part of x. Torch agreement is in reference.test.ts (tree_linear_*); here:
// every entry point with a tree model, and trainMany with copies of different leaf counts.
import { beforeAll, describe, expect, test } from 'vitest';
import { fitTrees, initParams, parseGraph, sample } from '@tinker/core';
import type { DataInfo, Graph } from '@tinker/core';
import { createEngine } from '../src/index.js';
import type { Engine, ParamAxis } from '../src/index.js';
import { compare } from './util.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

export function treeGraph(depth: number, optimizer: 'sgd' | 'adam' = 'sgd', lr = 0.2): Graph {
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
    data: { set: 'sine', n: 24, noise: 0.3, val: 0, seed: 1, standardize: false },
    train: { optimizer, lr, batch: 'full', seed: 1 },
  });
  if (!g.ok) throw new Error(g.error);
  return g.graph;
}

const info: DataInfo = sample(treeGraph(1).data).info;

/** Three flat steps: a depth-8 tree has 3 leaves. */
function stepData(n: number): { x: Float32Array; y: Float32Array; n: number } {
  const x = Float32Array.from({ length: n }, (_, i) => i / n);
  return { x, y: Float32Array.from(x, (v) => Math.floor(v * 3)), n };
}
/** Pseudo-random targets: a depth-8 tree has many leaves. */
function noisyData(n: number): { x: Float32Array; y: Float32Array; n: number } {
  const x = Float32Array.from({ length: n }, (_, i) => (i * 0.618034) % 1);
  return { x, y: Float32Array.from({ length: n }, (_, i) => Math.sin(37 * i)), n };
}
const fitOf = (g: Graph, b: { x: Float32Array; y: Float32Array; n: number }) =>
  fitTrees(g, { info, nTrain: b.n, xTrain: b.x, yTrain: b.y });

describe('a Tree model on the host one-hot', () => {
  test('predict, evaluate, exampleLosses, grids, curves and slopes read the fitted leaves', () => {
    const g = treeGraph(1);
    const b = stepData(24);
    const fits = fitOf(g, b);
    const model = engine.compile(g, info, fits);
    expect(model.analysis.paramShapes['lin1']?.['W']).toEqual([2, 1]);
    // W = leaf means, b = 0: the prediction is the leaf mean of each row, any rows (a grid too)
    const side = (i: number) => leafOf(fits, b, i);
    const mean = (v: number[]) => v.reduce((p, q) => p + q, 0) / v.length;
    const lm = mean(Array.from(b.y).filter((_, i) => side(i) === 0));
    const rm = mean(Array.from(b.y).filter((_, i) => side(i) === 1));
    const params = engine.fromHost(model, {
      lin1: { W: Float32Array.of(lm, rm), b: Float32Array.of(0) },
    } as never);
    const t = (fits['tree']?.nodes[0] as { threshold: number }).threshold;
    const grid = Float32Array.of(t - 0.2, t + 0.2, 0.01, 0.99);
    const out = engine.predict(model, params, grid, 4)[0];
    expect(Array.from(out ?? [], (v) => +v.toFixed(5))).toEqual(
      [lm, rm, lm, rm].map((v) => +v.toFixed(5)),
    );
    const loss = engine.evaluate(model, params, b);
    const hand =
      Array.from(b.y).reduce((s, y, i) => s + (y - (side(i) === 0 ? lm : rm)) ** 2, 0) / b.n;
    expect(loss).toBeCloseTo(hand, 5);
    const ex = engine.exampleLosses(model, params, b);
    expect(ex).not.toBeNull();
    expect(ex?.length).toBe(24);
    const axes: [ParamAxis, ParamAxis] = [
      { node: 'lin1', tensor: 'W', index: 0 },
      { node: 'lin1', tensor: 'W', index: 1 },
    ];
    const surf = engine.lossGrid(
      model,
      params,
      b,
      axes,
      Float32Array.of(lm, 0),
      Float32Array.of(rm),
    );
    expect(surf[0]).toBeCloseTo(hand, 5);
    expect(surf[1]).toBeGreaterThan(hand);
    const curve = engine.lossCurve(model, params, b, axes[0], Float32Array.of(lm, lm + 1));
    expect(curve[0]).toBeCloseTo(hand, 5);
    expect(curve[1]).toBeGreaterThan(hand);
    // at the leaf means the gradient of the mean squared error is 0
    expect(Math.abs(engine.lossSlope(model, params, b, axes[0]))).toBeLessThan(1e-5);
    expect(engine.lossSlope(model, params, b, { node: 'lin1', tensor: 'b', index: 0 })).toBeCloseTo(
      0,
      5,
    );
    engine.dispose(params);
    engine.disposeModel(model);
  });

  test('training reaches the leaf means (Adam)', () => {
    const g = treeGraph(2, 'adam', 0.05);
    const b = noisyData(24);
    const fits = fitOf(g, b);
    const model = engine.compile(g, info, fits);
    let state = engine.startTraining(model, engine.init(model, 3), 3);
    let last = Infinity;
    for (let i = 0; i < 400; i++) {
      const r = engine.step(model, state, b);
      expect(r.diverged).toBe(false);
      state = r.state;
      last = r.loss;
    }
    // the optimum is the within-leaf variance, reached from above
    const opt = engine.fromHost(model, {
      lin1: {
        W: Float32Array.from({ length: fits['tree']?.leaves ?? 0 }, (_, l) => {
          const rows = Array.from(b.y).filter((_, i) => leafOf(fits, b, i) === l);
          return rows.reduce((p, q) => p + q, 0) / Math.max(1, rows.length);
        }),
        b: Float32Array.of(0),
      },
    } as never);
    expect(last).toBeLessThan(engine.evaluate(model, opt, b) * 1.01 + 1e-4);
    engine.dispose(state);
    engine.dispose(opt);
    engine.disposeModel(model);
  });

  test('a model without fits says the tree has not been fitted', () => {
    expect(() => engine.compile(treeGraph(1), info)).toThrow(/not been fitted/);
  });
});

function leafOf(fits: ReturnType<typeof fitOf>, b: { x: Float32Array }, i: number): number {
  let node = 0;
  const f = fits['tree'];
  if (!f) return -1;
  for (;;) {
    const nd = f.nodes[node];
    if (!nd) return -1;
    if (nd.kind === 'leaf') return nd.leaf;
    node = (b.x[i] ?? 0) <= nd.threshold ? nd.left : nd.right;
  }
}

describe('trainMany with trees', () => {
  test('copies with different leaf counts equal one-by-one training and return their own fits and unpadded params', () => {
    const g = treeGraph(8, 'adam', 0.05);
    const sets = [stepData(24), noisyData(24), stepData(24), noisyData(24)];
    const seeds = [5, 6, 7, 8];
    const steps = 30;
    const fits = sets.map((b) => fitOf(g, b));
    const leaves = fits.map((f) => f['tree']?.leaves ?? 0);
    expect(leaves[0]).toBe(3);
    expect(leaves[1]).toBeGreaterThan(3);
    // the model's own fit gives only the widths; trainMany refits every copy
    const model = engine.compile(g, info, fits[0]);
    const many = engine.trainMany(model, seeds, sets, steps);
    expect(many).toHaveLength(4);
    many.forEach((res, c) => {
      expect(res.diverged).toBe(false);
      expect(res.fits['tree']?.leaves).toBe(leaves[c]);
      expect(res.params['lin1']?.['W']).toHaveLength(leaves[c] ?? 0);
      // the same copy, one step at a time, on a model made from its own fit
      const m = engine.compile(g, info, fits[c]);
      let st = engine.startTraining(m, engine.init(m, seeds[c] ?? 0), seeds[c] ?? 0);
      for (let i = 0; i < steps; i++) st = engine.step(m, st, sets[c] as never).state;
      const want = engine.toHost(st.params);
      for (const key of ['W', 'b']) {
        const cmp = compare(res.params['lin1']?.[key] ?? [], want['lin1']?.[key] ?? [], 1e-5);
        expect(cmp.ok, `copy ${c} ${key} max abs error ${cmp.maxAbs}`).toBe(true);
      }
      // a copy predicts with its own tree
      const ownModel = engine.compile(g, info, res.fits);
      const p = engine.fromHost(ownModel, res.params);
      const pred = engine.predict(ownModel, p, sets[c]?.x ?? new Float32Array(), 24)[0];
      const ref = engine.predict(m, st.params, sets[c]?.x ?? new Float32Array(), 24)[0];
      expect(compare(pred ?? [], ref ?? [], 1e-5).ok).toBe(true);
      engine.dispose(p);
      engine.disposeModel(ownModel);
      engine.dispose(st);
      engine.disposeModel(m);
    });
    engine.disposeModel(model);
  });

  test('zero steps returns each copy init with fan-in = its own leaf count', () => {
    const g = treeGraph(8);
    const sets = [stepData(24), noisyData(24)];
    const model = engine.compile(g, info, fitOf(g, sets[0] as never));
    const many = engine.trainMany(model, [1, 2], sets, 0);
    many.forEach((res, c) => {
      const f = fitOf(g, sets[c] as never);
      const m = engine.compile(g, info, f);
      const want = initParams(m.analysis, c + 1);
      expect(Array.from(res.params['lin1']?.['W'] ?? [])).toEqual(
        Array.from(want['lin1']?.['W'] ?? []),
      );
      engine.disposeModel(m);
    });
    engine.disposeModel(model);
  });
});
