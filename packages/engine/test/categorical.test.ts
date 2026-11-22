// Categorical columns (tinker-6t7.7): a Lookup of width 1 into Sigmoid and log loss converges to each
// category's sample rate, and One-hot / Lookup models leak no buffers. The probe must be installed
// before jax-js loads, hence the dynamic import.
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import { parseGraph, sample } from '@tinker/core';
import type { Graph } from '@tinker/core';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

const base = {
  version: 2,
  data: { set: 'delivery_late', n: 400, noise: 0.3, val: 0.25, seed: 5, standardize: false },
  train: { optimizer: 'adam', lr: 0.1, batch: 'full', seed: 1 },
};
const build = (raw: object): Graph => {
  const p = parseGraph({ ...base, ...raw });
  if (!p.ok) throw new Error(p.error);
  return p.graph;
};

/** zone -> Lookup(1) -> Sigmoid -> log loss. */
const lookupGraph = build({
  nodes: [
    { id: 'zone', type: 'input', columns: ['x4'] },
    { id: 'look', type: 'lookup', width: 1 },
    { id: 'p', type: 'link', fn: 'sigmoid' },
    { id: 'y', type: 'target' },
    { id: 'err', type: 'log_loss' },
    { id: 'loss', type: 'mean' },
  ],
  edges: [
    ['zone', 'look'],
    ['look', 'p'],
    ['p', 'err.pred'],
    ['y', 'err.target'],
    ['err', 'loss'],
  ],
  predictions: ['p'],
  loss: 'loss',
});

test('a width-1 Lookup trained by log loss converges to each zone sample late rate', () => {
  const ds = sample(lookupGraph.data);
  const model = engine.compile(lookupGraph, ds.info);
  const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
  let state = engine.startTraining(model, engine.init(model, 1), 1);
  for (let i = 0; i < 1500; i++) state = engine.step(model, state, batch).state;
  const w = engine.toHost(state.params)['look']?.['W'];
  expect(w).toHaveLength(4);
  for (let z = 0; z < 4; z++) {
    let late = 0;
    let n = 0;
    for (let i = 0; i < ds.nTrain; i++) {
      if (ds.xTrain[i * 4 + 3] === z) {
        n++;
        late += ds.yTrain[i] ?? 0;
      }
    }
    expect(n).toBeGreaterThan(20);
    const p = 1 / (1 + Math.exp(-(w?.[z] ?? NaN)));
    expect(Math.abs(p - late / n)).toBeLessThan(0.01);
  }
  // predict: one probability per row, the zone's rate
  const pred = engine.predict(model, state.params, ds.xTrain, ds.nTrain)[0];
  expect(pred).toHaveLength(ds.nTrain);
  engine.dispose(state);
  engine.disposeModel(model);
});

test('trainMany (vmap) works on a Lookup model', () => {
  const ds = sample(lookupGraph.data);
  const model = engine.compile(lookupGraph, ds.info);
  const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
  const out = engine.trainMany(model, [1, 2], [batch, batch], 20);
  expect(out).toHaveLength(2);
  expect(out.every((r) => !r.diverged && r.params['look']?.['W']?.length === 4)).toBe(true);
  engine.disposeModel(model);
});

/** zone -> One-hot -> Linear (+ numeric distance -> Linear) -> squared error; and a Lookup(2) variant. */
const mixedGraph = build({
  data: { ...base.data, set: 'delivery_time' },
  nodes: [
    { id: 'zone', type: 'input', columns: ['x4'] },
    { id: 'dist', type: 'input', columns: ['x1'] },
    { id: 'oh', type: 'onehot' },
    { id: 'lz', type: 'linear', out: 1, bias: false },
    { id: 'ld', type: 'linear', out: 1, bias: true },
    { id: 'sum', type: 'add' },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'sq', type: 'square' },
    { id: 'loss', type: 'mean' },
  ],
  edges: [
    ['zone', 'oh'],
    ['oh', 'lz'],
    ['dist', 'ld'],
    ['lz', 'sum.a'],
    ['ld', 'sum.b'],
    ['sum', 'diff.a'],
    ['y', 'diff.b'],
    ['diff', 'sq'],
    ['sq', 'loss'],
  ],
  predictions: ['sum'],
  loss: 'loss',
});

test('One-hot into a Linear learns the zone effect and the distance slope', () => {
  const ds = sample(mixedGraph.data);
  const model = engine.compile(mixedGraph, ds.info);
  const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
  let state = engine.startTraining(model, engine.init(model, 1), 1);
  let first = 0;
  let last = 0;
  for (let i = 0; i < 1500; i++) {
    const r = engine.step(model, state, batch);
    state = r.state;
    if (i === 0) first = r.loss;
    last = r.loss;
  }
  expect(last).toBeLessThan(first / 4);
  const h = engine.toHost(state.params);
  // the zone effects are -2, -1, 1, 2 minutes around the mean: differences between zones are recovered
  const wz = h['lz']?.['W'] ?? new Float32Array(0);
  expect((wz[3] ?? 0) - (wz[0] ?? 0)).toBeGreaterThan(2);
  expect(h['ld']?.['W']?.[0]).toBeGreaterThan(2);
  engine.dispose(state);
  engine.disposeModel(model);
});

test('a model with One-hot and Lookup leaves the live buffer count flat', () => {
  const lookup2 = build({
    nodes: [
      { id: 'zone', type: 'input', columns: ['x4'] },
      { id: 'look', type: 'lookup', width: 2 },
      { id: 'lin', type: 'linear', out: 1, bias: true },
      { id: 'p', type: 'link', fn: 'sigmoid' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['zone', 'look'],
      ['look', 'lin'],
      ['lin', 'p'],
      ['p', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
    predictions: ['p'],
    loss: 'loss',
  });
  for (const g of [lookup2, mixedGraph, lookupGraph]) {
    const ds = sample(g.data);
    const model = engine.compile(g, ds.info);
    const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
    let state = engine.startTraining(model, engine.init(model, 1), 1);
    const once = (i: number) => {
      const r = engine.step(model, state, batch);
      state = r.state;
      if (i % 10 === 0) engine.evaluate(model, state.params, batch);
      if (i % 25 === 0) engine.predict(model, state.params, ds.xTrain, ds.nTrain);
    };
    for (let i = 1; i <= 40; i++) once(i);
    const baseline = engine.liveArrays();
    expect(baseline).toBeGreaterThan(0);
    for (let i = 1; i <= 1500; i++) once(i);
    expect(engine.liveArrays()).toBe(baseline);
    engine.dispose(state);
    engine.disposeModel(model);
  }
  // compiling and discarding many models leaves no constants behind
  const ds = sample(lookup2.data);
  const cycle = () => {
    const model = engine.compile(lookup2, ds.info);
    let state = engine.startTraining(model, engine.init(model, 1), 1);
    const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
    for (let i = 0; i < 2; i++) state = engine.step(model, state, batch).state;
    engine.dispose(state);
    engine.disposeModel(model);
  };
  for (let i = 0; i < 5; i++) cycle();
  const baseline = engine.liveArrays();
  for (let i = 0; i < 100; i++) cycle();
  expect(engine.liveArrays()).toBe(baseline);
});
