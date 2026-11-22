// delivery_lateness (tinker-7bz): a softmax + cross-entropy model on the 3-class delivery target
// trains (loss falls, accuracy beats the majority class).
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import { analyse, parseGraph, sample } from '@tinker/core';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

const parsed = parseGraph({
  version: 2,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1', 'x2', 'x3'] },
    { id: 'lin', type: 'linear', out: 3, bias: true },
    { id: 'p', type: 'link', fn: 'softmax' },
    { id: 'y', type: 'target' },
    { id: 'err', type: 'cross_entropy' },
    { id: 'loss', type: 'mean' },
  ],
  edges: [
    ['x', 'lin'],
    ['lin', 'p'],
    ['p', 'err.pred'],
    ['y', 'err.target'],
    ['err', 'loss'],
  ],
  predictions: ['p'],
  loss: 'loss',
  data: { set: 'delivery_lateness', n: 500, noise: 0.3, val: 0.25, seed: 4, standardize: true },
  train: { optimizer: 'adam', lr: 0.05, batch: 'full', seed: 1 },
});
if (!parsed.ok) throw new Error(parsed.error);
const graph = parsed.graph;

test('softmax + cross-entropy on delivery_lateness: valid, loss decreases, beats the majority class', () => {
  const ds = sample(graph.data);
  expect(ds.info.kind).toBe('multiclass');
  expect(ds.info.numClasses).toBe(3);
  expect(analyse(graph, ds.info).trainable).toBe(true);
  const model = engine.compile(graph, ds.info);
  const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain };
  const before = engine.evaluate(model, engine.init(model, 1), batch);
  let state = engine.startTraining(model, engine.init(model, 1), 1);
  for (let i = 0; i < 300; i++) state = engine.step(model, state, batch).state;
  const after = engine.evaluate(model, state.params, batch);
  expect(after).toBeLessThan(before * 0.7);
  const probs = engine.predict(model, state.params, ds.xTrain, ds.nTrain)[0] ?? new Float32Array();
  let right = 0;
  const counts = [0, 0, 0];
  for (let i = 0; i < ds.nTrain; i++) {
    let best = 0;
    for (let c = 1; c < 3; c++) if ((probs[i * 3 + c] ?? 0) > (probs[i * 3 + best] ?? 0)) best = c;
    const label = ds.yTrain[i] ?? 0;
    if (best === label) right++;
    counts[label] = (counts[label] ?? 0) + 1;
  }
  expect(right / ds.nTrain).toBeGreaterThan(Math.max(...counts) / ds.nTrain + 0.1);
  engine.dispose(state);
  engine.disposeModel(model);
});
