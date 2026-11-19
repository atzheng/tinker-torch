// One output feeding many consumers (tinker-0sq): use counts, `.ref` and disposal in the engine.
// `w` feeds add1.a, add1.b, square and the first prediction pin (four readers); `add2` is both a
// prediction and (through the loss chain) read by the loss.
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import { analyse, parseGraph } from '@tinker/core';
import type { DataInfo, Edge, Graph, HostParams, Node } from '@tinker/core';
import { predict0 } from './util.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

const data: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 1 };

function fanGraph(): Graph {
  const nodes: Node[] = [
    { id: 'c', type: 'column', name: 'x1' },
    { id: 'w', type: 'scalew' },
    { id: 'add1', type: 'add' },
    { id: 'sq', type: 'square' },
    { id: 'add2', type: 'add' },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'sq2', type: 'square' },
    { id: 'm', type: 'mean' },
  ];
  const edges: Edge[] = [
    ['c', 'w'],
    ['w', 'add1.a'],
    ['w', 'add1.b'],
    ['w', 'sq'],
    ['add1', 'add2.a'],
    ['sq', 'add2.b'],
    ['add2', 'diff.a'],
    ['y', 'diff.b'],
    ['diff', 'sq2'],
    ['sq2', 'm'],
  ];
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: ['w', 'add2'],
    loss: 'm',
    data: { set: 'plane', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'sgd', lr: 0.01, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const W0 = 0.4;
const host = (): HostParams => {
  const h = Object.create(null) as HostParams;
  h['w'] = { W: Float32Array.of(W0) };
  return h;
};
function batch(n: number) {
  const x = new Float32Array(n * 2);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[2 * i] = Math.sin(i + 1);
    x[2 * i + 1] = Math.cos(i);
    y[i] = 0.3 * (x[2 * i] ?? 0) + 0.2;
  }
  return { x, y, n };
}

test('a node with four readers (a pin and three wires) computes the right values', () => {
  const g = fanGraph();
  expect(analyse(g, data).issues).toEqual([]);
  const b = batch(16);
  const model = engine.compile(g, data);
  const params = engine.fromHost(model, host());
  const loss = engine.evaluate(model, params, b);
  let expected = 0;
  for (let i = 0; i < b.n; i++) {
    const h = W0 * (b.x[2 * i] ?? 0);
    const t = 2 * h + h * h;
    expected += (t - (b.y[i] ?? 0)) ** 2;
  }
  expect(Math.abs(loss - expected / b.n)).toBeLessThan(1e-5);
  const first = predict0(engine, model, params, b.x, b.n);
  for (let i = 0; i < b.n; i++) {
    expect(Math.abs((first[i] ?? 0) - W0 * (b.x[2 * i] ?? 0))).toBeLessThan(1e-6);
  }
  engine.dispose(params);
});

test('200 training cycles of a node feeding three consumers leave the live buffer count flat', () => {
  const g = fanGraph();
  const b = batch(16);
  const model = engine.compile(g, data);
  let state = engine.startTraining(model, engine.fromHost(model, host()), 1);
  let first = NaN;
  let last = NaN;
  const once = (i: number) => {
    const r = engine.step(model, state, b);
    state = r.state;
    expect(r.diverged).toBe(false);
    if (i === 1) first = r.loss;
    last = r.loss;
    if (i % 10 === 0) engine.evaluate(model, state.params, b);
    if (i % 20 === 0) predict0(engine, model, state.params, b.x, b.n);
  };
  for (let i = 1; i <= 30; i++) once(i);
  const baseline = engine.liveArrays();
  expect(baseline).toBeGreaterThan(0);
  for (let i = 31; i <= 230; i++) {
    once(i);
    if (i % 50 === 0) expect(engine.liveArrays()).toBe(baseline);
  }
  expect(last).toBeLessThan(first);
  engine.dispose(state);
  expect(engine.liveArrays()).toBeLessThanOrEqual(2);
});
