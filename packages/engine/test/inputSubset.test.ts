// An Input with a column subset or reorder (tinker-daq): the right columns, and no constants
// leaked per compile. The probe must be installed before jax-js loads.
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import { parseGraph } from '@tinker/core';
import type { Graph } from '@tinker/core';
import { loadFixture, predict0 } from './util.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

/** linear_bias_d2 with its Input's columns replaced, over a 3-column dataset. */
function graphWith(columns: string[]): Graph {
  const { graph } = loadFixture('linear_bias_d2');
  const raw = {
    ...graph,
    nodes: graph.nodes.map((n) => (n.type === 'input' ? { ...n, columns } : n)),
  };
  const parsed = parseGraph(raw);
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}
const data = { columns: ['x1', 'x2', 'x3'], kind: 'regression' as const, numClasses: 1 };
const n = 5;
const x = Float32Array.from({ length: n * 3 }, (_, i) => Math.sin(i + 1));

test('an Input subset or reorder gives the host-selected columns', () => {
  for (const cols of [['x2'], ['x3', 'x1'], ['x1', 'x3'], ['x2', 'x3'], ['x3', 'x1', 'x2']]) {
    const model = engine.compile(graphWith(cols), data);
    const idx = cols.map((c) => Number(c.slice(1)) - 1);
    const W = Float32Array.from(idx.map((_, i) => 0.3 + 0.2 * i));
    const params = engine.fromHost(model, { lin1: { W, b: Float32Array.from([0.1]) } });
    const got = predict0(engine, model, params, x, n);
    engine.dispose(params);
    const want = Array.from({ length: n }, (_, r) =>
      idx.reduce((s, c, i) => s + (x[r * 3 + c] ?? 0) * (W[i] ?? 0), 0.1),
    );
    expect(got.length).toBe(n);
    got.forEach((v, r) => expect(Math.abs(v - (want[r] ?? NaN))).toBeLessThan(1e-4));
  }
});

test('compiling, training, predicting and disposing 200 subset-Input graphs leaves no buffers', () => {
  const y = Float32Array.from({ length: n }, (_, i) => i / 10);
  const batch = { x, y, n };
  for (const cols of [['x3', 'x1'], ['x1', 'x2']]) {
    const g = graphWith(cols);
    const cycle = () => {
      const model = engine.compile(g, data);
      let state = engine.startTraining(model, engine.init(model, 1), 1);
      for (let i = 0; i < 2; i++) state = engine.step(model, state, batch).state;
      predict0(engine, model, state.params, x, n);
      engine.evaluate(model, state.params, batch);
      engine.dispose(state);
    };
    for (let i = 0; i < 5; i++) cycle();
    const baseline = engine.liveArrays();
    for (let i = 0; i < 200; i++) cycle();
    expect(engine.liveArrays(), cols.join()).toBe(baseline);
  }
});

test('a predictions-only head (not read by the loss) leaves no buffers per compile cycle', () => {
  const { fx, graph, data: d } = loadFixture('prim_two_predictions');
  const batch = {
    x: Float32Array.from(fx.batch.x),
    y: Float32Array.from(fx.batch.y),
    n: fx.batch.n,
  };
  const cycle = () => {
    const model = engine.compile(graph, d);
    let state = engine.startTraining(model, engine.init(model, 1), 1);
    for (let i = 0; i < 2; i++) state = engine.step(model, state, batch).state;
    predict0(engine, model, state.params, batch.x, batch.n);
    engine.evaluate(model, state.params, batch);
    engine.dispose(state);
  };
  for (let i = 0; i < 5; i++) cycle();
  const baseline = engine.liveArrays();
  for (let i = 0; i < 200; i++) cycle();
  expect(engine.liveArrays()).toBe(baseline);
});
