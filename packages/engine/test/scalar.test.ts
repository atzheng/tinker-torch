// Scalar pieces (Column, Weight, Bias, Sum; tinker-9io.4): the paired "a Linear is the same thing"
// check, the leak test and the whole-engine entry points (lossGrid, trainMany) on a scalar graph.
// The probe must be installed before jax-js loads, hence the dynamic import.
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import { parseGraph } from '@tinker/core';
import type { Graph } from '@tinker/core';
import { compare, loadFixture, predict0, toBatch, toHostParams } from './util.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

/** The scalar regression graph of a fixture rebuilt as input -> linear(d, 1) -> same loss. */
function asLinear(g: Graph, d: number): Graph {
  const scalarTypes = ['column', 'scalew', 'bias', 'sum', 'act'];
  const keep = g.nodes.filter((n) => !scalarTypes.includes(n.type));
  const dropped = new Set(g.nodes.filter((n) => scalarTypes.includes(n.type)).map((n) => n.id));
  const edges = g.edges.filter(([a, b]) => ![a, b].some((e) => dropped.has(e.split('.')[0] ?? '')));
  const raw = {
    ...g,
    nodes: [
      { id: 'x', type: 'input', columns: Array.from({ length: d }, (_, i) => `x${i + 1}`) },
      { id: 'lin', type: 'linear', out: 1, bias: true },
      ...keep,
    ],
    edges: [...edges, ['x', 'lin'], ['lin', 'diff.a']],
    predictions: ['lin'],
  };
  const parsed = parseGraph(raw);
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}

/** Gradient of every parameter from one plain SGD step with learning rate 1 (grad = before - after). */
function sgdStep(
  graph: Graph,
  data: ReturnType<typeof loadFixture>['data'],
  host: object,
  b: object,
) {
  const g = { ...graph, train: { ...graph.train, optimizer: 'sgd' as const, lr: 1 } };
  const model = engine.compile(g, data);
  const params = engine.fromHost(model, host as never);
  const state = engine.startTraining(model, params, 0);
  const r = engine.step(model, state, b as never);
  const after = engine.toHost(r.state.params);
  engine.dispose(r.state);
  return { loss: r.loss, after };
}

test('column x d -> weight -> sum -> bias equals input -> linear(d, 1) (pred, loss, grads, 20 steps)', () => {
  const d = 3;
  const { fx, graph, data } = loadFixture('scalar_regression_d3');
  const batch = toBatch(fx);
  const lg = asLinear(graph, d);
  const host = toHostParams(fx.params);
  const w = [1, 2, 3].map((i) => host[`w${i}`]?.['W']?.[0] ?? NaN);
  const linHost = {
    lin: { W: Float32Array.from(w), b: Float32Array.from(host['b']?.['b'] ?? []) },
  };

  const sm = engine.compile(graph, data);
  const lm = engine.compile(lg, data);
  const sp = engine.fromHost(sm, host);
  const lp = engine.fromHost(lm, linHost);
  const pred = compare(
    predict0(engine, sm, sp, batch.x, batch.n),
    predict0(engine, lm, lp, batch.x, batch.n),
    1e-6,
  );
  expect(pred.ok, `pred max abs error ${pred.maxAbs}`).toBe(true);
  const loss = compare([engine.evaluate(sm, sp, batch)], [engine.evaluate(lm, lp, batch)], 1e-6);
  expect(loss.ok, `loss max abs error ${loss.maxAbs}`).toBe(true);
  engine.dispose(sp);
  engine.dispose(lp);

  // gradients: one SGD step at lr 1; then 20 steps of lr 0.1 must keep the two in step
  const s1 = sgdStep(graph, data, host, batch);
  const l1 = sgdStep(lg, data, linHost, batch);
  const sGrad = [1, 2, 3].map(
    (i) => (host[`w${i}`]?.['W']?.[0] ?? 0) - (s1.after[`w${i}`]?.['W']?.[0] ?? 0),
  );
  const lGrad = Array.from(linHost.lin.W, (v, i) => v - (l1.after['lin']?.['W']?.[i] ?? 0));
  expect(compare(sGrad, lGrad, 1e-6).ok).toBe(true);
  const bGrad = (host['b']?.['b']?.[0] ?? 0) - (s1.after['b']?.['b']?.[0] ?? 0);
  const lbGrad = (linHost.lin.b[0] ?? 0) - (l1.after['lin']?.['b']?.[0] ?? 0);
  expect(Math.abs(bGrad - lbGrad)).toBeLessThan(1e-6);

  let a = engine.startTraining(sm, engine.fromHost(sm, host), 0);
  let b = engine.startTraining(lm, engine.fromHost(lm, linHost), 0);
  for (let i = 0; i < 20; i++) {
    const ra = engine.step(sm, a, batch);
    const rb = engine.step(lm, b, batch);
    a = ra.state;
    b = rb.state;
    expect(Math.abs(ra.loss - rb.loss)).toBeLessThan(1e-5);
  }
  const ha = engine.toHost(a.params);
  const hb = engine.toHost(b.params);
  const wa = [1, 2, 3].map((i) => ha[`w${i}`]?.['W']?.[0] ?? NaN);
  expect(compare(wa, Array.from(hb['lin']?.['W'] ?? []), 1e-5).ok).toBe(true);
  expect(Math.abs((ha['b']?.['b']?.[0] ?? NaN) - (hb['lin']?.['b']?.[0] ?? 0))).toBeLessThan(1e-5);
  engine.dispose(a);
  engine.dispose(b);
});

test('lossGrid over a Weight and a Bias equals evaluate with the values written', () => {
  const { fx, graph, data } = loadFixture('scalar_regression_d3');
  const batch = toBatch(fx);
  const model = engine.compile(graph, data);
  const params = engine.fromHost(model, toHostParams(fx.params));
  const a = Float32Array.from([-0.5, 0, 0.7]);
  const b = Float32Array.from([-0.2, 0.3]);
  const grid = engine.lossGrid(
    model,
    params,
    batch,
    [
      { node: 'w2', tensor: 'W', index: 0 },
      { node: 'b', tensor: 'b', index: 0 },
    ],
    a,
    b,
  );
  expect(grid).toHaveLength(6);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      const p = engine.write(
        engine.write(engine.clone(params), 'w2', { W: Float32Array.of(a[i] ?? 0) }),
        'b',
        { b: Float32Array.of(b[j] ?? 0) },
      );
      const want = engine.evaluate(model, p, batch);
      expect(Math.abs((grid[i * b.length + j] ?? NaN) - want)).toBeLessThan(1e-5);
      engine.dispose(p);
    }
  }
  engine.dispose(params);
});

test('trainMany on a scalar graph equals training each copy alone', () => {
  const { fx, graph, data } = loadFixture('scalar_regression_d3');
  const batch = toBatch(fx);
  const model = engine.compile(
    { ...graph, train: { ...graph.train, optimizer: 'adam', lr: 0.05 } },
    data,
  );
  const seeds = [3, 4];
  const many = engine.trainMany(model, seeds, [batch, batch], 15);
  seeds.forEach((s, i) => {
    let state = engine.startTraining(model, engine.init(model, s), s);
    for (let t = 0; t < 15; t++) state = engine.step(model, state, batch).state;
    const host = engine.toHost(state.params);
    engine.dispose(state);
    const got = many[i]?.params;
    for (const id of ['w1', 'w2', 'w3']) {
      expect(Math.abs((got?.[id]?.['W']?.[0] ?? NaN) - (host[id]?.['W']?.[0] ?? 0))).toBeLessThan(
        1e-4,
      );
    }
    expect(Math.abs((got?.['b']?.['b']?.[0] ?? NaN) - (host['b']?.['b']?.[0] ?? 0))).toBeLessThan(
      1e-4,
    );
  });
});

test('a scalar graph trains 3,000 steps with the live buffer count flat', () => {
  const { fx, graph, data } = loadFixture('scalar_adam_5steps');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const once = (i: number) => {
    const r = engine.step(model, state, batch);
    state = r.state;
    expect(r.diverged).toBe(false);
    if (i % 10 === 0) engine.evaluate(model, state.params, batch);
    if (i % 25 === 0) predict0(engine, model, state.params, batch.x, batch.n);
  };
  for (let i = 1; i <= 60; i++) once(i);
  const baseline = engine.liveArrays();
  expect(baseline).toBeGreaterThan(0);
  const samples: number[] = [];
  for (let i = 1; i <= 3000; i++) {
    once(i);
    if (i % 500 === 0) samples.push(engine.liveArrays());
  }
  expect(samples).toEqual(samples.map(() => baseline));
  engine.dispose(state);
  expect(engine.liveArrays()).toBeLessThan(baseline);
});

test('compiling and training 200 scalar models leaves no buffers behind', () => {
  for (const name of ['scalar_regression_d3', 'scalar_sum_two_cols']) {
    const { fx, graph, data } = loadFixture(name);
    const batch = toBatch(fx);
    const cycle = () => {
      const model = engine.compile(graph, data);
      let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
      for (let i = 0; i < 2; i++) state = engine.step(model, state, batch).state;
      engine.dispose(state);
    };
    for (let i = 0; i < 5; i++) cycle();
    const baseline = engine.liveArrays();
    for (let i = 0; i < 200; i++) cycle();
    expect(engine.liveArrays(), name).toBe(baseline);
  }
});

test('unwired (inert) pieces change nothing: same losses and parameters, no params of their own, flat buffers', () => {
  const { fx, graph, data } = loadFixture('scalar_adam_5steps');
  const parsed = parseGraph({
    ...graph,
    nodes: [
      ...graph.nodes,
      { id: 'lone_in', type: 'input', columns: ['x1'] },
      { id: 'lone_lin', type: 'linear', out: 3, bias: true },
      { id: 'lone_sub', type: 'subtract' },
      { id: 'lone_ce', type: 'cross_entropy' },
      { id: 'lone_col', type: 'column', name: 'x1' },
    ],
    edges: [...graph.edges, ['lone_in', 'lone_lin']],
  });
  if (!parsed.ok) throw new Error(parsed.error);
  const withInert = parsed.graph;
  const batch = toBatch(fx);
  const run = (g: Graph) => {
    const model = engine.compile(g, data);
    let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
    const losses: number[] = [];
    for (let i = 0; i < 10; i++) {
      const r = engine.step(model, state, batch);
      state = r.state;
      losses.push(r.loss);
    }
    const params = engine.toHost(state.params);
    engine.dispose(state);
    return { losses, params };
  };
  const a = run(graph);
  const baseline = engine.liveArrays();
  const b = run(withInert);
  expect(b.losses).toEqual(a.losses);
  expect(b.params).toEqual(a.params);
  expect(Object.keys(b.params)).not.toContain('lone_lin');
  expect(engine.liveArrays()).toBe(baseline);
});
