// 10,000 training steps with the live-buffer count flat (CONTRACTS.md section 7, docs/spike.md
// section 4). The probe must be installed before jax-js loads, hence the dynamic import.
import { installProbe } from '../src/probe.js';
import { expect, test } from 'vitest';
import { fitFixtureTrees, loadFixture, toBatch, toHostParams, predict0 } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');

test('10,000 steps (with predict, evaluate, clone and diverged steps) leave the live buffer count flat', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
  const diverge = (state: ReturnType<typeof engine.startTraining>) => {
    const r = engine.step(model, state, bad);
    expect(r.diverged).toBe(true);
    return r.state;
  };

  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const once = (i: number) => {
    const r = engine.step(model, state, batch);
    state = r.state;
    expect(r.diverged).toBe(false);
    if (i % 10 === 0) engine.evaluate(model, state.params, batch);
    if (i % 25 === 0) predict0(engine, model, state.params, batch.x, batch.n);
    if (i % 100 === 0) engine.dispose(engine.clone(state.params));
    if (i % 500 === 0) state = diverge(state);
  };

  // warm up: jitted functions, library-global constant buffers
  for (let i = 1; i <= 60; i++) once(i);
  state = diverge(state);
  const baseline = engine.liveArrays();
  expect(baseline).toBeGreaterThan(0); // the probe works; 0 means it never saw the backend
  const samples: number[] = [];
  for (let i = 1; i <= 10_000; i++) {
    once(i);
    if (i % 1000 === 0) samples.push(engine.liveArrays());
  }
  expect(samples).toEqual(samples.map(() => baseline));

  engine.dispose(state);
  expect(engine.liveArrays()).toBeLessThan(baseline);
  // the engine's own use leaves only the library-global caches behind
  expect(engine.liveArrays()).toBeLessThanOrEqual(2);
}, 120_000);

test('a graph with dropout (key splitting, diverged steps) leaves the live buffer count flat', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('dropout_p0.5');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const once = (i: number) => {
    const r = engine.step(model, state, i % 50 === 0 ? bad : batch);
    state = r.state;
    if (i % 10 === 0) engine.evaluate(model, state.params, batch);
    if (i % 25 === 0) predict0(engine, model, state.params, batch.x, batch.n);
  };
  for (let i = 1; i <= 60; i++) once(i);
  const baseline = engine.liveArrays();
  expect(baseline).toBeGreaterThan(0);
  const samples: number[] = [];
  for (let i = 1; i <= 2000; i++) {
    once(i);
    if (i % 500 === 0) samples.push(engine.liveArrays());
  }
  expect(samples).toEqual(samples.map(() => baseline));
  engine.dispose(state);
  expect(engine.liveArrays()).toBeLessThan(baseline);
  expect(engine.liveArrays()).toBeLessThanOrEqual(2);
}, 120_000);

test('compiling many dropout and feature models leaves no constants behind', async () => {
  const engine = await createEngine();
  for (const name of ['dropout_p0.5', 'feature_deg3_sin_cos_products']) {
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
}, 120_000);

test.each(['momentum', 'adam'] as const)(
  '%s: 3,000 steps (with diverged steps) leave the live buffer count flat',
  async (optimizer) => {
    const engine = await createEngine();
    const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
    const model = engine.compile({ ...graph, train: { ...graph.train, optimizer } }, data);
    const batch = toBatch(fx);
    const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
    let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
    const once = (i: number) => {
      const r = engine.step(model, state, batch);
      expect(r.diverged).toBe(false);
      state = r.state;
      if (i % 200 === 0) {
        // divergence returns the optimizer state unchanged
        const d = engine.step(model, state, bad);
        expect(d.diverged).toBe(true);
        state = d.state;
      }
    };
    for (let i = 1; i <= 60; i++) once(i);
    const baseline = engine.liveArrays();
    expect(baseline).toBeGreaterThan(0);
    const samples: number[] = [];
    for (let i = 1; i <= 3_000; i++) {
      once(i);
      if (i % 500 === 0) samples.push(engine.liveArrays());
    }
    expect(samples).toEqual(samples.map(() => baseline));
    engine.dispose(state);
    expect(engine.liveArrays()).toBeLessThanOrEqual(2);
  },
  120_000,
);

test('failing calls (params of another model) leak no buffers', async () => {
  const engine = await createEngine();
  const a = loadFixture('mlp_tanh_sgd_5steps');
  const b = loadFixture('two_param');
  const model = engine.compile(a.graph, a.data);
  const foreign = engine.init(engine.compile(b.graph, b.data), 1);
  const batch = toBatch(a.fx);
  const state = engine.startTraining(model, engine.fromHost(model, toHostParams(a.fx.params)), 1);
  const bad = { ...state, params: foreign };
  const fail = () => {
    expect(() => predict0(engine, model, foreign, batch.x, batch.n)).toThrow();
    expect(() => engine.evaluate(model, foreign, batch)).toThrow();
    expect(() => engine.step(model, bad, batch)).toThrow();
  };
  fail();
  const before = engine.liveArrays();
  for (let i = 0; i < 100; i++) fail();
  expect(engine.liveArrays()).toBe(before);
  engine.dispose(state);
  engine.dispose(foreign);
});

test('steps with a weight penalty leave the live buffer count flat', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('penalty_l2');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const once = (i: number) => {
    const r = engine.step(model, state, batch);
    state = r.state;
    expect(r.diverged).toBe(false);
    if (i % 10 === 0) engine.evaluate(model, state.params, batch);
  };
  for (let i = 1; i <= 60; i++) once(i);
  const baseline = engine.liveArrays();
  const samples: number[] = [];
  for (let i = 1; i <= 3000; i++) {
    once(i);
    if (i % 500 === 0) samples.push(engine.liveArrays());
  }
  expect(samples).toEqual(samples.map(() => baseline));
  engine.dispose(state);
}, 60_000);

test.each([
  'loss_log_loss',
  'prim_log_loss_sigmoid',
  'loss_cross_entropy',
  'prim_cross_entropy_softmax',
  'prim_cross_entropy_softmax_k3',
])(
  'compiling %s repeatedly leaves no constants behind',
  async (name) => {
    const engine = await createEngine();
    const { fx, graph, data } = loadFixture(name);
    const batch = toBatch(fx);
    const cycle = () => {
      const model = engine.compile(graph, data);
      let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
      state = engine.step(model, state, batch).state;
      predict0(engine, model, state.params, batch.x, batch.n);
      engine.evaluate(model, state.params, batch);
      engine.dispose(state);
    };
    for (let i = 0; i < 5; i++) cycle();
    const baseline = engine.liveArrays();
    for (let i = 0; i < 200; i++) cycle();
    expect(engine.liveArrays()).toBe(baseline);
  },
  120_000,
);

test('a softmax link shared with another consumer (non-fused cross-entropy) leaves no constants behind', async () => {
  const engine = await createEngine();
  const a = loadFixture('prim_cross_entropy_softmax');
  const shared = loadFixture('prim_link_shared');
  const graph = structuredClone(a.graph);
  const extra = (id: string) => {
    const n = shared.graph.nodes.find((m) => m.id === id);
    if (!n) throw new Error(id);
    return n;
  };
  graph.nodes = [
    ...graph.nodes.filter((n) => n.id !== 'loss'),
    extra('m1'),
    extra('sq'),
    extra('m2'),
    extra('sc'),
    extra('loss'),
  ].map((n) => structuredClone(n));
  graph.edges = [
    ...a.graph.edges.filter((e) => !String(e[0]).startsWith('err') && e[1] !== 'loss'),
    ['err', 'm1'],
    ['p', 'sq'],
    ['sq', 'm2'],
    ['m2', 'sc'],
    ['m1', 'loss.a'],
    ['sc', 'loss.b'],
  ] as typeof graph.edges;
  const batch = toBatch(a.fx);
  const cycle = () => {
    const model = engine.compile(graph, a.data);
    let state = engine.startTraining(model, engine.init(model, 1), 1);
    state = engine.step(model, state, batch).state;
    engine.evaluate(model, state.params, batch);
    engine.dispose(state);
  };
  for (let i = 0; i < 5; i++) cycle();
  const baseline = engine.liveArrays();
  for (let i = 0; i < 200; i++) cycle();
  expect(engine.liveArrays()).toBe(baseline);
}, 120_000);

test('a Tree model (steps, grid predictions, evaluate, grids, trainMany with different leaf counts) leaves the live buffer count flat', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('tree_linear_adam');
  const batch = toBatch(fx);
  const model = engine.compile(graph, data, fitFixtureTrees(graph, data, batch));
  // a second training set whose tree has another leaf count
  const other = {
    ...batch,
    y: Float32Array.from(batch.y, (_, i) => ((batch.x[2 * i] ?? 0) > 0 ? 1 : 0)),
  };
  const axes = [
    { node: 'lin1', tensor: 'W' as const, index: 0 },
    { node: 'lin1', tensor: 'W' as const, index: 1 },
  ] as [{ node: string; tensor: 'W'; index: number }, { node: string; tensor: 'W'; index: number }];
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const grid = Float32Array.from({ length: 2 * 50 }, (_, i) => (i % 7) / 3 - 1);
  const once = (i: number) => {
    const r = engine.step(model, state, batch);
    state = r.state;
    expect(r.diverged).toBe(false);
    if (i % 5 === 0) engine.evaluate(model, state.params, batch);
    if (i % 5 === 0) predict0(engine, model, state.params, grid, 50);
    if (i % 25 === 0) engine.exampleLosses(model, state.params, batch);
    if (i % 50 === 0) {
      engine.lossGrid(
        model,
        state.params,
        batch,
        axes,
        Float32Array.of(0, 1),
        Float32Array.of(0, 1),
      );
      engine.lossSlope(model, state.params, batch, axes[0]);
    }
    if (i % 100 === 0) {
      const many = engine.trainMany(model, [1, 2], [batch, other], 3);
      expect(many[0]?.fits['tree']?.leaves).not.toBe(many[1]?.fits['tree']?.leaves);
    }
  };
  for (let i = 1; i <= 100; i++) once(i);
  const baseline = engine.liveArrays();
  expect(baseline).toBeGreaterThan(0);
  const samples: number[] = [];
  for (let i = 1; i <= 2000; i++) {
    once(i);
    if (i % 400 === 0) samples.push(engine.liveArrays());
  }
  expect(samples).toEqual(samples.map(() => baseline));
  engine.dispose(state);
  engine.disposeModel(model);
}, 120_000);
