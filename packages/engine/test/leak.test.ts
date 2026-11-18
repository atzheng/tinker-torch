// 10,000 training steps with the live-buffer count flat (CONTRACTS.md section 7, docs/spike.md
// section 4). The probe must be installed before jax-js loads, hence the dynamic import.
import { installProbe } from '../src/probe.js';
import { expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams } from './util.js';

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
    if (i % 25 === 0) engine.predict(model, state.params, batch.x, batch.n);
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
    expect(() => engine.predict(model, foreign, batch.x, batch.n)).toThrow();
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
