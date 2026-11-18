import { installProbe } from '../src/probe.js';
import { expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams } from './util.js';
import type { ParamAxis } from '../src/types.js';

installProbe();
const { createEngine } = await import('../src/index.js');

const W: ParamAxis = { node: 'lin1', tensor: 'W', index: 0 };
const B: ParamAxis = { node: 'lin1', tensor: 'b', index: 0 };

test('two_param: lossGrid equals evaluate with the values written', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('two_param');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const params = engine.fromHost(model, toHostParams(fx.params));
  const a = Float32Array.from([-2, -0.5, 0, 1.25, 3]);
  const b = Float32Array.from([-1, 0.3, 2]);
  const grid = engine.lossGrid(model, params, batch, [W, B], a, b);
  expect(grid.length).toBe(15);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      const p = engine.write(
        engine.write(engine.clone(params), 'lin1', { W: Float32Array.of(a[i] ?? 0) }),
        'lin1',
        { b: Float32Array.of(b[j] ?? 0) },
      );
      const want = engine.evaluate(model, p, batch);
      engine.dispose(p);
      expect(grid[i * b.length + j]).toBeCloseTo(want, 4);
    }
  }
  // transposed axes give the transposed grid; the params are untouched
  const t = engine.lossGrid(model, params, batch, [B, W], b, a);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++)
      expect(t[j * a.length + i]).toBeCloseTo(grid[i * b.length + j] ?? NaN, 5);
  expect(engine.toHost(params)['lin1']?.W[0]).toBeCloseTo(fx.params['lin1']?.W[0] ?? NaN, 6);
  engine.dispose(params);
});

test('two_param: an exact overwrite for a huge or non-finite old value', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('two_param');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const a = Float32Array.from([-0.5, 3]);
  const b = Float32Array.from([0.3]);
  for (const bad of [1e8, NaN, Infinity]) {
    const params = engine.fromHost(model, {
      lin1: { W: Float32Array.of(bad), b: Float32Array.of(0) },
    });
    const grid = engine.lossGrid(model, params, batch, [W, B], a, b);
    for (let i = 0; i < a.length; i++) {
      const p = engine.fromHost(model, {
        lin1: { W: Float32Array.of(a[i] ?? 0), b: Float32Array.of(0.3) },
      });
      expect(grid[i]).toBeCloseTo(engine.evaluate(model, p, batch), 4);
      engine.dispose(p);
    }
    engine.dispose(params);
  }
});

test('an MLP: two entries of different layers match evaluate; errors', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const host = toHostParams(fx.params);
  const params = engine.fromHost(model, host);
  const ids = Object.keys(fx.params);
  const n1 = ids[0] ?? '';
  const n2 = ids[ids.length - 1] ?? '';
  const ax: ParamAxis = { node: n1, tensor: 'W', index: 1 };
  const bx: ParamAxis = { node: n2, tensor: 'b', index: 0 };
  const a = Float32Array.from([-1, 0.5, 2]);
  const b = Float32Array.from([-0.7, 0.2]);
  const grid = engine.lossGrid(model, params, batch, [ax, bx], a, b);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 2; j++) {
      const w = Float32Array.from(host[n1]?.W ?? []);
      w[1] = a[i] ?? 0;
      const p = engine.write(engine.write(engine.clone(params), n1, { W: w }), n2, {
        b: Float32Array.of(b[j] ?? 0),
      });
      const want = engine.evaluate(model, p, batch);
      engine.dispose(p);
      expect(grid[i * 2 + j]).toBeCloseTo(want, 4);
    }
  expect(() => engine.lossGrid(model, params, batch, [ax, ax], a, b)).toThrow();
  expect(() => engine.lossGrid(model, params, batch, [ax, { ...bx, index: 99 }], a, b)).toThrow();
  expect(() =>
    engine.lossGrid(model, params, batch, [ax, { ...bx, node: 'nope' }], a, b),
  ).toThrow();
  expect(engine.lossGrid(model, params, batch, [ax, bx], new Float32Array(0), b).length).toBe(0);
  engine.dispose(params);
});

test('50x50 grid timing', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('two_param');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const params = engine.fromHost(model, toHostParams(fx.params));
  const lin = (lo: number, hi: number) =>
    Float32Array.from({ length: 50 }, (_, i) => lo + ((hi - lo) * i) / 49);
  engine.lossGrid(model, params, batch, [W, B], lin(-3, 3), lin(-3, 3)); // trace
  const t0 = performance.now();
  const reps = 10;
  for (let i = 0; i < reps; i++)
    engine.lossGrid(model, params, batch, [W, B], lin(-3, 3), lin(-3, 3));
  const ms = (performance.now() - t0) / reps;
  console.log(`lossGrid 50x50 (two_param): ${ms.toFixed(1)} ms`);
  expect(ms).toBeLessThan(50);
  engine.dispose(params);
});

test('50x50 grid with 500 rows stays fast', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  const model = engine.compile(graph, data);
  const b0 = toBatch(fx);
  const n = 500;
  const batch = {
    n,
    x: Float32Array.from(
      { length: n * (b0.x.length / b0.n) },
      (_, i) => b0.x[i % b0.x.length] ?? 0,
    ),
    y: Float32Array.from({ length: n }, (_, i) => b0.y[i % b0.y.length] ?? 0),
  };
  const params = engine.fromHost(model, toHostParams(fx.params));
  const ids = Object.keys(fx.params);
  const axes: [ParamAxis, ParamAxis] = [
    { node: ids[0] ?? '', tensor: 'W', index: 0 },
    { node: ids[ids.length - 1] ?? '', tensor: 'b', index: 0 },
  ];
  const g = Float32Array.from({ length: 50 }, (_, i) => -2 + (4 * i) / 49);
  engine.lossGrid(model, params, batch, axes, g, g);
  const t0 = performance.now();
  engine.lossGrid(model, params, batch, axes, g, g);
  const ms = performance.now() - t0;
  console.log(`lossGrid 50x50 (mlp, n=500): ${ms.toFixed(1)} ms`);
  expect(ms).toBeLessThan(250);
  engine.dispose(params);
});

test.each(['two_param', 'mlp_tanh_sgd_5steps'])(
  '%s: 200 compiled models with lossGrid and dispose leave the live buffer count flat',
  async (name) => {
    const engine = await createEngine();
    const { fx, graph, data } = loadFixture(name);
    const batch = toBatch(fx);
    const ids = Object.keys(fx.params);
    const first = ids[0] ?? '';
    const last = ids[ids.length - 1] ?? '';
    const axes: [ParamAxis, ParamAxis] = [
      { node: first, tensor: 'W', index: 0 },
      { node: last, tensor: 'b', index: 0 },
    ];
    const a = Float32Array.from([-1, 0, 1]);
    const cycle = () => {
      const model = engine.compile(graph, data);
      const params = engine.fromHost(model, toHostParams(fx.params));
      engine.lossGrid(model, params, batch, axes, a, a);
      engine.dispose(params);
    };
    for (let i = 0; i < 5; i++) cycle();
    const baseline = engine.liveArrays();
    for (let i = 0; i < 200; i++) cycle();
    expect(engine.liveArrays()).toBe(baseline);
    // failing calls leak nothing either
    const model = engine.compile(graph, data);
    const params = engine.fromHost(model, toHostParams(fx.params));
    const before = engine.liveArrays();
    for (let i = 0; i < 20; i++)
      expect(() => engine.lossGrid(model, params, batch, [axes[0], axes[0]], a, a)).toThrow();
    expect(engine.liveArrays()).toBe(before);
    engine.dispose(params);
  },
  120_000,
);

test('the probe sees the buffers of the lossGrid tests (mlp fixture)', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  const model = engine.compile(graph, data);
  const before = engine.liveArrays();
  const params = engine.fromHost(model, toHostParams(fx.params));
  expect(engine.liveArrays()).toBeGreaterThan(before);
  engine.dispose(params);
});
