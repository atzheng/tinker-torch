// Graph v2 behaviour of the engine (tinker-8xw.3): pins, fusion, taps, no-prediction graphs,
// leaks, lossGrid and trainMany on primitive graphs. Values against PyTorch are in reference.test.ts.
import { installProbe } from '../src/probe.js';
import { tree } from '@jax-js/jax';
import { analyse } from '@tinker/core';
import { hostToTree, treeToHost } from '../src/convert.js';
import { describe, expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams, withLr } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

/** Scales every parameter so the logits are far out in the tails. */
function scaledHost(name: string, factor: number) {
  const { fx, graph, data } = loadFixture(name);
  const host = toHostParams(fx.params);
  for (const layer of Object.values(host)) {
    layer.W = layer.W?.map((w) => w * factor) ?? new Float32Array(0);
    if (layer.b) layer.b = layer.b.map((w) => w * factor);
  }
  return { fx, graph, data, host, batch: toBatch(fx) };
}

describe('stable fusion', () => {
  test.each(['prim_log_loss_sigmoid', 'prim_cross_entropy_softmax', 'prim_link_shared'])(
    '%s: analyse lists the fusable pair',
    (name) => {
      const { graph, data } = loadFixture(name);
      expect(analyse(graph, data).fusable.length).toBe(1);
    },
  );

  // Logits at +-1000 and beyond: the semantic sigmoid -> log_loss would give inf or NaN.
  describe.each([
    'prim_log_loss_sigmoid',
    'prim_cross_entropy_softmax',
    'prim_link_shared',
    'loss_log_loss',
    'loss_cross_entropy',
  ])('%s with logits far in the tails', (name) => {
    for (const factor of [500, 5000]) {
      test(`parameters x${factor}: loss, prediction, gradient and step stay finite`, () => {
        const { graph, data, host, batch } = scaledHost(name, factor);
        const model = engine.compile(withLr(graph, 1), data);
        const params = engine.fromHost(model, host);
        const loss = engine.evaluate(model, params, batch);
        expect(Number.isFinite(loss)).toBe(true);
        for (const p of engine.predict(model, params, batch.x, batch.n)[0] ?? []) {
          expect(Number.isFinite(p)).toBe(true);
          expect(p).toBeGreaterThanOrEqual(0);
          expect(p).toBeLessThanOrEqual(1);
        }
        const r = engine.step(model, engine.startTraining(model, params, 0), batch);
        expect(r.diverged).toBe(false);
        expect(Number.isFinite(r.loss)).toBe(true);
        for (const layer of Object.values(engine.toHost(r.state.params))) {
          for (const w of layer.W ?? []) expect(Number.isFinite(w)).toBe(true);
        }
        engine.dispose(r.state);
        engine.disposeModel(model);
      });
    }
  });

  test('a confidently wrong prediction has the exact logit-sized loss', () => {
    // one example, logit z = 1000, label 0: loss = softplus(1000) = 1000
    const { graph, data } = loadFixture('prim_log_loss_sigmoid');
    const model = engine.compile(graph, data);
    const host = toHostParams({
      lin1: { W: [0, 0, 0, 0, 0, 0], b: [0, 0, 0] },
      lin2: { W: [0, 0, 0], b: [1000] },
    });
    const params = engine.fromHost(model, host);
    const batch = { x: Float32Array.of(0.5, -0.5), y: Float32Array.of(0), n: 1 };
    expect(engine.evaluate(model, params, batch)).toBeCloseTo(1000, 3);
    engine.dispose(params);
  });
});

describe('pins', () => {
  test('a graph with no prediction pin trains and predicts nothing', () => {
    const { fx, graph, data } = loadFixture('prim_no_prediction');
    expect(graph.predictions).toEqual([]);
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    const params = engine.fromHost(model, toHostParams(fx.params));
    expect(engine.predict(model, params, batch.x, batch.n)).toEqual([]);
    let state = engine.startTraining(model, params, 1);
    const first = engine.step(model, state, batch);
    state = first.state;
    let last = first.loss;
    for (let i = 0; i < 60; i++) {
      const r = engine.step(model, state, batch);
      state = r.state;
      last = r.loss;
    }
    expect(last).toBeLessThan(first.loss);
    engine.dispose(state);
  });

  test('predict returns one array per pin, in pin order', () => {
    const { fx, graph, data } = loadFixture('prim_two_predictions');
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    const params = engine.fromHost(model, toHostParams(fx.params));
    const out = engine.predict(model, params, batch.x, batch.n);
    expect(out.length).toBe(2);
    expect(out.map((o) => o.length)).toEqual([batch.n, batch.n]);
    engine.dispose(params);
  });

  test('a Linear read only through a Weights tap is initialised, trained and listed', () => {
    const { fx, graph, data } = loadFixture('prim_weights_tap_only');
    const model = engine.compile(graph, data);
    expect(model.analysis.paramNodes).toContain('tap');
    expect(model.analysis.order).not.toContain('tap');
    const host = engine.toHost(engine.init(model, 3));
    expect(Object.keys(host).sort()).toEqual(['lin1', 'lin2', 'tap']);
    // the L2 penalty on `tap` shrinks its weights while nothing else touches them
    const batch = toBatch(fx);
    let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
    const before = engine.toHost(state.params)['tap']?.W ?? new Float32Array(0);
    for (let i = 0; i < 20; i++) state = engine.step(model, state, batch).state;
    const after = engine.toHost(state.params)['tap']?.W ?? new Float32Array(0);
    const sq = (a: Float32Array) => a.reduce((s, v) => s + v * v, 0);
    expect(sq(after)).toBeLessThan(sq(before));
    engine.dispose(state);
  });

  test('an untrainable graph throws the analysis sentence', () => {
    const { graph, data } = loadFixture('prim_square_mean');
    expect(() => engine.compile({ ...graph, loss: null }, data)).toThrow(/No loss is pinned/);
  });
});

describe('lossGrid and trainMany on primitive graphs', () => {
  test('lossGrid over a Weights-tapped Linear matches evaluate', () => {
    const { fx, graph, data } = loadFixture('prim_weights_tap_only');
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    const host = toHostParams(fx.params);
    const params = engine.fromHost(model, host);
    const axes = [
      { node: 'tap', tensor: 'W' as const, index: 0 },
      { node: 'lin2', tensor: 'b' as const, index: 0 },
    ] satisfies [unknown, unknown];
    const a = Float32Array.of(-1, 0.5);
    const b = Float32Array.of(0.25, 2);
    const grid = engine.lossGrid(model, params, batch, axes, a, b);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const h = toHostParams(fx.params);
        const tap = h['tap'];
        const l2 = h['lin2'];
        if (!tap?.W || !l2?.b) throw new Error('fixture changed');
        tap.W[0] = a[i] ?? 0;
        l2.b[0] = b[j] ?? 0;
        const p = engine.fromHost(model, h);
        expect(grid[i * 2 + j]).toBeCloseTo(engine.evaluate(model, p, batch), 4);
        engine.dispose(p);
      }
    }
    engine.dispose(params);
  });

  test.each(['prim_weights_tap_only', 'prim_link_shared', 'prim_no_prediction'])(
    'trainMany copy equals a plain run: %s',
    (name) => {
      const { fx, graph, data } = loadFixture(name);
      const model = engine.compile(graph, data);
      const batch = toBatch(fx);
      const res = engine.trainMany(model, [4, 9], [batch, batch], 8);
      expect(res.map((r) => r.diverged)).toEqual([false, false]);
      for (const [i, seed] of [4, 9].entries()) {
        let state = engine.startTraining(model, engine.init(model, seed), seed);
        for (let k = 0; k < 8; k++) state = engine.step(model, state, batch).state;
        const want = engine.toHost(state.params);
        for (const id of Object.keys(want)) {
          const g = res[i]?.params[id]?.W ?? new Float32Array(0);
          g.forEach((v, k) => expect(v).toBeCloseTo(want[id]?.W?.[k] ?? NaN, 4));
        }
        engine.dispose(state);
      }
    },
  );
});

describe('leaks', () => {
  const names = [
    'prim_weights_tap_only',
    'prim_two_predictions',
    'prim_link_shared',
    'prim_no_prediction',
    'prim_dag_dropout_skip',
  ];

  test.each(names)(
    '%s: 2,000 steps with predict and evaluate leave the buffer count flat',
    (name) => {
      const { fx, graph, data } = loadFixture(name);
      const model = engine.compile(graph, data);
      const batch = toBatch(fx);
      const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
      let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
      const once = (i: number) => {
        const r = engine.step(model, state, i % 400 === 0 ? bad : batch);
        state = r.state;
        if (i % 10 === 0) engine.evaluate(model, state.params, batch);
        if (i % 25 === 0) engine.predict(model, state.params, batch.x, batch.n);
      };
      for (let i = 1; i <= 60; i++) once(i);
      const baseline = engine.liveArrays();
      expect(baseline).toBeGreaterThan(0);
      for (let i = 1; i <= 2000; i++) {
        once(i);
        if (i % 500 === 0) expect(engine.liveArrays(), `step ${i}`).toBe(baseline);
      }
      engine.dispose(state);
      engine.disposeModel(model);
    },
    120_000,
  );

  test('200 compile / train / dispose cycles over primitive graphs leave the buffer count flat', () => {
    const loaded = names.map((n) => loadFixture(n));
    const cycle = (k: number) => {
      const { fx, graph, data } = loaded[k % loaded.length] as ReturnType<typeof loadFixture>;
      const model = engine.compile(graph, data);
      const batch = toBatch(fx);
      let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
      for (let i = 0; i < 2; i++) state = engine.step(model, state, batch).state;
      engine.predict(model, state.params, batch.x, batch.n);
      engine.evaluate(model, state.params, batch);
      engine.dispose(state);
      engine.disposeModel(model);
    };
    for (let i = 0; i < 10; i++) cycle(i);
    const baseline = engine.liveArrays();
    for (let i = 0; i < 200; i++) cycle(i);
    expect(engine.liveArrays()).toBe(baseline);
  }, 120_000);
});

describe('generic named parameter tensors (stub owner)', () => {
  // No piece owns parameters besides Linear yet: exercise the tree conversions directly.
  const shapes = new Map([['stub', { g: [2, 3], s: [1] }]]);
  const host = {
    stub: { g: Float32Array.from([1, 2, 3, 4, 5, 6]), s: Float32Array.from([7]) },
  };

  test('hostToTree / treeToHost round trip and mismatches throw', () => {
    const t = hostToTree(shapes, host);
    expect(t['stub']?.['g']?.shape).toEqual([2, 3]);
    const back = treeToHost(t);
    expect(Array.from(back['stub']?.['g'] ?? [])).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(back['stub']?.['s'] ?? [])).toEqual([7]);
    tree.dispose(t);
    expect(() => hostToTree(shapes, { stub: { g: new Float32Array(5), s: host.stub.s } })).toThrow(
      /expected 6/,
    );
    expect(() => hostToTree(shapes, { stub: { g: host.stub.g } })).toThrow(/expected 1/);
    expect(() =>
      hostToTree(shapes, { stub: { ...host.stub, extra: new Float32Array(1) } }),
    ).toThrow(/"extra"/);
  });
});
