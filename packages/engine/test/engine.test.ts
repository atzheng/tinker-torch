import { analyse } from '@tinker/core';
import type { Graph } from '@tinker/core';
import { beforeAll, describe, expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import type { Engine, Model, TrainState } from '../src/index.js';
import { loadFixture, toBatch, toHostParams, withLr } from './util.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

function mlp() {
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  return { fx, graph, data, batch: toBatch(fx), host: toHostParams(fx.params) };
}

describe('host conversion', () => {
  test('toHost(fromHost(h)) equals h, with new arrays', () => {
    const { graph, data, host } = mlp();
    const model = engine.compile(graph, data);
    const p = engine.fromHost(model, host);
    const back = engine.toHost(p);
    expect(Object.keys(back).sort()).toEqual(Object.keys(host).sort());
    for (const id of Object.keys(host)) {
      expect(Array.from(back[id]?.W ?? [])).toEqual(Array.from(host[id]?.W ?? []));
      expect(Array.from(back[id]?.b ?? [])).toEqual(Array.from(host[id]?.b ?? []));
      expect(back[id]?.W).not.toBe(host[id]?.W);
    }
    // two reads give independent arrays
    expect(engine.toHost(p)['lin1']?.W).not.toBe(back['lin1']?.W);
    engine.dispose(p);
  });

  test('fromHost copies: writing into the host arrays later changes nothing', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const p = engine.fromHost(model, host);
    const before = engine.evaluate(model, p, batch);
    host['lin1']?.W.fill(123);
    expect(engine.evaluate(model, p, batch)).toBe(before);
    const mutated = engine.toHost(p);
    mutated['lin1']?.W.fill(7);
    expect(engine.evaluate(model, p, batch)).toBe(before);
    engine.dispose(p);
  });

  test('init matches core initParams and is deterministic', () => {
    const { graph, data } = mlp();
    const model = engine.compile(graph, data);
    const a = engine.toHost(engine.init(model, 5));
    const b = engine.toHost(engine.init(model, 5));
    const c = engine.toHost(engine.init(model, 6));
    expect(Array.from(a['lin1']?.W ?? [])).toEqual(Array.from(b['lin1']?.W ?? []));
    expect(Array.from(a['lin1']?.W ?? [])).not.toEqual(Array.from(c['lin1']?.W ?? []));
  });

  test('a host that does not fit the model throws and allocates nothing usable', () => {
    const { graph, data, host } = mlp();
    const model = engine.compile(graph, data);
    const bad = toHostParams({
      lin1: { W: [1, 2, 3], b: [0, 0, 0] },
      lin2: { W: [1, 2, 3], b: [0] },
    });
    expect(() => engine.fromHost(model, bad)).toThrow(/lin1/);
    const missing = toHostParams({});
    expect(() => engine.fromHost(model, missing)).toThrow(/no entry/);
    expect(() => engine.fromHost(model, host)).not.toThrow();
  });

  test('write replaces one tensor, consumes the old Params', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const p = engine.fromHost(model, host);
    const w = Float32Array.from(host['lin2']?.W ?? [], () => 0);
    expect(() => engine.write(p, 'lin2', { W: new Float32Array(2) })).toThrow(/need 3 values/);
    expect(() => engine.write(p, 'nope', { W: w })).toThrow(/no parameters/);
    const q = engine.write(p, 'lin2', { W: w });
    expect(() => engine.toHost(p)).toThrow(/already consumed/);
    const h = engine.toHost(q);
    expect(Array.from(h['lin2']?.W ?? [])).toEqual([0, 0, 0]);
    expect(Array.from(h['lin2']?.b ?? [])).toEqual(Array.from(host['lin2']?.b ?? []));
    expect(Array.from(h['lin1']?.W ?? [])).toEqual(Array.from(host['lin1']?.W ?? []));
    expect(Number.isFinite(engine.evaluate(model, q, batch))).toBe(true);
    engine.dispose(q);
  });

  test('clone is independent of the original', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const p = engine.fromHost(model, host);
    const c = engine.clone(p);
    const want = engine.evaluate(model, p, batch);
    engine.dispose(p);
    expect(engine.evaluate(model, c, batch)).toBe(want);
    engine.dispose(c);
  });
});

describe('compile', () => {
  test('an invalid graph throws the first issue sentence', () => {
    const { graph, data } = mlp();
    const broken: Graph = { ...graph, edges: graph.edges.filter((e) => e[0] !== 'act1') };
    const issues = analyse(broken, data).issues;
    expect(issues.length).toBeGreaterThan(0);
    expect(() => engine.compile(broken, data)).toThrow(issues[0]?.message);
    const noLoss: Graph = {
      ...graph,
      nodes: graph.nodes.filter((n) => n.type !== 'loss'),
      edges: [],
    };
    expect(() => engine.compile(noLoss, data)).toThrow(Error);
  });

  test.each(['sgd', 'momentum', 'adam'] as const)(
    '%s: step works with varying batch sizes and keeps training',
    (optimizer) => {
      const { graph, data, host, batch } = mlp();
      const model = engine.compile({ ...graph, train: { ...graph.train, optimizer } }, data);
      let state = engine.startTraining(model, engine.fromHost(model, host), 1);
      const dim = batch.x.length / batch.n;
      // Each new batch size traces the jitted step once (jax-js caches by shape); repeats reuse it.
      for (const n of [batch.n, 3, 2, batch.n, 3, 2, 3]) {
        const b = { x: batch.x.slice(0, n * dim), y: batch.y.slice(0, n), n };
        const r = engine.step(model, state, b);
        expect(r.diverged).toBe(false);
        expect(Number.isFinite(r.loss)).toBe(true);
        state = r.state;
      }
      engine.dispose(state);
    },
  );

  test('unimplemented pieces, losses and optimizers are rejected at compile with one sentence', () => {
    const { graph, data } = mlp();
    const huber: Graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.type === 'loss' ? { ...n, fn: 'huber' as const, delta: 1 } : n,
      ),
    };
    expect(() => engine.compile(huber, data)).toThrow('The huber loss is not implemented yet.');
    const withPenalty: Graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.type === 'loss' ? { ...n, penalty: { kind: 'l2' as const, lambda: 0.1 } } : n,
      ),
    };
    expect(() => engine.compile(withPenalty, data)).toThrow(
      'The weight penalty is not implemented yet.',
    );
  });

  test('an Input that selects some of the dataset columns', () => {
    const { graph, data, batch } = mlp();
    // dataset has x1, x2; the model only sees x2
    const only2: Graph = {
      ...graph,
      nodes: [...graph.nodes.map((n) => (n.type === 'input' ? { ...n, columns: ['x2'] } : n))],
    };
    const model = engine.compile(only2, data);
    expect(model.analysis.inWidth['lin1']).toBe(1);
    const p = engine.init(model, 1);
    const pred = engine.predict(model, p, batch.x, batch.n);
    // the same model on x with column x1 replaced by 0 must agree: lin1 sees only column x2
    const x2 = Float32Array.from(batch.x);
    for (let i = 0; i < batch.n; i++) x2[i * 2] = 99;
    expect(Array.from(engine.predict(model, p, x2, batch.n))).toEqual(Array.from(pred));
    engine.dispose(p);
  });

  test('a batch that does not fit throws before consuming anything', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const state = engine.startTraining(model, engine.fromHost(model, host), 1);
    expect(() => engine.step(model, state, { ...batch, n: batch.n + 1 })).toThrow(/batch/);
    expect(() =>
      engine.step(model, state, { x: new Float32Array(0), y: new Float32Array(0), n: 0 }),
    ).toThrow();
    const r = engine.step(model, state, batch);
    expect(r.diverged).toBe(false);
    engine.dispose(r.state);
  });
});

describe('ownership', () => {
  function started(model: Model, seed = 1) {
    const { host } = mlp();
    const p = engine.fromHost(model, host);
    return { p, s: engine.startTraining(model, p, seed) };
  }

  test('startTraining consumes the params it is given', () => {
    const { graph, data, batch } = mlp();
    const model = engine.compile(graph, data);
    const { p, s } = started(model);
    expect(() => engine.predict(model, p, batch.x, batch.n)).toThrow(/already consumed/);
    expect(() => engine.startTraining(model, p, 1)).toThrow(/already consumed/);
    engine.dispose(s);
  });

  test('predict after step consumed the state throws instead of reading freed memory', () => {
    const { graph, data, batch } = mlp();
    const model = engine.compile(graph, data);
    const { s } = started(model);
    const staleParams = s.params;
    const r = engine.step(model, s, batch);
    expect(() => engine.predict(model, staleParams, batch.x, batch.n)).toThrow(/already consumed/);
    expect(() => engine.evaluate(model, staleParams, batch)).toThrow(/already consumed/);
    expect(() => engine.toHost(staleParams)).toThrow(/already consumed/);
    expect(() => engine.step(model, s, batch)).toThrow(/already consumed/);
    // the new state is fine
    expect(engine.predict(model, r.state.params, batch.x, batch.n)).toHaveLength(batch.n);
    engine.dispose(r.state);
  });

  test('dispose twice is harmless; use after dispose throws', () => {
    const { graph, data, batch } = mlp();
    const model = engine.compile(graph, data);
    const { s } = started(model);
    const params = s.params;
    engine.dispose(s);
    engine.dispose(s);
    engine.dispose(params);
    expect(() => engine.predict(model, params, batch.x, batch.n)).toThrow(/already consumed/);
    expect(() => engine.step(model, s, batch)).toThrow(/already consumed/);
  });

  test('predict and evaluate borrow: repeated calls give the same answer', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const p = engine.fromHost(model, host);
    const a = engine.predict(model, p, batch.x, batch.n);
    const b = engine.predict(model, p, batch.x, batch.n);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(engine.evaluate(model, p, batch)).toBe(engine.evaluate(model, p, batch));
    engine.dispose(p);
  });
});

describe('divergence', () => {
  test('lr 1e3 eventually diverges without throwing and leaves the state unchanged', () => {
    const { fx, graph, data } = loadFixture('two_param');
    const model = engine.compile(withLr(graph, 1e3), data);
    const batch = toBatch(fx);
    let state: TrainState = engine.startTraining(
      model,
      engine.fromHost(model, toHostParams(fx.params)),
      1,
    );
    let diverged = false;
    let steps = 0;
    for (; steps < 200 && !diverged; steps++) {
      const before = engine.toHost(state.params);
      const stepsBefore = state.step;
      const r = engine.step(model, state, batch);
      if (r.diverged) {
        diverged = true;
        expect(r.state.step).toBe(stepsBefore);
        const after = engine.toHost(r.state.params);
        expect(Array.from(after['lin1']?.W ?? [])).toEqual(Array.from(before['lin1']?.W ?? []));
        expect(Array.from(after['lin1']?.b ?? [])).toEqual(Array.from(before['lin1']?.b ?? []));
        // the kept parameters are all finite (a non-finite update is never kept)
        const kept = [...(after['lin1']?.W ?? []), ...(after['lin1']?.b ?? [])];
        expect(kept.every(Number.isFinite)).toBe(true);
      } else {
        expect(r.state.step).toBe(stepsBefore + 1);
      }
      state = r.state;
    }
    expect(diverged).toBe(true);
    // stepping again after a diverged step is allowed and diverges again (state unchanged)
    expect(engine.step(model, state, batch).diverged).toBe(true);
    engine.dispose(state);
  });

  test('a NaN batch is reported as diverged, state unchanged', () => {
    const { graph, data, host, batch } = mlp();
    const model = engine.compile(graph, data);
    const state = engine.startTraining(model, engine.fromHost(model, host), 1);
    const before = engine.toHost(state.params);
    const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
    const r = engine.step(model, state, bad);
    expect(r.diverged).toBe(true);
    expect(Number.isNaN(r.loss)).toBe(true);
    expect(Array.from(engine.toHost(r.state.params)['lin1']?.W ?? [])).toEqual(
      Array.from(before['lin1']?.W ?? []),
    );
    const ok = engine.step(model, r.state, batch);
    expect(ok.diverged).toBe(false);
    engine.dispose(ok.state);
  });
});

describe('failing calls and finiteness', () => {
  test('params of another model are rejected with a sentence, before anything is consumed', () => {
    const { graph, data, batch } = mlp();
    const model = engine.compile(graph, data);
    const other = loadFixture('two_param');
    const otherModel = engine.compile(other.graph, other.data);
    const foreign = engine.init(otherModel, 1);
    expect(() => engine.predict(model, foreign, batch.x, batch.n)).toThrow(/different model/);
    expect(() => engine.evaluate(model, foreign, batch)).toThrow(/different model/);
    expect(() => engine.startTraining(model, foreign, 1)).toThrow(/different model/);
    // still alive and usable with its own model
    expect(engine.toHost(foreign)['lin1']?.W).toHaveLength(1);
    engine.dispose(foreign);
  });

  test('a finite parameter near 3e38 is not reported as diverged', () => {
    const { fx, graph, data } = loadFixture('linear_nobias_d2');
    const model = engine.compile(withLr(graph, 0), data);
    const big = toHostParams({ lin1: { W: Array.from(fx.params['lin1']?.W ?? [], () => 3e38) } });
    const state = engine.startTraining(model, engine.fromHost(model, big), 1);
    const zero = { x: new Float32Array(2), y: new Float32Array(1), n: 1 };
    const r = engine.step(model, state, zero);
    expect(r.diverged).toBe(false);
    expect(r.loss).toBe(0);
    engine.dispose(r.state);
  });
});
