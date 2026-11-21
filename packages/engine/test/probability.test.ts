// Unfused log loss / cross-entropy (tinker-ddg): no clipping, `probabilityRange`, SGD on a bare probability.
import { beforeAll, describe, expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import type { Engine } from '../src/index.js';
import { loadFixture, toBatch, toHostParams } from './util.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

describe('probabilityRange', () => {
  test('the unwired Bias: min = max = b, in and out of range, params borrowed', () => {
    const { fx, graph, data } = loadFixture('prob_direct_log_loss');
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    for (const b of [0.3, 1.3, -0.05, 0, 1]) {
      const params = engine.fromHost(model, toHostParams({ b: { b: [b] } }));
      const r = engine.probabilityRange(model, params, batch);
      expect(r).toHaveLength(1);
      expect(r[0]?.node).toBe('err');
      expect(r[0]?.min).toBeCloseTo(b, 6);
      expect(r[0]?.max).toBeCloseTo(b, 6);
      // the params are still usable afterwards
      expect(engine.toHost(params)['b']?.b?.[0]).toBeCloseTo(b, 6);
      engine.dispose(params);
    }
    engine.disposeModel(model);
  });

  test('cross-entropy on sigmoid outputs: min and max over every entry; fused graphs report nothing', () => {
    const { fx, graph, data } = loadFixture('prob_direct_cross_entropy');
    const model = engine.compile(graph, data);
    const params = engine.fromHost(model, toHostParams(fx.params));
    const batch = toBatch(fx);
    const [r] = engine.probabilityRange(model, params, batch);
    expect(r?.node).toBe('err');
    expect(r?.min).toBeGreaterThan(0);
    expect(r?.max).toBeLessThan(1);
    expect(r?.min).toBeLessThan(r?.max ?? 0);
    // pushing one output unit's bias far up takes the max past 1
    const host = toHostParams(fx.params);
    const b = Float32Array.from(host['lin1']?.b ?? []);
    b[0] = 50;
    const up = engine.fromHost(model, { ...host, lin1: { ...host['lin1'], b } });
    expect(engine.probabilityRange(model, up, batch)[0]?.max).toBeCloseTo(1, 6);
    engine.dispose(up);
    engine.dispose(params);
    engine.disposeModel(model);

    const f = loadFixture('prim_log_loss_sigmoid');
    const fm = engine.compile(f.graph, f.data);
    const fp = engine.fromHost(fm, toHostParams(f.fx.params));
    expect(engine.probabilityRange(fm, fp, toBatch(f.fx))).toEqual([]);
    engine.dispose(fp);
    engine.disposeModel(fm);
  });

  test('a graph with no probability loss reports nothing', () => {
    const { fx, graph, data } = loadFixture('intercept_only_mse');
    const model = engine.compile(graph, data);
    const params = engine.fromHost(model, toHostParams(fx.params));
    expect(engine.probabilityRange(model, params, toBatch(fx))).toEqual([]);
    engine.dispose(params);
    engine.disposeModel(model);
  });
});

describe('range checked on every step', () => {
  /** Linear(3) straight into cross-entropy (k = 3 classes, no softmax, no sigmoid). */
  function bareCrossEntropy() {
    const { fx, graph, data } = loadFixture('prob_direct_cross_entropy');
    const g = {
      ...graph,
      nodes: graph.nodes.filter((n) => n.id !== 'a1'),
      edges: graph.edges
        .filter((e) => e[1] !== 'a1')
        .map((e) => (e[0] === 'a1' ? (['lin1', e[1]] as typeof e) : e)),
    };
    // the Linear's output is the probability row: W = 0, so the output is b
    const host = (b: number[]) => toHostParams({ lin1: { W: new Array<number>(6).fill(0), b } });
    return { fx, graph: g, data, host };
  }

  test('cross-entropy with p[class] above 1 (a finite negative loss) stops the step, update not applied', () => {
    const { fx, graph, data, host } = bareCrossEntropy();
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    // inside (0, 1): trains
    let st = engine.startTraining(model, engine.fromHost(model, host([0.3, 0.4, 0.5])), 1);
    const ok = engine.step(model, st, batch);
    expect(ok.diverged).toBe(false);
    expect(ok.ranges?.[0]?.min).toBeCloseTo(0.3, 5);
    expect(ok.ranges?.[0]?.max).toBeCloseTo(0.5, 5);
    engine.dispose(ok.state);
    // one entry above 1 (not necessarily the picked one): the loss may be finite, the step is refused
    st = engine.startTraining(model, engine.fromHost(model, host([0.3, 1.5, 0.5])), 1);
    const before = engine.toHost(st.params).lin1?.b?.slice();
    const bad = engine.step(model, st, batch);
    expect(bad.diverged).toBe(true);
    expect(bad.ranges?.[0]?.max).toBeCloseTo(1.5, 5);
    expect(Array.from(engine.toHost(bad.state.params).lin1?.b ?? [])).toEqual(
      Array.from(before ?? []),
    );
    expect(bad.state.step).toBe(0);
    engine.dispose(bad.state);
    // all entries above 1 and the picked one too: loss finite and negative, still refused
    st = engine.startTraining(model, engine.fromHost(model, host([1.5, 1.5, 1.5])), 1);
    const neg = engine.step(model, st, batch);
    expect(Number.isFinite(neg.loss) && neg.loss < 0).toBe(true);
    expect(neg.diverged).toBe(true);
    engine.dispose(neg.state);
    engine.disposeModel(model);
  });

  test('log loss: the step that sees p outside (0, 1) is refused with the range; a model with no probability loss has no ranges', () => {
    const { fx, graph, data } = loadFixture('prob_direct_log_loss');
    const model = engine.compile(graph, data);
    const st = engine.startTraining(
      model,
      engine.fromHost(model, toHostParams({ b: { b: [1.3] } })),
      1,
    );
    const r = engine.step(model, st, toBatch(fx));
    expect(r.diverged).toBe(true);
    expect(r.ranges?.[0]?.max).toBeCloseTo(1.3, 5);
    engine.dispose(r.state);
    engine.disposeModel(model);

    const f = loadFixture('mlp_tanh_sgd_5steps');
    const m2 = engine.compile(f.graph, f.data);
    const s2 = engine.startTraining(m2, engine.fromHost(m2, toHostParams(f.fx.params)), 1);
    const r2 = engine.step(m2, s2, toBatch(f.fx));
    expect(r2.ranges).toBeUndefined();
    engine.dispose(r2.state);
    engine.disposeModel(m2);
  });
});

describe('no clipping', () => {
  test('a pred outside (0, 1) is a non-finite loss and a diverged step with the state unchanged', () => {
    const { fx, graph, data } = loadFixture('prob_direct_log_loss');
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    for (const b of [1.3, -0.05, 0, 1]) {
      const params = engine.fromHost(model, toHostParams({ b: { b: [b] } }));
      expect(Number.isFinite(engine.evaluate(model, params, batch)), `b = ${b}`).toBe(false);
      const r = engine.step(model, engine.startTraining(model, params, 1), batch);
      expect(r.diverged).toBe(true);
      expect(engine.toHost(r.state.params)['b']?.b?.[0]).toBeCloseTo(b, 6);
      engine.dispose(r.state);
    }
    engine.disposeModel(model);
  });

  test('SGD from p = 0.3 converges to the mean of the targets', () => {
    const { fx, graph, data } = loadFixture('prob_direct_log_loss');
    const model = engine.compile({ ...graph, train: { ...graph.train, lr: 0.1 } }, data);
    const batch = toBatch(fx);
    const mean = batch.y.reduce((a, v) => a + v, 0) / batch.n;
    let state = engine.startTraining(
      model,
      engine.fromHost(model, toHostParams({ b: { b: [0.3] } })),
      1,
    );
    for (let i = 0; i < 300; i++) {
      const r = engine.step(model, state, batch);
      expect(r.diverged).toBe(false);
      state = r.state;
    }
    expect(engine.toHost(state.params)['b']?.b?.[0]).toBeCloseTo(mean, 3);
    engine.dispose(state);
    engine.disposeModel(model);
  });
});
