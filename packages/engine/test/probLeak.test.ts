// Leak test for `probabilityRange` (tinker-ddg). The probe must be installed before jax-js loads,
// hence the dynamic import (same as leak.test.ts).
import { installProbe } from '../src/probe.js';
import { expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');

test.each(['prob_direct_log_loss', 'prob_direct_cross_entropy'])(
  '%s: probabilityRange (steps, fresh params, a throwing call) leaves the live buffer count flat',
  async (name) => {
    const engine = await createEngine();
    const { fx, graph, data } = loadFixture(name);
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
    const once = () => {
      state = engine.step(model, state, batch).state;
      expect(engine.probabilityRange(model, state.params, batch)).toHaveLength(1);
      const other = engine.fromHost(model, toHostParams(fx.params));
      engine.probabilityRange(model, other, batch);
      engine.dispose(other);
      // a batch that does not fit throws before anything is allocated
      const bad = engine.fromHost(model, toHostParams(fx.params));
      expect(() => engine.probabilityRange(model, bad, { ...batch, n: batch.n + 1 })).toThrow();
      engine.dispose(bad);
    };
    for (let i = 0; i < 20; i++) once();
    const baseline = engine.liveArrays();
    expect(baseline).toBeGreaterThan(0);
    for (let i = 0; i < 500; i++) once();
    expect(engine.liveArrays()).toBe(baseline);
    engine.dispose(state);
    engine.disposeModel(model);
  },
  60_000,
);
