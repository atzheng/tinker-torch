// Numerical stability of the fused losses (tinker-hj4.2): logits of +-50 and beyond stay finite,
// in the value and in the gradient.
import { beforeAll, describe, expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import type { Engine } from '../src/index.js';
import { loadFixture, toBatch, toHostParams, withLr } from './util.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

/** Scales every weight so the logits are far out in the tails. */
function scaled(name: string, factor: number) {
  const { fx, graph, data } = loadFixture(name);
  const host = toHostParams(fx.params);
  for (const layer of Object.values(host)) {
    layer.W = layer.W.map((w) => w * factor);
    if (layer.b) layer.b = layer.b.map((w) => w * factor);
  }
  return { fx, graph, data, host };
}

describe.each(['loss_log_loss', 'loss_cross_entropy'])('%s with huge logits', (name) => {
  for (const factor of [50, 500]) {
    test(`weights x${factor}: loss, prediction and gradient stay finite`, () => {
      const { fx, graph, data, host } = scaled(name, factor);
      const batch = toBatch(fx);
      const model = engine.compile(withLr(graph, 1), data);
      const params = engine.fromHost(model, host);
      const loss = engine.evaluate(model, params, batch);
      expect(Number.isFinite(loss)).toBe(true);
      const pred = engine.predict(model, params, batch.x, batch.n);
      for (const p of pred) {
        expect(Number.isFinite(p)).toBe(true);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
      const r = engine.step(model, engine.startTraining(model, params, 0), batch);
      expect(r.diverged).toBe(false);
      expect(Number.isFinite(r.loss)).toBe(true);
      for (const layer of Object.values(engine.toHost(r.state.params))) {
        for (const w of layer.W) expect(Number.isFinite(w)).toBe(true);
      }
      engine.dispose(r.state);
    });
  }
});
