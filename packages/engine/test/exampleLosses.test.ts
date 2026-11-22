// Engine.exampleLosses (tinker-6t7.5): the values the Mean at the L pin averages; leak-free.
import { installProbe } from '../src/probe.js';
import { expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');

for (const name of ['loss_log_loss', 'loss_cross_entropy', 'prim_log_loss_sigmoid']) {
  test(`${name}: mean of the per-example values equals evaluate; params borrowed; no leak`, async () => {
    const engine = await createEngine();
    const { fx, graph, data } = loadFixture(name);
    const model = engine.compile(graph, data);
    const batch = toBatch(fx);
    const params = engine.fromHost(model, toHostParams(fx.params));
    const ex = engine.exampleLosses(model, params, batch);
    expect(ex?.length).toBe(batch.n);
    const mean = (ex as Float32Array).reduce((a, b) => a + b, 0) / batch.n;
    expect(mean).toBeCloseTo(engine.evaluate(model, params, batch), 5);
    const before = engine.liveArrays();
    for (let i = 0; i < 50; i++) engine.exampleLosses(model, params, batch);
    expect(engine.liveArrays()).toBe(before);
    engine.dispose(params);
    engine.disposeModel(model);
  });
}
