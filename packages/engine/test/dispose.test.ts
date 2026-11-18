// disposeModel (CONTRACTS.md section 7, tinker-klv): frees the jitted functions, throws on later use.
import { installProbe } from '../src/probe.js';
import { describe, expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

const SENTENCE = 'This model was disposed.';

function setup(name: string) {
  const { fx, graph, data } = loadFixture(name);
  const batch = toBatch(fx);
  const linear = graph.nodes.filter((n) => n.type === 'linear').map((n) => n.id);
  const axes = [
    { node: linear[0] as string, tensor: 'W' as const, index: 0 },
    { node: linear[0] as string, tensor: 'W' as const, index: 1 },
  ] satisfies [unknown, unknown];
  return { fx, graph, data, batch, axes };
}

/** One full cycle touching every jitted function the model builds. Returns nothing alive. */
function cycle(s: ReturnType<typeof setup>): void {
  const model = engine.compile(s.graph, s.data);
  const params = engine.fromHost(model, toHostParams(s.fx.params));
  let state = engine.startTraining(model, params, 1);
  for (let i = 0; i < 2; i++) state = engine.step(model, state, s.batch).state;
  engine.predict(model, state.params, s.batch.x, s.batch.n);
  engine.evaluate(model, state.params, s.batch);
  engine.lossGrid(
    model,
    state.params,
    s.batch,
    s.axes,
    Float32Array.of(0, 1),
    Float32Array.of(0, 1),
  );
  engine.trainMany(model, [1, 2], [s.batch, s.batch], 2);
  engine.disposeModel(model);
  engine.dispose(state);
}

describe('disposeModel', () => {
  test.each(['mlp_tanh_adam_5steps', 'dropout_p0.5'])(
    '1000 compile -> use -> dispose cycles keep liveArrays flat and memory bounded (%s)',
    (name) => {
      const s = setup(name);
      for (let i = 0; i < 20; i++) cycle(s); // warm up
      const baseline = engine.liveArrays();
      expect(baseline).toBeGreaterThanOrEqual(0);
      const rss0 = process.memoryUsage().rss;
      for (let i = 0; i < 1000; i++) {
        cycle(s);
        if (i % 250 === 0) expect(engine.liveArrays()).toBe(baseline);
      }
      expect(engine.liveArrays()).toBe(baseline);
      const grewMb = (process.memoryUsage().rss - rss0) / 1e6;
      // Disposing does not change this number: jax-js's partialEvalFlat pushes a trace onto its
      // global traceStack without ever popping it, so each valueAndGrad trace (one per compiled
      // step) keeps its trace objects (about 0.3 MB of JS heap) alive. Upstream bug; bounded here
      // only so a regression that leaks buffers (tens of MB per model) fails.
      expect(grewMb).toBeLessThan(1500);
    },
    300_000,
  );

  test('every method that takes the model throws one sentence once it is disposed; disposing twice is a no-op', () => {
    const s = setup('mlp_tanh_adam_5steps');
    const model = engine.compile(s.graph, s.data);
    const params = engine.fromHost(model, toHostParams(s.fx.params));
    const state = engine.startTraining(model, engine.clone(params), 1);
    engine.disposeModel(model);
    engine.disposeModel(model);
    const calls: Record<string, () => unknown> = {
      init: () => engine.init(model, 1),
      fromHost: () => engine.fromHost(model, toHostParams(s.fx.params)),
      startTraining: () => engine.startTraining(model, engine.clone(params), 1),
      step: () => engine.step(model, state, s.batch),
      predict: () => engine.predict(model, params, s.batch.x, s.batch.n),
      evaluate: () => engine.evaluate(model, params, s.batch),
      lossGrid: () =>
        engine.lossGrid(model, params, s.batch, s.axes, Float32Array.of(0), Float32Array.of(1)),
      trainMany: () => engine.trainMany(model, [1], [s.batch], 1),
    };
    for (const [name, call] of Object.entries(calls)) {
      expect(call, name).toThrow(SENTENCE);
    }
    // Params and TrainStates made for it remain valid arrays
    expect(Object.keys(engine.toHost(params))).toEqual(Object.keys(engine.toHost(state.params)));
    engine.dispose(engine.clone(params));
    engine.dispose(params);
    engine.dispose(state);
  });

  test('a model disposed before any use is fine, and another model is unaffected', () => {
    const s = setup('mlp_tanh_adam_5steps');
    const a = engine.compile(s.graph, s.data);
    const b = engine.compile(s.graph, s.data);
    engine.disposeModel(a);
    const p = engine.fromHost(b, toHostParams(s.fx.params));
    expect(engine.predict(b, p, s.batch.x, s.batch.n)).toHaveLength(s.batch.n);
    engine.dispose(p);
    engine.disposeModel(b);
  });
});
