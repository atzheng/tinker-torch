// disposeModel (CONTRACTS.md section 7, tinker-klv): frees the jitted functions, throws on later use.
import { installProbe } from '../src/probe.js';
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { describe, expect, test } from 'vitest';
import { loadFixture, toBatch, toHostParams, predict0 } from './util.js';

installProbe();
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

// Heap bound for 1000 cycles, measured with heapUsed after a forced gc. With the jax-js patch
// (patches/, tinker-d7c; see docs/jax-js-upstream-issue.md) the growth is a few MB; unpatched it
// was 320 MB+ per 1000 cycles (global jaxpr caches that were never freed). The measurement yields
// to the event loop first: V8 keeps WeakRef targets alive until the end of the current job.
const MAX_HEAP_GROWTH_MB_PER_1000_CYCLES = 100;
// Lighter always-on run; the full 1000 cycles is behind TINKER_BENCH.
const CYCLES = process.env['TINKER_BENCH'] ? 1000 : 300;

setFlagsFromString('--expose-gc');
const gc = runInNewContext('gc') as () => void;
async function heapMb(): Promise<number> {
  await new Promise((r) => setTimeout(r, 10));
  gc();
  gc();
  return process.memoryUsage().heapUsed / 1e6;
}

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
  predict0(engine, model, state.params, s.batch.x, s.batch.n);
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
    'compile -> use -> dispose cycles keep liveArrays flat and heapUsed bounded (%s)',
    async (name) => {
      const s = setup(name);
      for (let i = 0; i < 20; i++) cycle(s); // warm up
      const baseline = engine.liveArrays();
      expect(baseline).toBeGreaterThanOrEqual(0);
      const heap0 = await heapMb();
      for (let i = 0; i < CYCLES; i++) {
        cycle(s);
        if (i % 250 === 0) expect(engine.liveArrays()).toBe(baseline);
      }
      expect(engine.liveArrays()).toBe(baseline);
      const grewMb = (await heapMb()) - heap0;
      if (process.env['TINKER_BENCH'])
        console.log(`heap growth over ${CYCLES} cycles (${name}): ${grewMb.toFixed(1)} MB`);
      // Regression guard for the jax-js leaks: unpatched, each compile retained ~0.3 MB (step) to
      // ~1 MB (with trainMany), far over this bound.
      expect(grewMb).toBeLessThan((MAX_HEAP_GROWTH_MB_PER_1000_CYCLES * CYCLES) / 1000);
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
      predict: () => predict0(engine, model, params, s.batch.x, s.batch.n),
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
    expect(predict0(engine, b, p, s.batch.x, s.batch.n)).toHaveLength(s.batch.n);
    engine.dispose(p);
    engine.disposeModel(b);
  });
});
