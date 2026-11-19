// tinker-xqp: jax-js 0.1.27 built a new WebAssembly.Instance per kernel dispatch (about 20 per
// training step). Hundreds of thousands of dead instances made Chrome stall for ~550 ms every
// ~3000 steps. patches/@jax-js__jax@0.1.27.patch caches the instance per compiled kernel.
import { expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import { loadFixture, toBatch, toHostParams } from './util.js';

test('training steps do not create WebAssembly instances once kernels are warm', async () => {
  const engine = await createEngine();
  const { fx, graph, data } = loadFixture('mlp_tanh_sgd_5steps');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 1);
  const once = () => {
    state = engine.step(model, state, batch).state;
  };
  for (let i = 0; i < 20; i++) once();

  // The engine tsconfig has no DOM or Wasm lib types, hence the structural cast.
  const wa = (
    globalThis as unknown as { WebAssembly: { Instance: new (...a: unknown[]) => object } }
  ).WebAssembly;
  const Real = wa.Instance;
  let created = 0;
  class Counting extends Real {
    constructor(...args: unknown[]) {
      super(...args);
      created++;
    }
  }
  wa.Instance = Counting;
  try {
    for (let i = 0; i < 200; i++) once();
  } finally {
    wa.Instance = Real;
  }
  expect(created).toBe(0);
  engine.dispose(state);
});
