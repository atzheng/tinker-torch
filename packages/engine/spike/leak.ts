// Leak test (spike item 4). Method: liveArrays() from probe.ts reads the size of the Wasm
// backend's private slot table (one entry per live buffer). wasmBytes() reads the linear memory
// size as a second signal. A step loop is leak-free when both are flat after warm-up.
import './probe.ts'; // must be the first import (see probe.ts)
import { jit, numpy as np, random, tree, valueAndGrad } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import * as optax from '@jax-js/optax';
import {
  initJax,
  initParams,
  liveArrays,
  makeManyTrainer,
  makeOptimizer,
  makeTrainer,
  sineData,
  wasmBytes,
} from './lib.ts';
import type { OptName, Params } from './lib.ts';

export type LeakResult = {
  name: string;
  steps: number;
  liveAtWarmup: number;
  liveAtEnd: number;
  liveMax: number;
  wasmBytesAtWarmup: number;
  wasmBytesAtEnd: number;
  flat: boolean;
  /** Negative control: it leaks on purpose, so `flat === false` is the passing outcome. */
  control: boolean;
};

type Loop = { step(): void; dispose(): void };

function measure(name: string, steps: number, make: () => Loop, control = false): LeakResult {
  // jax-js keeps a few constant buffers in library-global caches (e.g. the jitted random.uniform);
  // build and drop the loop once so they are not counted as ours.
  const pre = make();
  pre.step();
  pre.step();
  pre.dispose();
  const base = liveArrays();
  const loop = make();
  const warm = 50;
  for (let i = 0; i < warm; i++) loop.step();
  const liveWarm = liveArrays() - base;
  const bytesWarm = wasmBytes();
  let liveMax = liveWarm;
  for (let i = warm; i < steps; i++) {
    loop.step();
    if (i % 100 === 0) liveMax = Math.max(liveMax, liveArrays() - base);
  }
  const liveEnd = liveArrays() - base;
  const bytesEnd = wasmBytes();
  loop.dispose();
  const afterDispose = liveArrays() - base;
  return {
    name: name + (afterDispose === 0 ? '' : ` (${afterDispose} buffers left after dispose!)`),
    steps,
    liveAtWarmup: liveWarm,
    liveAtEnd: liveEnd,
    liveMax,
    wasmBytesAtWarmup: bytesWarm,
    wasmBytesAtEnd: bytesEnd,
    flat:
      liveEnd === liveWarm && liveMax === liveWarm && bytesEnd === bytesWarm && afterDispose === 0,
    control,
  };
}

/** MLP with dropout on the hidden layer; the PRNG key is part of the jitted state. */
function dropoutLoop(optName: OptName, p: number): Loop {
  const opt = makeOptimizer(optName, 0.02);
  const data = sineData(40, 1);
  const lossFn = (params: Params, key: JArray, x: JArray, y: JArray) => {
    const ids = Object.keys(params);
    let h = x;
    ids.forEach((id, i) => {
      const l = params[id]!;
      h = np.matmul(h, l.W).add(l.b);
      if (i < ids.length - 1) {
        h = np.tanh(h);
        const mask = random.bernoulli(key.ref, 1 - p, h.shape);
        h = np.where(mask, h.div(1 - p), 0);
      }
    });
    key.dispose();
    return np.square(h.sub(y)).mean();
  };
  const step = jit(
    (params: Params, optState: optax.OptState, key: JArray, x: JArray, y: JArray) => {
      const [k1, k2] = np.unstack(random.split(key, 2));
      const [loss, grads] = valueAndGrad(lossFn)(tree.ref(params), k1!, x, y);
      const [updates, newOpt] = opt.update(grads, optState);
      return [optax.applyUpdates(params, updates), newOpt, k2!, loss] as [
        Params,
        optax.OptState,
        JArray,
        JArray,
      ];
    },
  );
  let params = initParams([1, 16, 1], 1);
  let optState = opt.init(tree.ref(params));
  let key = random.key(42);
  const x = np.array(data.x, { shape: [data.n, 1] });
  const y = np.array(data.y, { shape: [data.n, 1] });
  return {
    step() {
      const [p2, o2, k2, loss] = step(params, optState, key, x.ref, y.ref);
      params = p2;
      optState = o2;
      key = k2;
      loss.item();
    },
    dispose() {
      tree.dispose(params);
      tree.dispose(optState);
      key.dispose();
      x.dispose();
      y.dispose();
      step.dispose();
    },
  };
}

/** Training steps plus a predict call every step (what the page does between frames). */
function trainerLoop(optName: OptName, tanh: boolean): Loop {
  const sizes = tanh ? [1, 16, 1] : [1, 1];
  const t = makeTrainer(sizes, tanh, optName, 0.02, sineData(40, 1), 1);
  const grid = new Float32Array(256).map((_, i) => -3 + (6 * i) / 255);
  return {
    step() {
      t.run(1);
      t.predict(grid);
    },
    dispose: () => t.dispose(),
  };
}

function manyLoop(): Loop {
  const k = 20;
  const m = makeManyTrainer(
    [1, 16, 1],
    true,
    'adam',
    0.02,
    Array.from({ length: k }, (_, i) => sineData(40, 10 + i)),
    Array.from({ length: k }, (_, i) => i),
  );
  return {
    step: () => void m.run(1),
    dispose: () => m.dispose(),
  };
}

/** Negative control: a loop that forgets to dispose one array per step. Must be reported as a leak. */
function leakyLoop(): Loop {
  const kept: JArray[] = [];
  return {
    step() {
      kept.push(np.array([1, 2, 3]).mul(2).add(1)); // realised below, never disposed
      kept[kept.length - 1]!.ref.dataSync();
    },
    dispose() {
      kept.forEach((a) => a.dispose());
    },
  };
}

export async function runLeakTests(
  log: (s: string) => void,
  steps = 10_000,
): Promise<LeakResult[]> {
  await initJax();
  const cases: [string, () => Loop, number][] = [
    ['linear sgd, train + predict each step', () => trainerLoop('sgd', false), steps],
    ['mlp sgd, train + predict each step', () => trainerLoop('sgd', true), steps],
    ['mlp momentum, train + predict each step', () => trainerLoop('momentum', true), steps],
    ['mlp adam, train + predict each step', () => trainerLoop('adam', true), steps],
    ['mlp adam-optax, train + predict each step', () => trainerLoop('adam-optax', true), steps],
    ['mlp adam + dropout (key in state)', () => dropoutLoop('adam', 0.2), steps],
    ['trainMany k=20, vmap(jit step)', manyLoop, Math.min(steps, 2_000)],
    ['CONTROL: forgets to dispose one array per step', leakyLoop, 1_000],
  ];
  const out: LeakResult[] = [];
  for (const [name, make, n] of cases) {
    const r = measure(name, n, make, name.startsWith('CONTROL'));
    out.push(r);
    log(
      `${r.flat ? 'flat ' : 'LEAK '}${r.control ? '(expected) ' : ''}${r.name}: ${r.steps} steps, live ${r.liveAtWarmup} -> ${r.liveAtEnd} (max ${r.liveMax}), wasm ${(r.wasmBytesAtWarmup / 1e6).toFixed(2)} -> ${(r.wasmBytesAtEnd / 1e6).toFixed(2)} MB`,
    );
  }
  return out;
}
