// Shared pieces of the jax-js spike: backend init with a live-buffer probe, tiny datasets, an MLP
// over a params object keyed by node id, and jitted training steps.
//
// Ownership conventions used here (jax-js: every function consumes its Array arguments, `.ref`
// adds a reference). Every function below says what it consumes.
import { liveArrays, wasmBytes, wasmShared } from './probe.ts';
import { init, jit, numpy as np, random, tree, valueAndGrad, vmap } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import * as optax from '@jax-js/optax';

export type Layer = { W: JArray; b: JArray };
/** Parameters keyed by node id, the shape the real engine will use. */
export type Params = Record<string, Layer>;
/** 'adam' is the exact-bias-correction version below; 'adam-optax' is optax.adam as shipped. */
export type OptName = 'sgd' | 'momentum' | 'adam' | 'adam-optax';

let ready: Promise<void> | null = null;
/** Initialise the Wasm backend. Call once before any jax-js work. */
export function initJax(): Promise<void> {
  ready ??= (async () => {
    await init('wasm');
  })();
  return ready;
}

// --- deterministic data -------------------------------------------------------------------------
/** Small seeded generator (spike only; the app uses core/rng.ts). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Data = { x: Float32Array; y: Float32Array; n: number };

/** y = 2x + 1 + noise, x in [-1, 1]. */
export function linearData(n: number, seed: number): Data {
  const r = mulberry32(seed);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = r() * 2 - 1;
    y[i] = 2 * x[i]! + 1 + (r() - 0.5) * 0.2;
  }
  return { x, y, n };
}

/** y = sin(x) + noise, x in [-3, 3]. */
export function sineData(n: number, seed: number): Data {
  const r = mulberry32(seed);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = r() * 6 - 3;
    y[i] = Math.sin(x[i]!) + (r() - 0.5) * 0.1;
  }
  return { x, y, n };
}

// --- model ---------------------------------------------------------------------------------------
/** Consumes nothing (builds fresh arrays). Layer ids are keys of the returned object. */
export function initParams(sizes: number[], seed: number): Params {
  let key = random.key(seed);
  const params: Params = {};
  for (let i = 0; i < sizes.length - 1; i++) {
    const nin = sizes[i]!;
    const nout = sizes[i + 1]!;
    const [k1, k2, next] = np.unstack(random.split(key, 3)); // unstack consumes the [3, 2] array
    const bound = 1 / Math.sqrt(nin);
    params[`lin${i + 1}`] = {
      W: random.uniform(k1!, [nin, nout], { minval: -bound, maxval: bound }),
      b: random.uniform(k2!, [nout], { minval: -bound, maxval: bound }),
    };
    key = next!;
  }
  key.dispose();
  return params;
}

/**
 * Forward pass. Consumes `params` and `x`. tanh between layers when `tanh` is true.
 * Linear layers are x @ W + b, x of shape [n, d].
 */
export function forward(params: Params, x: JArray, tanh: boolean): JArray {
  const ids = Object.keys(params);
  let h = x;
  ids.forEach((id, i) => {
    const l = params[id]!;
    h = np.matmul(h, l.W).add(l.b);
    if (tanh && i < ids.length - 1) h = np.tanh(h);
  });
  return h;
}

/** Mean squared error of the model. Consumes params, x, y. */
export function mseLoss(params: Params, x: JArray, y: JArray, tanh: boolean): JArray {
  const pred = forward(params, x, tanh);
  return np.square(pred.sub(y)).mean();
}

/**
 * Adam with exact bias correction. optax.adam 0.1.3 computes 1 - b^t as `1 - pow(b, t)` in float32,
 * which cancels badly for b2 = 0.999 (the first step is 0.6% too large). Here the state carries
 * bc1 = 1 - b1^t and bc2 = 1 - b2^t directly through bc' = b*bc + (1-b), which has no
 * cancellation. update() consumes `updates`, `state` and `params` (the unused params argument is
 * disposed, as optax does). Same GradientTransformation interface, so it drops into makeStep.
 */
export function adamExact(
  lr: number,
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
): optax.GradientTransformation {
  type Tree = Record<string, Layer>;
  type S = { bc1: JArray; bc2: JArray; mu: Tree; nu: Tree };
  return {
    init(params) {
      const mu = optax.treeZerosLike(tree.ref(params));
      const nu = optax.treeZerosLike(params);
      return { bc1: np.array(0), bc2: np.array(0), mu, nu } as unknown as optax.OptState;
    },
    update(updates, state, params) {
      tree.dispose(params);
      const s = state as unknown as S;
      const bc1 = s.bc1.mul(b1).add(1 - b1);
      const bc2 = s.bc2.mul(b2).add(1 - b2);
      const g = updates as unknown as Tree;
      const mu = tree.map(
        (g_: JArray, m: JArray) => g_.mul(1 - b1).add(m.mul(b1)),
        tree.ref(g),
        s.mu,
      );
      const nu = tree.map(
        (g_: JArray, v: JArray) =>
          np
            .square(g_)
            .mul(1 - b2)
            .add(v.mul(b2)),
        g,
        s.nu,
      );
      const out = tree.map(
        (m: JArray, v: JArray) =>
          m
            .div(bc1.ref)
            .div(np.sqrt(v.div(bc2.ref)).add(eps))
            .mul(-lr),
        tree.ref(mu),
        tree.ref(nu),
      );
      return [out, { bc1, bc2, mu, nu }] as unknown as [typeof updates, optax.OptState];
    },
  };
}

export function makeOptimizer(name: OptName, lr: number): optax.GradientTransformation {
  if (name === 'sgd') return optax.sgd(lr);
  if (name === 'momentum') return optax.sgd(lr, { momentum: 0.9 });
  if (name === 'adam-optax') return optax.adam(lr, { b1: 0.9, b2: 0.999, eps: 1e-8 });
  return adamExact(lr);
}

/**
 * One training step, jitted. Signature of the returned function:
 *   (params, optState, x, y) -> [newParams, newOptState, loss]
 * It CONSUMES all four arguments (jit follows the jax-js rule), so callers pass `.ref` for x and y
 * when they reuse a batch, and tree.ref(...) when they keep the old params.
 * The loss returned is the loss before the update.
 */
export function makeStep(opt: optax.GradientTransformation, tanh: boolean) {
  const lossFn = (p: Params, x: JArray, y: JArray) => mseLoss(p, x, y, tanh);
  return jit((params: Params, optState: optax.OptState, x: JArray, y: JArray) => {
    const [loss, grads] = valueAndGrad(lossFn)(tree.ref(params), x, y);
    const [updates, newOpt] = opt.update(grads, optState);
    const newParams = optax.applyUpdates(params, updates);
    return [newParams, newOpt, loss] as [Params, optax.OptState, JArray];
  });
}

export type Trainer = {
  /** Run `steps` full-batch steps; returns the loss before each update. */
  run(steps: number): Float32Array;
  /** Current params as plain arrays (copies). */
  read(): Record<string, { W: Float32Array; b: Float32Array }>;
  /** Predict on x. Does not consume anything the caller owns. */
  predict(x: Float32Array): Float32Array;
  /** Dispose every array the trainer owns. */
  dispose(): void;
};

/** Full-batch trainer on Data. The trainer owns params, optimizer state and the batch arrays. */
export function makeTrainer(
  sizes: number[],
  tanh: boolean,
  optName: OptName,
  lr: number,
  data: Data,
  seed: number,
): Trainer {
  const opt = makeOptimizer(optName, lr);
  const step = makeStep(opt, tanh);
  let params = initParams(sizes, seed);
  let optState = opt.init(tree.ref(params));
  const x = np.array(data.x, { shape: [data.n, 1] });
  const y = np.array(data.y, { shape: [data.n, 1] });
  return {
    run(steps) {
      const losses = new Float32Array(steps);
      for (let i = 0; i < steps; i++) {
        const [p, o, loss] = step(params, optState, x.ref, y.ref);
        params = p;
        optState = o;
        losses[i] = loss.item(); // item() consumes loss
      }
      return losses;
    },
    read() {
      const out: Record<string, { W: Float32Array; b: Float32Array }> = {};
      for (const [id, l] of Object.entries(params)) {
        out[id] = {
          W: Float32Array.from(l.W.ref.dataSync() as Float32Array),
          b: Float32Array.from(l.b.ref.dataSync() as Float32Array),
        };
      }
      return out;
    },
    predict(xs) {
      const out = forward(tree.ref(params), np.array(xs, { shape: [xs.length, 1] }), tanh);
      return Float32Array.from(out.dataSync() as Float32Array);
    },
    dispose() {
      tree.dispose(params);
      tree.dispose(optState);
      x.dispose();
      y.dispose();
      step.dispose();
    },
  };
}

// --- many copies at once (the resampling overlay) -------------------------------------------------
/**
 * k independent trainings, one vmap over the whole step, jitted. Each copy has its own data and
 * params (stacked on a leading axis of size k). Returns the stacked final params as host arrays.
 */
export type ManyTrainer = {
  run(steps: number): Float32Array; // returns final loss per copy [k]
  read(): Record<string, { W: Float32Array; b: Float32Array }>;
  dispose(): void;
};

/** Stack k trees on a new leading axis. Consumes nothing; the inputs' arrays are disposed. */
function stackParams(list: Params[]): Params {
  const out: Params = {};
  for (const id of Object.keys(list[0]!)) {
    out[id] = {
      W: np.stack(list.map((p) => p[id]!.W)),
      b: np.stack(list.map((p) => p[id]!.b)),
    };
  }
  return out;
}

export function makeManyTrainer(
  sizes: number[],
  tanh: boolean,
  optName: OptName,
  lr: number,
  datasets: Data[],
  seeds: number[],
  syncEvery = 1,
): ManyTrainer {
  const k = datasets.length;
  const opt = makeOptimizer(optName, lr);
  const step = makeStep(opt, tanh);
  const vstep = jit(vmap(step, 0));
  let params = stackParams(seeds.map((s) => initParams(sizes, s)));
  // optimizer state for the stacked tree, with the scalar step count broadcast to [k]
  let optState = tree.map(
    (a: JArray) => (a.ndim === 0 ? np.broadcastTo(a, [k]) : a),
    opt.init(tree.ref(params)),
  ) as optax.OptState;
  const n = datasets[0]!.n;
  const xs = new Float32Array(k * n);
  const ys = new Float32Array(k * n);
  datasets.forEach((d, i) => {
    xs.set(d.x, i * n);
    ys.set(d.y, i * n);
  });
  const x = np.array(xs, { shape: [k, n, 1] });
  const y = np.array(ys, { shape: [k, n, 1] });
  let lastLoss: JArray | null = null;
  return {
    run(steps) {
      for (let i = 0; i < steps; i++) {
        const [p, o, loss] = vstep(params, optState, x.ref, y.ref);
        params = p;
        optState = o;
        lastLoss?.dispose();
        lastLoss = loss;
        // jax-js is lazy: unless something is read, step i+1 stays an unevaluated graph on top of
        // step i. Reading the loss every `syncEvery` steps forces evaluation (see docs/spike.md).
        if ((i + 1) % syncEvery === 0) lastLoss.ref.dataSync();
      }
      return Float32Array.from(lastLoss!.ref.dataSync() as Float32Array);
    },
    read() {
      const out: Record<string, { W: Float32Array; b: Float32Array }> = {};
      for (const [id, l] of Object.entries(params)) {
        out[id] = {
          W: Float32Array.from(l.W.ref.dataSync() as Float32Array),
          b: Float32Array.from(l.b.ref.dataSync() as Float32Array),
        };
      }
      return out;
    },
    dispose() {
      tree.dispose(params);
      tree.dispose(optState);
      lastLoss?.dispose();
      x.dispose();
      y.dispose();
      vstep.dispose();
      step.dispose();
    },
  };
}

export { liveArrays, wasmBytes, wasmShared };
