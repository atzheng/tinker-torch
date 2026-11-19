// trainMany: k copies of one compiled model trained in one jitted, vmapped step (CONTRACTS.md
// section 7). Copy i is exactly `init(seeds[i])` + `startTraining(.., seeds[i])` + `steps` calls of
// `engine.step` on data[i], the whole thing vectorised.
//
// Ownership: everything allocated here is disposed before returning, also when a step throws. The
// data and the stacked state are passed to the jitted function as arguments (never captured: an
// array constant captured in a trace is never freed). The jitted function consumes its arguments.
import { jit, numpy as np, random, tree, vmap } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type * as optax from '@jax-js/optax';
import { v2 } from '@tinker/core';
import type { Batch, HostParams } from '@tinker/core';
import { internals } from './compile.js';
import type { Internals } from './compile.js';
import type { ParamTree } from './convert.js';
import type { ManyResult, Model } from './types.js';

type ManyStepFn = (
  params: ParamTree,
  opt: optax.OptState,
  x: JArray,
  y: JArray,
  key: JArray,
) => [ParamTree, optax.OptState, JArray, JArray];

/** The state a copy carries between steps: stacked params, optimizer state and dropout key. */
type Carry = [ParamTree, optax.OptState, JArray];

/** One jitted vmapped step per compiled model (it holds no array constants, only code). */
const manySteps = new WeakMap<Internals, ManyStepFn>();

/**
 * The vmapped step. Inputs are stacked on a leading axis k. Returns [params, opt, ok, key] with
 * ok [k] = 1 for a copy whose loss and updated parameters are all finite this step.
 */
function manyStep(m: Internals): ManyStepFn {
  const cached = manySteps.get(m);
  if (cached) return cached;
  const vstep = vmap(m.stepFn());
  const fn = jit((params: ParamTree, opt: optax.OptState, x: JArray, y: JArray, key: JArray) => {
    const [newP, newO, loss, check, newK] = vstep(params, opt, x, y, key) as [
      ParamTree,
      optax.OptState,
      JArray,
      JArray,
      JArray,
    ];
    // jax-js folds x * 0 to 0 (NaN included), so finiteness is tested with isfinite: the loss and
    // every updated parameter of the copy
    check.dispose();
    const k = loss.shape[0] ?? 1;
    let finite = np.isfinite(loss).astype(np.float32);
    let count = 0;
    for (const leaf of tree.leaves(tree.ref(newP)) as JArray[]) {
      const per = leaf.shape.slice(1).reduce((p, q) => p * q, 1);
      count += per;
      finite = finite.add(np.isfinite(leaf).astype(np.float32).reshape([k, per]).sum(1));
    }
    return [newP, newO, finite.equal(count + 1).astype(np.float32), newK];
  }) as unknown as ManyStepFn;
  manySteps.set(m, fn);
  m.disposables.add(fn as unknown as { dispose(): void });
  return fn;
}

/**
 * Per copy, keeps `fresh` where `alive` is 1 and `old` where it is 0 (a diverged copy stays at its
 * last finite state). Only used once some copy has diverged. Consumes all three arguments.
 */
let freezeFn: ((alive: JArray, fresh: Carry, old: Carry) => Carry) | null = null;
function freeze(alive: JArray, fresh: Carry, old: Carry): Carry {
  freezeFn ??= jit((a: JArray, f: Carry, o: Carry) => {
    const sel = (x: JArray, y: JArray): JArray => {
      const cond = a.ref.reshape([a.shape[0] ?? 1, ...x.shape.slice(1).map(() => 1)]);
      return np.where(cond.greater(0), x, y);
    };
    const out = tree.map(sel, f, o) as Carry;
    a.dispose();
    return out;
  }) as unknown as (alive: JArray, fresh: Carry, old: Carry) => Carry;
  return freezeFn(alive, fresh, old);
}

const size = (a: JArray): number => a.shape.reduce((p, q) => p * q, 1);

/** Stacks one host array per copy into a [k, ...shape] jax array. */
function stacked(parts: Float32Array[], shape: number[]): JArray {
  const each = parts[0]?.length ?? 0;
  const flat = new Float32Array(parts.length * each);
  parts.forEach((p, i) => flat.set(p, i * each));
  return np.array(flat, { shape: [parts.length, ...shape], dtype: np.float32 });
}

/** Throws a plain sentence when the arguments do not fit (before anything is allocated). */
function checkMany(m: Internals, seeds: number[], data: Batch[], steps: number): void {
  const k = seeds.length;
  if (k < 1) throw new Error('trainMany needs at least one copy.');
  if (data.length !== k) throw new Error('Each copy needs its own training set.');
  if (!Number.isInteger(steps) || steps < 0)
    throw new Error('The number of steps must be 0 or more.');
  const d = m.dataColumns.length;
  const n = data[0]?.n ?? 0;
  if (!Number.isInteger(n) || n < 1) throw new Error('Each training set needs at least one row.');
  for (const b of data) {
    if (b.n !== n) throw new Error('All copies need training sets of the same size.');
    if (b.x.length !== n * d)
      throw new Error(`A training set x has ${b.x.length} values, expected ${n} x ${d}.`);
    if (b.y.length !== n)
      throw new Error(`A training set y has ${b.y.length} values, expected ${n}.`);
  }
}

/**
 * Borrows everything. Trains k = `seeds.length` copies for `steps` full-batch steps and returns the
 * trained parameters per copy (index i = seeds[i], data[i]). Every `data[i]` must have the same `n`.
 * The loss is read back every step (jax-js is lazy). A copy that diverges stops at its last finite
 * parameters and is flagged `diverged: true`; the others are unaffected. Stops early when every
 * copy has diverged. Throws a plain sentence on a bad argument.
 */
export function trainMany(
  model: Model,
  seeds: number[],
  data: Batch[],
  steps: number,
): ManyResult[] {
  const m = internals(model);
  checkMany(m, seeds, data, steps);
  const k = seeds.length;
  const n = data[0]?.n ?? 0;
  const d = m.dataColumns.length;

  const inits = seeds.map((s) => v2.initParams(m.analysis, s));
  let params: ParamTree = {};
  for (const [id, s] of m.shapes) {
    const layer: ParamTree[string] = {
      W: stacked(
        inits.map((h) => h[id]?.W ?? new Float32Array(0)),
        s.W,
      ),
    };
    if (s.b)
      layer.b = stacked(
        inits.map((h) => h[id]?.b ?? new Float32Array(0)),
        s.b,
      );
    params[id] = layer;
  }
  let opt = tree.map(
    // the step counter is a scalar: give every copy its own
    (a: JArray) => (a.ndim === 0 ? np.broadcastTo(a, [k]) : a),
    m.optimizer().init(tree.ref(params)),
  ) as optax.OptState;
  let key = np.stack(seeds.map((s) => random.key(s)));
  const x = stacked(
    data.map((b) => b.x),
    [n, d],
  );
  const y = stacked(
    data.map((b) => b.y),
    [n],
  );
  const diverged = new Array<boolean>(k).fill(false);

  const step = manyStep(m);
  let prev: Carry | null = null;
  try {
    for (let i = 0; i < steps; i++) {
      // the call consumes params, opt and key: keep references to the state before the step, which
      // a copy that diverges in this step returns to
      prev = [tree.ref(params), tree.ref(opt), key.ref];
      const out = step(params, opt, x.ref, y.ref, key);
      const ok = out[2].dataSync() as Float32Array; // reading every step forces evaluation
      const wasClean = !diverged.includes(true);
      for (let c = 0; c < k; c++) if (ok[c] === 0) diverged[c] = true;
      if (wasClean && !diverged.includes(true)) {
        tree.dispose(prev[0]);
        tree.dispose(prev[1]);
        prev[2].dispose();
        [params, opt, , key] = out;
      } else {
        const alive = np.array(Float32Array.from(diverged, (d) => (d ? 0 : 1)));
        [params, opt, key] = freeze(alive, [out[0], out[1], out[3]], prev);
      }
      prev = null;
      if (!diverged.includes(false)) break;
    }
    const result: ManyResult[] = [];
    const host = (a: JArray, c: number, size: number): Float32Array =>
      Float32Array.from((a.ref.dataSync() as Float32Array).subarray(c * size, (c + 1) * size));
    for (let c = 0; c < k; c++) {
      const hp = Object.create(null) as HostParams;
      for (const [id, l] of Object.entries(params)) {
        const W = host(l.W, c, size(l.W) / k);
        hp[id] = l.b ? { W, b: host(l.b, c, size(l.b) / k) } : { W };
      }
      result.push({ params: hp, diverged: diverged[c] ?? false });
    }
    return result;
  } finally {
    // after a throwing step some of these were already consumed by it
    for (const h of [params, opt, key, x, y, prev]) {
      try {
        if (h) tree.dispose(h as ParamTree);
      } catch {
        // already consumed
      }
    }
  }
}
