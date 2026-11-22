// Loss over a 1-D sweep of one parameter entry, and its slope at the current value (1-D loss curve).
//
// Same mechanism as surface.ts: the entry is overwritten inside the trace by a one-hot mask; the
// mask and the values are jit ARGUMENTS (an array constant made inside a jitted function is never
// freed). Dropout is off.
import { jit, numpy as np, tree, valueAndGrad, vmap } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { Batch } from '@tinker/core';
import { batchArrays, checkBatch, checkTree, guarded, internals } from './compile.js';
import type { Internals } from './compile.js';
import type { Layer, ParamTree } from './convert.js';
import { checkAxis, oneHot, overwrite } from './surface.js';
import type { Model, ParamAxis } from './types.js';

type CurveFn = (params: ParamTree, x: JArray, y: JArray, oh: JArray, v: JArray) => JArray;
type SlopeFn = (used: Layer, rest: ParamTree, x: JArray, y: JArray) => Layer;

const CHUNK_ELEMENTS = 1 << 20;
const curveCache = new WeakMap<Internals, Map<string, CurveFn>>();
const slopeCache = new WeakMap<Internals, Map<string, SlopeFn>>();

function cached<F>(
  cache: WeakMap<Internals, Map<string, F>>,
  m: Internals,
  key: string,
  make: () => F,
): F {
  let byKey = cache.get(m);
  if (!byKey) cache.set(m, (byKey = new Map()));
  const hit = byKey.get(key);
  if (hit) return hit;
  const fn = make();
  byKey.set(key, fn);
  m.disposables.add(fn as unknown as { dispose(): void });
  return fn;
}

/**
 * Borrows `model` and `t`. `values.length` losses (mean loss plus penalty over `batch`): entry i is
 * the loss with parameter `axis` = `values[i]`. Non-finite values come back as they are.
 */
export function lossCurve(
  model: Model,
  t: ParamTree,
  batch: Batch,
  axis: ParamAxis,
  values: Float32Array,
): Float32Array {
  const m = internals(model);
  checkBatch(m, batch.x, batch.y, batch.n);
  checkTree(m, t);
  const shape = checkAxis(m, axis);
  const total = values.length;
  const out = new Float32Array(total);
  if (total === 0) return out;
  const fn = cached(curveCache, m, JSON.stringify(axis), () => {
    const one = (p: ParamTree, x: JArray, y: JArray, oh: JArray, v: JArray) =>
      m.evalFn()(overwrite(p, axis, oh, v), x, y);
    return jit(vmap(one, [null, null, null, null, 0])) as unknown as CurveFn;
  });
  const chunk = Math.min(total, Math.max(1, Math.floor(CHUNK_ELEMENTS / Math.max(1, batch.n))));
  for (let start = 0; start < total; start += chunk) {
    // the last chunk is padded with repeats so every call has the same shape (one trace)
    const vv = new Float32Array(chunk);
    for (let k = 0; k < chunk; k++) vv[k] = values[Math.min(start + k, total - 1)] ?? 0;
    const { x, y } = batchArrays(m, batch);
    const pRef = tree.ref(t);
    const oh = oneHot(shape, axis.index);
    const jv = np.array(vv, { shape: [chunk], dtype: np.float32 });
    const res = guarded([x, y, pRef, oh, jv], () => fn(pRef, x, y, oh, jv));
    out.set((res.dataSync() as Float32Array).subarray(0, Math.min(chunk, total - start)), start);
  }
  return out;
}

/**
 * Borrows `model` and `t`. The derivative of the full-batch loss (mean loss plus penalty over
 * `batch`, dropout off) with respect to entry `axis` at its current value. It is the entry of the
 * gradient a training step computes, taken the same way (so conventions at kinks, e.g. |w| at 0,
 * agree), and equals that gradient only for a full batch and no dropout. It is NOT the update:
 * momentum, Adam and the learning rate change what a step does with it.
 *
 * Like `makeStep`, only the leaves the loss plan reads are differentiated (a Linear read only
 * through a Weights tap: W; a layer only the predictions read: none, slope 0): differentiating an
 * unread leaf leaves a constant buffer that jit never frees.
 */
export function lossSlope(model: Model, t: ParamTree, batch: Batch, axis: ParamAxis): number {
  const m = internals(model);
  checkBatch(m, batch.x, batch.y, batch.n);
  checkTree(m, t);
  checkAxis(m, axis);
  const key = axis.node;
  const layer = t[key];
  if (!layer) throw new Error(`"${key}" has no parameters to vary.`);
  const lossPlan = m.plans.loss;
  const read = lossPlan.layerUses.has(key);
  const wOnly = read && !lossPlan.steps.some((st) => st.node.id === key); // read through Weights taps only
  const differentiated = (k: string) => read && (!wOnly || k === 'W');
  if (!differentiated(axis.tensor)) return 0;
  const fn = cached(slopeCache, m, key, () => {
    // valueAndGrad differentiates the first argument (the leaves the loss reads) and consumes the others.
    const vg = valueAndGrad((used: Layer, rest: ParamTree, x: JArray, y: JArray) =>
      m.evalFn()({ ...rest, [key]: used }, x, y),
    );
    return jit((used: Layer, rest: ParamTree, x: JArray, y: JArray) => {
      const [loss, g] = vg(used, rest, x, y) as unknown as [JArray, Layer];
      loss.dispose();
      return g;
    }) as unknown as SlopeFn;
  });
  const used: Layer = {};
  for (const [k, a] of Object.entries(layer)) if (differentiated(k)) used[k] = a.ref;
  // Leaves the loss plan never reads are left out of the tree: a jitted function that is handed
  // an array it does not use leaves it alive (one buffer per call).
  const rest: ParamTree = {};
  for (const [id, l] of Object.entries(t)) {
    if (id === key || !lossPlan.layerUses.has(id)) continue;
    const own = lossPlan.steps.some((st) => st.node.id === id);
    rest[id] = own ? l : l.W ? { W: l.W } : {};
  }
  const { x, y } = batchArrays(m, batch);
  const rRef = tree.ref(rest);
  const g = guarded([x, y, used, rRef], () => fn(used, rRef, x, y));
  let out = 0;
  for (const [k, a] of Object.entries(g)) {
    if (k === axis.tensor)
      out = (a.dataSync() as Float32Array)[axis.index] ?? 0; // dataSync consumes
    else a.dispose();
  }
  return out;
}
