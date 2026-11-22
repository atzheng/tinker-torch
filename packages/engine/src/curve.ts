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
type SlopeFn = (layer: Layer, rest: ParamTree, x: JArray, y: JArray) => Layer;

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
 * Borrows `model` and `t`. The derivative of the loss (mean loss plus penalty over `batch`, dropout
 * off) with respect to entry `axis` at its current value: the entry of the gradient of the whole
 * tensor, taken the way a training step takes it (so convention at kinks, e.g. |w| at 0, agree).
 */
export function lossSlope(model: Model, t: ParamTree, batch: Batch, axis: ParamAxis): number {
  const m = internals(model);
  checkBatch(m, batch.x, batch.y, batch.n);
  checkTree(m, t);
  checkAxis(m, axis);
  // a layer the loss does not read (a prediction-only head) has zero gradient; differentiating it
  // would leave a constant buffer that jit never frees (see compile.ts makeStep)
  if (!m.plans.loss.layerUses.has(axis.node)) return 0;
  const key = axis.node;
  const fn = cached(slopeCache, m, key, () => {
    // valueAndGrad differentiates the first argument (the node's layer) and consumes the others.
    const vg = valueAndGrad((layer: Layer, rest: ParamTree, x: JArray, y: JArray) =>
      m.evalFn()({ ...rest, [key]: layer }, x, y),
    );
    return jit((layer: Layer, rest: ParamTree, x: JArray, y: JArray) => {
      const [loss, g] = vg(layer, rest, x, y) as unknown as [JArray, Layer];
      loss.dispose();
      return g;
    }) as unknown as SlopeFn;
  });
  const layer = t[axis.node];
  if (!layer) throw new Error(`"${axis.node}" has no parameters to vary.`);
  const rest: ParamTree = {};
  for (const [id, l] of Object.entries(t)) if (id !== key) rest[id] = l;
  const { x, y } = batchArrays(m, batch);
  const lRef = tree.ref(layer);
  const rRef = tree.ref(rest);
  const g = guarded([x, y, lRef, rRef], () => fn(lRef, rRef, x, y));
  let out = 0;
  for (const [k, a] of Object.entries(g)) {
    if (k === axis.tensor)
      out = (a.dataSync() as Float32Array)[axis.index] ?? 0; // dataSync consumes
    else a.dispose();
  }
  return out;
}
