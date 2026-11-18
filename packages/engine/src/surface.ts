// Loss over a 2-D grid of two parameter values, every other parameter fixed (loss-surface view).
//
// The grid points are a vmapped batch of scalars (a, b); the two parameter entries are overwritten
// inside the trace by one-hot masks. The masks and the grid values are jit ARGUMENTS: an array
// constant created inside a jitted function is never freed (jitted functions are never disposed).
// Dropout is off (the jitted eval function of the model is reused).
import { jit, numpy as np, tree, vmap } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { Batch } from '@tinker/core';
import { batchArrays, checkBatch, checkTree, guarded, internals } from './compile.js';
import type { Internals } from './compile.js';
import type { ParamTree } from './convert.js';
import type { Model, ParamAxis } from './types.js';

type GridFn = (
  params: ParamTree,
  x: JArray,
  y: JArray,
  ohA: JArray,
  ohB: JArray,
  a: JArray,
  b: JArray,
) => JArray;

/** Upper bound on (grid points per call) x (batch rows): keeps vmapped activations small. */
const CHUNK_ELEMENTS = 1 << 20;

const cache = new WeakMap<Internals, Map<string, GridFn>>();

function axisShape(m: Internals, ax: ParamAxis): number[] {
  const s = m.shapes.get(ax.node);
  if (!s) throw new Error(`"${ax.node}" has no parameters to vary.`);
  const shape = ax.tensor === 'W' ? s.W : s.b;
  if (!shape) throw new Error(`"${ax.node}" has no bias.`);
  return shape;
}

function checkAxis(m: Internals, ax: ParamAxis): number[] {
  const shape = axisShape(m, ax);
  const size = shape.reduce((p, v) => p * v, 1);
  if (!Number.isInteger(ax.index) || ax.index < 0 || ax.index >= size) {
    throw new Error(`Parameter ${ax.index} of "${ax.node}" does not exist.`);
  }
  return shape;
}

/** Writes `val` (a scalar tracer) into entry `ax` of the tree; consumes `oh` and `val`. Borrows nothing else. */
function overwrite(t: ParamTree, ax: ParamAxis, oh: JArray, val: JArray): ParamTree {
  const layer = t[ax.node];
  if (!layer) throw new Error(`"${ax.node}" has no parameters to vary.`);
  const old = ax.tensor === 'W' ? layer.W : layer.b;
  if (!old) throw new Error(`"${ax.node}" has no bias.`);
  // a select, not arithmetic: exact for any old value (1e8, NaN, Inf)
  const next = np.where(oh.greater(0.5), val, old);
  return { ...t, [ax.node]: ax.tensor === 'W' ? { ...layer, W: next } : { ...layer, b: next } };
}

function gridFn(m: Internals, axes: [ParamAxis, ParamAxis]): GridFn {
  const key = JSON.stringify(axes);
  let byKey = cache.get(m);
  if (!byKey) cache.set(m, (byKey = new Map()));
  const hit = byKey.get(key);
  if (hit) return hit;
  const [ax, bx] = axes;
  const one = (
    params: ParamTree,
    x: JArray,
    y: JArray,
    ohA: JArray,
    ohB: JArray,
    a: JArray,
    b: JArray,
  ) => {
    let t = overwrite(params, ax, ohA, a);
    t = overwrite(t, bx, ohB, b);
    return m.evalFn()(t, x, y);
  };
  const fn = jit(vmap(one, [null, null, null, null, null, 0, 0])) as unknown as GridFn;
  byKey.set(key, fn);
  return fn;
}

function oneHot(shape: number[], index: number): JArray {
  const size = shape.reduce((p, v) => p * v, 1);
  const v = new Float32Array(size);
  v[index] = 1;
  return np.array(v, { shape, dtype: np.float32 });
}

/**
 * Borrows `model` and `params` (the values in `params` are never changed). Returns
 * `[a.length, b.length]` row-major: entry `i * b.length + j` is the mean loss plus penalty with
 * parameter `axes[0]` = `a[i]` and `axes[1]` = `b[j]`. Non-finite values are returned as they are.
 */
export function lossGrid(
  model: Model,
  t: ParamTree,
  batch: Batch,
  axes: [ParamAxis, ParamAxis],
  a: Float32Array,
  b: Float32Array,
): Float32Array {
  const m = internals(model);
  checkBatch(m, batch.x, batch.y, batch.n);
  checkTree(m, t);
  const shapeA = checkAxis(m, axes[0]);
  const shapeB = checkAxis(m, axes[1]);
  if (
    axes[0].node === axes[1].node &&
    axes[0].tensor === axes[1].tensor &&
    axes[0].index === axes[1].index
  ) {
    throw new Error('The two axes must be different parameters.');
  }
  const na = a.length;
  const nb = b.length;
  const total = na * nb;
  const out = new Float32Array(total);
  if (total === 0) return out;

  const fn = gridFn(m, axes);
  const chunk = Math.min(total, Math.max(1, Math.floor(CHUNK_ELEMENTS / Math.max(1, batch.n))));
  for (let start = 0; start < total; start += chunk) {
    // the last chunk is padded with repeats so that every call has the same shape (one trace)
    const av = new Float32Array(chunk);
    const bv = new Float32Array(chunk);
    for (let k = 0; k < chunk; k++) {
      const p = Math.min(start + k, total - 1);
      av[k] = a[Math.floor(p / nb)] ?? 0;
      bv[k] = b[p % nb] ?? 0;
    }
    const { x, y } = batchArrays(m, batch);
    const held: unknown[] = [x, y];
    const pRef = tree.ref(t);
    held.push(pRef);
    const ohA = oneHot(shapeA, axes[0].index);
    const ohB = oneHot(shapeB, axes[1].index);
    held.push(ohA, ohB);
    const ja = np.array(av, { shape: [chunk], dtype: np.float32 });
    const jb = np.array(bv, { shape: [chunk], dtype: np.float32 });
    held.push(ja, jb);
    const res = guarded(held, () => fn(pRef, x, y, ohA, ohB, ja, jb));
    const vals = res.dataSync() as Float32Array;
    out.set(vals.subarray(0, Math.min(chunk, total - start)), start);
  }
  return out;
}
