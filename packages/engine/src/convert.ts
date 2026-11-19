// HostParams <-> jax arrays. The only place arrays cross the package boundary (with predict's readback).
import { numpy as np, tree } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { HostParams, NodeId, ParamShapes } from '@tinker/core';
import type { Params } from './types.js';

/** The named tensors of one parameter-owning node (Linear: W and b). */
export type Layer = Record<string, JArray>;
/** Parameters keyed by node id (every node that owns parameters). */
export type ParamTree = Record<NodeId, Layer>;
export type LayerShape = ParamShapes;

class ParamsImpl {
  readonly __brand = 'Params' as const;
  /** null once consumed or disposed. */
  tree: ParamTree | null;
  constructor(t: ParamTree) {
    this.tree = t;
  }
}

export function wrapParams(t: ParamTree): Params {
  return new ParamsImpl(t);
}

function impl(p: Params): ParamsImpl {
  if (!(p instanceof ParamsImpl)) throw new Error('Not a Params object made by this engine.');
  return p;
}

/** Borrow the tree. Throws when the Params was consumed or disposed. */
export function peekParams(p: Params): ParamTree {
  const t = impl(p).tree;
  if (t === null) throw new Error('These Params were already consumed or disposed.');
  return t;
}

/** Take ownership of the tree: the Params is dead afterwards. Throws when already dead. */
export function takeParams(p: Params): ParamTree {
  const t = peekParams(p);
  impl(p).tree = null;
  return t;
}

/** Dispose the arrays and mark dead. Does nothing when already dead. */
export function disposeParams(p: Params): void {
  const t = impl(p).tree;
  if (t === null) return;
  impl(p).tree = null;
  tree.dispose(t);
}

/** Borrows `src`. Result owns new references to the same arrays (jax arrays are immutable). */
export function cloneTree(src: ParamTree): ParamTree {
  return tree.ref(src);
}

/** Copies the host arrays into new jax arrays. Does not keep a reference to `host`. Throws on a mismatch. */
export function hostToTree(shapes: ReadonlyMap<NodeId, LayerShape>, host: HostParams): ParamTree {
  const out: ParamTree = {};
  // Check everything first so that a mismatch allocates nothing.
  const size = (shape: readonly number[]) => shape.reduce((n, d) => n * d, 1);
  for (const [id, s] of shapes) {
    const h = Object.hasOwn(host, id) ? host[id] : undefined;
    if (!h) throw new Error(`The parameters have no entry for "${id}".`);
    for (const [k, shape] of Object.entries(s)) {
      const v = Object.hasOwn(h, k) ? h[k] : undefined;
      if (v?.length !== size(shape)) {
        throw new Error(
          `The "${k}" parameters of "${id}" have ${v?.length ?? 0} values, expected ${size(shape)}.`,
        );
      }
    }
    for (const k of Object.keys(h)) {
      if (!Object.hasOwn(s, k))
        throw new Error(`"${id}" has no "${k}" parameters but the host gives some.`);
    }
  }
  for (const [id, s] of shapes) {
    const h = host[id];
    if (!h) continue;
    const layer: Layer = {};
    for (const [k, shape] of Object.entries(s)) {
      const v = h[k];
      if (v) layer[k] = np.array(Float32Array.from(v), { shape, dtype: np.float32 });
    }
    out[id] = layer;
  }
  return out;
}

function read(a: JArray): Float32Array {
  // dataSync consumes its array, so read through a reference; copy so the result is independent.
  return Float32Array.from(a.ref.dataSync() as Float32Array);
}

/** Borrows `t`. Always allocates new arrays; the result is prototype-free like core's HostParams. */
export function treeToHost(t: ParamTree): HostParams {
  const host = Object.create(null) as HostParams;
  for (const [id, l] of Object.entries(t)) {
    const h: Record<string, Float32Array> = Object.create(null) as Record<string, Float32Array>;
    for (const [k, a] of Object.entries(l)) h[k] = read(a);
    host[id] = h;
  }
  return host;
}
