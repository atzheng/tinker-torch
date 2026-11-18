// HostParams <-> jax arrays. The only place arrays cross the package boundary (with predict's readback).
import { numpy as np, tree } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { HostParams, NodeId } from '@tinker/core';
import type { Params } from './types.js';

export type Layer = { W: JArray; b?: JArray };
/** Parameters keyed by node id (linear nodes only). */
export type ParamTree = Record<NodeId, Layer>;
export type LayerShape = { W: [number, number]; b?: [number] };

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
  for (const [id, s] of shapes) {
    const h = Object.hasOwn(host, id) ? host[id] : undefined;
    if (!h) throw new Error(`The parameters have no entry for "${id}".`);
    if (h.W.length !== s.W[0] * s.W[1]) {
      throw new Error(
        `The weights of "${id}" have ${h.W.length} values, expected ${s.W[0] * s.W[1]}.`,
      );
    }
    if (s.b && h.b?.length !== s.b[0]) {
      throw new Error(`The bias of "${id}" must have ${s.b[0]} values.`);
    }
    if (!s.b && h.b !== undefined)
      throw new Error(`"${id}" has no bias but the parameters give one.`);
  }
  for (const [id, s] of shapes) {
    const h = host[id];
    if (!h) continue;
    const layer: Layer = {
      W: np.array(Float32Array.from(h.W), { shape: [s.W[0], s.W[1]], dtype: np.float32 }),
    };
    if (s.b && h.b)
      layer.b = np.array(Float32Array.from(h.b), { shape: [s.b[0]], dtype: np.float32 });
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
    host[id] = l.b ? { W: read(l.W), b: read(l.b) } : { W: read(l.W) };
  }
  return host;
}
