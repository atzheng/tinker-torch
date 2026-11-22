import type { Array as JArray } from '@jax-js/jax';
import { nn, numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * One-hot of a category column. `x` is `[n, 1]` holding category indices as floats (0..K-1); the
 * result is `[n, K]` with a 1 in the column of each row's category. Consumes `x`.
 */
function oneHotOf(x: JArray, k: number): JArray {
  const n = x.shape[0] ?? 0;
  // The indices are exact integers stored as floats; round guards against a stray 2.9999.
  return nn.oneHot(np.round(x).reshape([n]).astype(np.int32), k).astype(np.float32);
}

/** `[n, 1]` category indices -> `[n, K]` one-hot columns, K = the dataset's category count. Consumes `inputs.in`. */
export const onehot: EnginePiece<NodeOf<'onehot'>> = {
  forward(node, inputs, _layer, ctx) {
    const x = inputs['in'];
    if (!x) throw new Error('A One-hot piece is missing its input.');
    return oneHotOf(x, ctx.categoryCount(node.id));
  },
};

/**
 * Lookup: row `index` of the table W `[K, width]` per example, computed as one-hot times W (a matmul,
 * so the gradient is the plain one: each category's row sees only its own examples). With width 1 it
 * is one learned number per category; a larger width is an embedding. Consumes `inputs.in` and the layer.
 */
export const lookup: EnginePiece<NodeOf<'lookup'>> = {
  forward(node, inputs, layer, ctx) {
    const x = inputs['in'];
    const W = layer?.['W'];
    if (!x || !W) throw new Error('A Lookup piece is missing its input or table.');
    return np.matmul(oneHotOf(x, ctx.categoryCount(node.id)), W);
  },
};
