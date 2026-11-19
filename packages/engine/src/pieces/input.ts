import { numpy as np } from '@jax-js/jax';
import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Takes x and keeps the node's columns, in order. `inputs` is empty (an Input has no input port). */
export const input: EnginePiece<v2.NodeOf<'input'>> = {
  forward(node, _inputs, _layer, ctx) {
    const x = ctx.takeX();
    const idx = node.columns.map((c) => ctx.dataColumns.indexOf(c));
    const identity = idx.length === ctx.dataColumns.length && idx.every((v, i) => v === i);
    if (identity) return x;
    if (idx.some((v) => v < 0)) throw new Error('An Input column is not in the dataset.');
    return np.take(x, np.array(idx, { dtype: np.int32 }), 1);
  },
};
