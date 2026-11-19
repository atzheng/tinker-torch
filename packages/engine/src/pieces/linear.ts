import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** x @ W (+ b), x [n, in], W [in, out]. Consumes `inputs.in` and the layer. */
export const linear: EnginePiece<NodeOf<'linear'>> = {
  forward(_node, inputs, layer) {
    const x = inputs['in'];
    if (!x || !layer) throw new Error('A Linear piece is missing its input or parameters.');
    const h = np.matmul(x, layer.W);
    return layer.b ? h.add(layer.b) : h;
  },
};
