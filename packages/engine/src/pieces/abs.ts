import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Elementwise |x| (np.abs: |x|'(0) = 1, owner decision, unlike torch's 0). Consumes `inputs.in`. */
export const abs: EnginePiece<NodeOf<'abs'>> = {
  forward(_node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('An Abs piece is missing its input.');
    return np.abs(x);
  },
};
