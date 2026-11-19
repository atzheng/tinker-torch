import { numpy as np } from '@jax-js/jax';
import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Elementwise x^2. Consumes `inputs.in`. */
export const square: EnginePiece<v2.NodeOf<'square'>> = {
  forward(_node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('A Square piece is missing its input.');
    return np.square(x);
  },
};
