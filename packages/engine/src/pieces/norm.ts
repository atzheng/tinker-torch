import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * sum w^2 (l2) or sum |w| (l1) of a weight matrix -> a scalar (np.abs: |w|'(0) = 1, owner decision).
 * Consumes `inputs.in`.
 */
export const norm: EnginePiece<NodeOf<'norm'>> = {
  forward(node, inputs) {
    const w = inputs['in'];
    if (!w) throw new Error('A Norm piece is missing its input.');
    return (node.kind === 'l2' ? np.square(w) : np.abs(w)).sum();
  },
};
