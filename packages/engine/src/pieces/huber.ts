import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Elementwise Huber of a difference d, as nn.HuberLoss(delta): 0.5 d^2 where |d| < delta, else
 * delta * (|d| - 0.5 delta). Consumes `inputs.in`.
 */
export const huber: EnginePiece<NodeOf<'huber'>> = {
  forward(node, inputs) {
    const d = inputs['in'];
    if (!d) throw new Error('A Huber piece is missing its input.');
    const a = np.abs(d.ref);
    const quad = np.square(d).mul(0.5);
    const lin = a.ref.sub(0.5 * node.delta).mul(node.delta);
    return np.where(a.less(node.delta), quad, lin);
  },
};
