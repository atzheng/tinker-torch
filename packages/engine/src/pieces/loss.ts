import type { NodeOf } from '@tinker/core';
import { LOSSES } from '../losses.js';
import type { EnginePiece } from './types.js';

/**
 * inputs.pred is the PRE-link value, inputs.target is y. Returns the mean loss (no penalty).
 * Consumes both.
 */
export const loss: EnginePiece<NodeOf<'loss'>> = {
  implemented: true,
  forward(node, inputs, _layer, ctx) {
    const z = inputs['pred'];
    const y = inputs['target'];
    if (!z || !y) throw new Error('The loss is missing its prediction or target.');
    return LOSSES[node.fn].compute(node, z, y, ctx.link);
  },
};
