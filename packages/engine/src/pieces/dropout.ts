import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Inverted dropout (nn.Dropout): while training, x * mask / (1 - p); otherwise the identity.
 * The 0/1 keep mask comes from `ctx.dropoutMask` (sampled outside the differentiated function by
 * the step; injected by the test-only seam, CONTRACTS.md section 11). Consumes `inputs.in`.
 */
export const dropout: EnginePiece<v2.NodeOf<'dropout'>> = {
  forward(node, inputs, _layer, ctx) {
    const x = inputs['in'];
    if (!x) throw new Error('A Dropout piece is missing its input.');
    if (!ctx.train || node.p <= 0) return x;
    const mask = ctx.dropoutMask(node.id);
    return x.mul(mask).mul(1 / (1 - node.p));
  },
};
