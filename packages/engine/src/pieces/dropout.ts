import { numpy as np, random } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Inverted dropout (nn.Dropout): while training, x * mask / (1 - p); otherwise the identity.
 * The keep mask comes from `ctx.dropoutMask` (the test-only injection seam, CONTRACTS.md section
 * 11) or from `random.bernoulli` with a fresh subkey from `ctx.nextKey()`. Consumes `inputs.in`.
 */
export const dropout: EnginePiece<NodeOf<'dropout'>> = {
  implemented: true,
  forward(node, inputs, _layer, ctx) {
    const x = inputs['in'];
    if (!x) throw new Error('A Dropout piece is missing its input.');
    if (!ctx.train || node.p <= 0) return x;
    const injected = ctx.dropoutMask?.(node.id);
    const mask =
      injected ?? random.uniform(ctx.nextKey(), x.shape).greaterEqual(node.p).astype(np.float32);
    return x.mul(mask).mul(1 / (1 - node.p));
  },
};
