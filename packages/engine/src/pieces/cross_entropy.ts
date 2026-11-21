import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import { logSoftmax } from './softmax.js';
import type { EnginePiece } from './types.js';

/**
 * Per example -log p[class], `[n, 1]`. `pred` is `[n, k]`, `target` is `[n, 1]` class indices as floats.
 * Consumes both.
 *
 * Fused (the softmax in front was folded in, `ctx.fused`): `pred` holds the logits and the value is
 * -logSoftmax(z)[class]: finite for any finite z. Otherwise `pred` is probabilities used as they are,
 * with no clipping and no requirement that a row sums to 1 (tinker-ddg): -log p[class].
 */
export const cross_entropy: EnginePiece<NodeOf<'cross_entropy'>> = {
  forward(node, inputs, _layer, ctx) {
    const pred = inputs['pred'];
    const y = inputs['target'];
    if (!pred || !y) throw new Error('The cross-entropy is missing its prediction or target.');
    const logp = ctx.fused(node.id) ? logSoftmax(pred) : np.log(pred);
    // Pick logp[class] with one static slice per class and a scalar comparison. np.arange, cumsum, astype of a comparison and
    // gather all capture array constants in the trace, and jax-js never frees those.
    const k = logp.shape[1] ?? 1;
    let picked: typeof logp | undefined;
    for (let j = 0; j < k; j++) {
      const col = (j < k - 1 ? logp.ref : logp).slice([], [j, j + 1]);
      const term = np.where((j < k - 1 ? y.ref : y).equal(j), col, 0);
      picked = picked ? picked.add(term) : term;
    }
    return (picked as NonNullable<typeof picked>).neg();
  },
};
