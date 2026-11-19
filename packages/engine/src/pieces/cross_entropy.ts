import { nn, numpy as np } from '@jax-js/jax';
import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Per example -log p[class], `[n, 1]`. `pred` is `[n, k]`, `target` is `[n, 1]` class indices as floats.
 * Consumes both.
 *
 * Fused (the softmax in front was folded in, `ctx.fused`): `pred` holds the logits and the value is
 * -logSoftmax(z)[class]: finite for any finite z. Otherwise `pred` is probabilities, clipped to
 * [1e-7, 1] before the log (safety net: analyse only lets a softmax link feed this piece).
 */
export const cross_entropy: EnginePiece<v2.NodeOf<'cross_entropy'>> = {
  forward(node, inputs, _layer, ctx) {
    const pred = inputs['pred'];
    const y = inputs['target'];
    if (!pred || !y) throw new Error('The cross-entropy is missing its prediction or target.');
    const k = pred.shape[1] ?? 1;
    const logp = ctx.fused(node.id) ? nn.logSoftmax(pred, -1) : np.log(np.clip(pred, 1e-7, 1));
    const onehot = np.arange(k).reshape([1, k]).equal(y).astype(logp.dtype);
    return logp.mul(onehot).sum(1, { keepdims: true }).neg();
  },
};
