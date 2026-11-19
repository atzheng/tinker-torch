import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Stable logistic loss on a logit: max(z,0) + log1p(exp(-|z|)) - z*y, finite for any finite z.
 * max(z,0) is written (z+|z|)/2 so the gradient at z = 0 is 0.5 - y. Consumes `z` and `y`.
 * No array constants: jax-js never frees constants captured in a trace.
 */
export function logLossFromLogit(z: np.Array, y: np.Array): np.Array {
  const relu = z.ref.add(np.abs(z.ref)).mul(0.5);
  return relu.add(np.log1p(np.exp(np.abs(z.ref).neg()))).sub(z.mul(y));
}

/**
 * Per example -(y log p + (1 - y) log(1 - p)), `[n, 1]`. Consumes `pred` and `target`.
 * Fused (`ctx.fused`): `pred` is the logit, see `logLossFromLogit`. Otherwise `pred` is a
 * probability clipped to [1e-7, 1 - 1e-7] (safety net: analyse only lets a sigmoid feed this).
 */
export const log_loss: EnginePiece<NodeOf<'log_loss'>> = {
  forward(node, inputs, _layer, ctx) {
    const pred = inputs['pred'];
    const y = inputs['target'];
    if (!pred || !y) throw new Error('The log loss is missing its prediction or target.');
    if (ctx.fused(node.id)) {
      return logLossFromLogit(pred, y);
    }
    const p = np.clip(pred, 1e-7, 1 - 1e-7);
    const pos = y.ref.mul(np.log(p.ref));
    const neg = y
      .neg()
      .add(1)
      .mul(np.log(p.neg().add(1)));
    return pos.add(neg).neg();
  },
};
