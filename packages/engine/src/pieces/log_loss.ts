import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Per example -(y log p + (1 - y) log(1 - p)), `[n, 1]`. Consumes `pred` and `target` (both `[n, 1]`).
 *
 * Fused (the sigmoid in front was folded in, `ctx.fused`): `pred` is the logit z and the value is
 * logsumexp([z, 0]) - z*y, computed as max(z, 0) + log1p(exp(-|z|)) - z*y: finite for any finite z (jax-js softplus/logSigmoid overflow in float32,
 * docs/spike.md). Otherwise `pred` is a probability, clipped to [1e-7, 1 - 1e-7] so a stray 0 or 1
 * cannot give inf; analyse only lets a sigmoid link feed this piece, so that path is a safety net.
 */
export const log_loss: EnginePiece<NodeOf<'log_loss'>> = {
  forward(node, inputs, _layer, ctx) {
    const pred = inputs['pred'];
    const y = inputs['target'];
    if (!pred || !y) throw new Error('The log loss is missing its prediction or target.');
    if (ctx.fused(node.id)) {
      const sp = np.maximum(pred.ref, 0).add(np.log1p(np.exp(np.abs(pred.ref).neg())));
      return sp.sub(pred.mul(y));
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
