import { numpy as np } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { LossFn, NodeOf, OutputLink } from '@tinker/core';
import { applyLink } from './pieces/output.js';
import { notYet } from './pieces/types.js';

export type LossImpl = {
  implemented: boolean;
  /**
   * Mean loss over the batch, a scalar. `z` is the PRE-link prediction [n, width], `y` is [n]; the
   * loss applies (or fuses) the link itself. CONSUMES `z` and `y`.
   */
  compute(node: NodeOf<'loss'>, z: JArray, y: JArray, link: OutputLink): JArray;
};

function placeholder(name: string): LossImpl {
  return {
    implemented: false,
    compute() {
      return notYet(name);
    },
  };
}

export const LOSSES: Record<LossFn, LossImpl> = {
  squared_error: {
    implemented: true,
    compute(_node, z, y, link) {
      const pred = applyLink(link, z);
      const n = y.shape[0] ?? 0;
      return np.square(pred.sub(y.reshape([n, 1]))).mean();
    },
  },
  // tinker-hj4.2 fills these in (fused sigmoid + log_loss as logsumexp([z, 0]) - z*y, etc.).
  absolute_error: placeholder('Absolute error'),
  huber: placeholder('Huber loss'),
  log_loss: placeholder('Log loss'),
  cross_entropy: placeholder('Cross-entropy'),
};
