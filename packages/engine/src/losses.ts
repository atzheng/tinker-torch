import { nn, numpy as np } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { LossFn, NodeOf, OutputLink } from '@tinker/core';
import { applyLink } from './pieces/output.js';

export type LossImpl = {
  implemented: boolean;
  /**
   * Mean loss over the batch, a scalar. `z` is the PRE-link prediction [n, width], `y` is [n]; the
   * loss applies (or fuses) the link itself. CONSUMES `z` and `y`.
   */
  compute(node: NodeOf<'loss'>, z: JArray, y: JArray, link: OutputLink): JArray;
};

export const LOSSES: Record<LossFn, LossImpl> = {
  squared_error: {
    implemented: true,
    compute(_node, z, y, link) {
      const pred = applyLink(link, z);
      const n = y.shape[0] ?? 0;
      return np.square(pred.sub(y.reshape([n, 1]))).mean();
    },
  },
  absolute_error: {
    implemented: true,
    compute(_node, z, y, link) {
      const pred = applyLink(link, z);
      const n = y.shape[0] ?? 0;
      // np.abs: |x|'(0) = 1 (owner decision), unlike torch's 0
      return np.abs(pred.sub(y.reshape([n, 1]))).mean();
    },
  },
  huber: {
    implemented: true,
    compute(node, z, y, link) {
      const delta = node.delta ?? 1;
      const pred = applyLink(link, z);
      const n = y.shape[0] ?? 0;
      const d = pred.sub(y.reshape([n, 1]));
      const a = np.abs(d.ref);
      // nn.HuberLoss(delta): 0.5 d^2 inside delta, delta * (|d| - 0.5 delta) outside
      const quad = np.square(d).mul(0.5);
      const lin = a.ref.sub(0.5 * delta).mul(delta);
      return np.where(a.less(delta), quad, lin).mean();
    },
  },
  log_loss: {
    implemented: true,
    compute(_node, z, y, link) {
      const n = y.shape[0] ?? 0;
      if (link !== 'sigmoid') {
        // analyse() only allows a sigmoid output with the log loss; this keeps other links defined
        const p = np.clip(applyLink(link, z), 1e-7, 1 - 1e-7);
        const yy = y.reshape([n, 1]);
        const pos = yy.ref.mul(np.log(p.ref));
        const neg = yy
          .neg()
          .add(1)
          .mul(np.log(p.neg().add(1)));
        return pos.add(neg).neg().mean();
      }
      // BCE with logits: logsumexp([z, 0]) - z*y, finite at +-100 (docs/spike.md)
      const zz = z.reshape([n]);
      const stacked = np.stack([zz.ref, np.zeros([n])]);
      return nn.logsumexp(stacked, 0).sub(zz.mul(y)).mean();
    },
  },
  cross_entropy: {
    implemented: true,
    compute(_node, z, y, link) {
      const n = y.shape[0] ?? 0;
      const k = z.shape[1] ?? 1;
      const logp = nn.logSoftmax(link === 'softmax' ? z : applyLink(link, z), -1);
      const onehot = np
        .arange(k)
        .reshape([1, k])
        .equal(y.reshape([n, 1]))
        .astype(logp.dtype);
      return logp.mul(onehot).sum(1).neg().mean();
    },
  },
};
