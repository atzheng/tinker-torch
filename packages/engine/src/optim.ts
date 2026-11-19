import { numpy as np, tree } from '@jax-js/jax';
import * as optax from '@jax-js/optax';
import type { TrainConfig } from '@tinker/core';

export type OptimImpl = { implemented: boolean; make(lr: number): optax.GradientTransformation };

type Tree = Parameters<optax.GradientTransformation['init']>[0];

/**
 * Adam with a monotone second moment (b1 0.9, b2 0.999, eps 1e-8): the denominator is
 * `sqrt(max over t of nuHat_t) + eps`, where `nuHat` is the bias-corrected second moment. This is
 * AMSGrad applied to the bias-corrected estimate. Plain Adam's denominator shrinks as the
 * gradients vanish near a minimum, so the step grows back to about lr and the loss spikes at
 * roughly even intervals (tinker-df1, CONTRACTS.md section 7); the running maximum prevents that.
 * `torch.optim.Adam(amsgrad=True)` is NOT the same: it takes the maximum of the raw `nu` and then
 * bias-corrects, so its denominator still shrinks for the first few thousand steps and it still
 * spikes there. optax 0.1.3 has no AMSGrad, so this is `scaleByAdam` plus the maximum.
 *
 * Like every optax transformation, `update` consumes `updates`, `state` and `params`; `init`
 * consumes `params`.
 */
function scaleByAdamMax(b1 = 0.9, b2 = 0.999, eps = 1e-8): optax.GradientTransformation {
  return {
    init(params) {
      return {
        count: np.array(0, { dtype: np.uint32 }),
        mu: optax.treeZerosLike(tree.ref(params)),
        nu: optax.treeZerosLike(tree.ref(params)),
        nuMax: optax.treeZerosLike(params),
      };
    },
    update(updates, state, params) {
      tree.dispose(params);
      // jax-js trees are untyped here; the state shape is the one built in `init`.
      const s = state as { count: np.Array; mu: Tree; nu: Tree; nuMax: Tree };
      const mu = optax.treeUpdateMoment(tree.ref(updates), s.mu, b1, 1);
      const nu = optax.treeUpdateMoment(updates, s.nu, b2, 2);
      const count = s.count.add(1);
      const muHat = optax.treeBiasCorrection(tree.ref(mu), b1, count.ref);
      const nuHat = optax.treeBiasCorrection(tree.ref(nu), b2, count.ref);
      const nuMax = tree.map((a: np.Array, b: np.Array) => np.maximum(a, b), s.nuMax, nuHat) as Tree;
      const out = tree.map(
        (m: np.Array, v: np.Array) => m.div(np.sqrt(v).add(eps)),
        muHat,
        tree.ref(nuMax),
      ) as unknown as typeof updates;
      return [out, { count, mu, nu, nuMax }];
    },
  };
}

export const OPTIMIZERS: Record<TrainConfig['optimizer'], OptimImpl> = {
  sgd: { implemented: true, make: (lr) => optax.sgd(lr) },
  // torch SGD(momentum 0.9, dampening 0): buf = 0.9 buf + g, first buf = g. optax's trace starts
  // at zero, so the first step is the same.
  momentum: { implemented: true, make: (lr) => optax.sgd(lr, { momentum: 0.9 }) },
  // Plain Adam spikes periodically near convergence; the default keeps a running maximum of the
  // bias-corrected second moment (CONTRACTS.md section 7, tinker-df1).
  adam: {
    implemented: true,
    make: (lr) => optax.chain(scaleByAdamMax(), optax.scaleByLearningRate(lr)),
  },
};
