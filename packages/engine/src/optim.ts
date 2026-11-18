import * as optax from '@jax-js/optax';
import type { TrainConfig } from '@tinker/core';

export type OptimImpl = { implemented: boolean; make(lr: number): optax.GradientTransformation };

export const OPTIMIZERS: Record<TrainConfig['optimizer'], OptimImpl> = {
  sgd: { implemented: true, make: (lr) => optax.sgd(lr) },
  // torch SGD(momentum 0.9, dampening 0): buf = 0.9 buf + g, first buf = g. optax's trace starts
  // at zero, so the first step is the same.
  momentum: { implemented: true, make: (lr) => optax.sgd(lr, { momentum: 0.9 }) },
  // Owner decision (CONTRACTS.md section 7): optax.adam as is (b1 0.9, b2 0.999, eps 1e-8), not torch-exact.
  adam: { implemented: true, make: (lr) => optax.adam(lr) },
};
