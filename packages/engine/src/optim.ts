import * as optax from '@jax-js/optax';
import type { TrainConfig } from '@tinker/core';
import { notYet } from './pieces/types.js';

export type OptimImpl = { implemented: boolean; make(lr: number): optax.GradientTransformation };

export const OPTIMIZERS: Record<TrainConfig['optimizer'], OptimImpl> = {
  sgd: { implemented: true, make: (lr) => optax.sgd(lr) },
  // tinker-hj4.4: optax.sgd(lr, { momentum: 0.9 }) and optax.adam(lr, ...).
  momentum: { implemented: false, make: () => notYet('Momentum') },
  adam: { implemented: false, make: () => notYet('Adam') },
};
