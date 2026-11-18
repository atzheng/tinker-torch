import { numpy as np } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { Penalty } from '@tinker/core';
import type { ParamTree } from './convert.js';

/** The weight penalty (L1 / L2 over Linear W only, never biases). */
export const PENALTY_IMPLEMENTED = true;

/**
 * Penalty term lambda * R(W) as a scalar: R = sum w^2 (l2) or sum |w| (l1) over every Linear W
 * (np.abs: |w|'(0) = 1, owner decision). CONSUMES `params` (the caller passes `tree.ref(params)`).
 */
export function penaltyTerm(penalty: Penalty, params: ParamTree): JArray {
  let total = np.array(0);
  for (const layer of Object.values(params)) {
    const { W, b } = layer;
    b?.dispose();
    total = total.add((penalty.kind === 'l2' ? np.square(W) : np.abs(W)).sum());
  }
  return total.mul(penalty.lambda);
}
