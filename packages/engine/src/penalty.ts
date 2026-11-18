import type { Penalty } from '@tinker/core';
import { notYet } from './pieces/types.js';

/** The weight penalty (L1 / L2 over Linear W only). Placeholder: tinker-hj4.2 sets this to true. */
export const PENALTY_IMPLEMENTED = false;

/** Penalty term lambda * R(W). Will CONSUME the weights it is given (use tree.ref before). */
export function penaltyTerm(_penalty: Penalty): never {
  return notYet('The weight penalty');
}
