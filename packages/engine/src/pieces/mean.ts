import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Mean over the batch and the elements: per-example `[n, w]` -> a scalar. Consumes `inputs.in`. */
export const mean: EnginePiece<NodeOf<'mean'>> = {
  forward(_node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('A Mean piece is missing its input.');
    return x.mean();
  },
};
