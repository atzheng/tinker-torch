import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Elementwise a + b (skip connection). Consumes `inputs.a` and `inputs.b`. */
export const add: EnginePiece<NodeOf<'add'>> = {
  implemented: true,
  forward(_node, inputs) {
    const a = inputs['a'];
    const b = inputs['b'];
    if (!a || !b) throw new Error('An Add piece needs both of its inputs.');
    return a.add(b);
  },
};
