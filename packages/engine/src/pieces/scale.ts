import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** c * x, elementwise. Consumes `inputs.in`. */
export const scale: EnginePiece<v2.NodeOf<'scale'>> = {
  forward(node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('A Scale piece is missing its input.');
    return x.mul(node.c);
  },
};
