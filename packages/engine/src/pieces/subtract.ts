import type { v2 } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** a - b, same shape. Consumes both inputs. */
export const subtract: EnginePiece<v2.NodeOf<'subtract'>> = {
  forward(_node, inputs) {
    const a = inputs['a'];
    const b = inputs['b'];
    if (!a || !b) throw new Error('A Subtract piece needs both of its inputs.');
    return a.sub(b);
  },
};
