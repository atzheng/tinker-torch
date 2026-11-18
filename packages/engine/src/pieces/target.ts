import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Gives y, shape [n]. */
export const target: EnginePiece<NodeOf<'target'>> = {
  implemented: true,
  forward(_node, _inputs, _layer, ctx) {
    return ctx.takeY();
  },
};
