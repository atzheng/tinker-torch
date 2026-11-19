import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** The dataset's y as `[n, 1]` (a class index for classification). */
export const target: EnginePiece<NodeOf<'target'>> = {
  forward(_node, _inputs, _layer, ctx) {
    const y = ctx.takeY();
    return y.reshape([y.shape[0] ?? 0, 1]);
  },
};
