import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * The tap: the weight matrix W [in, out] of the Linear `of` (never its bias). Reads parameters, so
 * gradients flow back into W. `layer` is that Linear's parameters (one reference for this use);
 * the bias reference is released. Consumes `layer`.
 */
export const weights: EnginePiece<NodeOf<'weights'>> = {
  forward(_node, _inputs, layer) {
    if (!layer) throw new Error('A Weights piece cannot find its Linear.');
    layer.b?.dispose();
    return layer.W;
  },
};
