import { nn } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import { safeSigmoid } from './activation.js';
import type { EnginePiece } from './types.js';

/**
 * sigmoid (tanh form) or softmax over the last axis. Always the semantic function: when a loss
 * fuses it, the loss reads the link's input instead and this piece is skipped in the loss pass
 * (unless something else needs it). Consumes `inputs.in`.
 */
export const link: EnginePiece<NodeOf<'link'>> = {
  forward(node, inputs) {
    const z = inputs['in'];
    if (!z) throw new Error('A Link piece is missing its input.');
    return node.fn === 'sigmoid' ? safeSigmoid(z) : nn.softmax(z, -1);
  },
};
