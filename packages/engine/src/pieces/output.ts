import { nn } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { NodeOf, OutputLink } from '@tinker/core';
import { safeSigmoid } from './activation.js';
import type { EnginePiece } from './types.js';

/** Applies the link to the pre-link value `z` [n, width]. Consumes `z`. */
export function applyLink(link: OutputLink, z: JArray): JArray {
  if (link === 'identity') return z;
  if (link === 'sigmoid') return safeSigmoid(z);
  return nn.softmax(z, -1);
}

/**
 * Returns the post-link value, or `z` itself when `ctx.rawOutput` (the loss then owns the link).
 * Consumes `inputs.in`.
 */
export const output: EnginePiece<NodeOf<'output'>> = {
  implemented: true,
  forward(node, inputs, _layer, ctx) {
    const z = inputs['in'];
    if (!z) throw new Error('An Output piece is missing its input.');
    return ctx.rawOutput ? z : applyLink(node.link, z);
  },
};
