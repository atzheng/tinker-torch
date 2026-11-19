import type { Array as JArray } from '@jax-js/jax';
import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * One dataset column as `[n, 1]` (x[:, idx]). Has no input port. Takes the batch x through the
 * context (one reference, consumed here), like an Input.
 */
export const column: EnginePiece<NodeOf<'column'>> = {
  forward(node, _inputs, _layer, ctx) {
    const x = ctx.takeX();
    const idx = ctx.dataColumns.indexOf(node.name);
    if (idx < 0) {
      x.dispose();
      throw new Error('A Column names a column that is not in the dataset.');
    }
    // a static slice, not an index array: constants captured by a jitted function are never freed
    return x.slice([], [idx, idx + 1]);
  },
};

/** in * W with W `[1, 1]` broadcasting over `[n, 1]`. Consumes `inputs.in` and the layer. */
export const scalew: EnginePiece<NodeOf<'scalew'>> = {
  forward(_node, inputs, layer) {
    const x = inputs['in'];
    const W = layer?.['W'];
    if (!x || !W) throw new Error('A Weight piece is missing its input or parameter.');
    return x.mul(W);
  },
};

/**
 * in + b with b `[1]`. Consumes `inputs.in` and the layer. With nothing wired in, b broadcast to
 * `[n, 1]`: takes the batch x through the context (one reference, consumed here) only to read n
 * statically, so no array constant is created inside a jit.
 */
export const bias: EnginePiece<NodeOf<'bias'>> = {
  forward(_node, inputs, layer, ctx) {
    const x = inputs['in'];
    const b = layer?.['b'];
    if (!b) throw new Error('A Bias piece is missing its parameter.');
    if (!x) {
      const bx = ctx.takeX();
      const n = bx.shape[0] ?? 0;
      bx.dispose();
      return np.broadcastTo(b, [n, 1]);
    }
    return x.add(b);
  },
};

/**
 * Adds all its inputs. The walker passes a variadic port's wires as `in`, `in#1`, `in#2`, ... in
 * `analysis.inputs` (edge) order; they are folded in that order. Consumes every input.
 */
export const sum: EnginePiece<NodeOf<'sum'>> = {
  forward(_node, inputs) {
    const xs: JArray[] = Object.values(inputs);
    const [first, ...rest] = xs;
    if (!first) throw new Error('A Sum piece needs at least one input.');
    return rest.reduce((acc, v) => acc.add(v), first);
  },
};
