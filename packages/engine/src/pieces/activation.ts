import { nn, numpy as np } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { ActivationFn, NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * 0.5 * (1 + tanh(x / 2)). jax-js `nn.sigmoid` has a NaN gradient below about -88 (docs/spike.md
 * defect 3). Consumes `x`.
 */
export function safeSigmoid(x: JArray): JArray {
  return np.tanh(x.mul(0.5)).add(1).mul(0.5);
}

const FNS: Record<ActivationFn, (x: JArray) => JArray> = {
  sigmoid: safeSigmoid,
  tanh: (x) => np.tanh(x),
  // jax-js built-ins (owner decision, tinker-e1h.2): relu'(0) = 0.5 and gelu in its tanh form.
  relu: (x) => nn.relu(x),
  gelu: (x) => nn.gelu(x),
};

/** Consumes `inputs.in`. */
export const activation: EnginePiece<NodeOf<'activation'>> = {
  implemented: true,
  forward(node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('An Activation piece is missing its input.');
    return FNS[node.fn](x);
  },
};
