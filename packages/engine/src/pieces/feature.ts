import { numpy as np } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Feature expansion, columns in the CONTRACTS.md section 3 order: x^p for p = 1..degree (power-major,
 * then column), then sin(x), then cos(x), then x_i * x_j for i < j (row-major pairs). Consumes
 * `inputs.in` ([n, d]); returns [n, width].
 */
export const feature: EnginePiece<NodeOf<'feature'>> = {
  implemented: true,
  forward(node, inputs) {
    const x = inputs['in'];
    if (!x) throw new Error('A Feature piece is missing its input.');
    const d = x.shape[1] ?? 0;
    const parts: JArray[] = [x.ref];
    let power = x.ref;
    for (let p = 2; p <= node.degree; p++) {
      power = power.mul(x.ref);
      parts.push(power.ref);
    }
    power.dispose();
    if (node.sin) parts.push(np.sin(x.ref));
    if (node.cos) parts.push(np.cos(x.ref));
    if (node.products) {
      // Static slices, not index arrays: constants captured by a jitted function are never freed.
      const col = (i: number) => x.ref.slice([], [i, i + 1]);
      for (let i = 0; i < d; i++) {
        for (let j = i + 1; j < d; j++) parts.push(col(i).mul(col(j)));
      }
    }
    x.dispose();
    const [only] = parts;
    return parts.length === 1 && only ? only : np.concatenate(parts, 1);
  },
};
