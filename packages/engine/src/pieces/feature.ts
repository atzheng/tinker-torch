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
    if (node.products && d > 1) {
      const left: number[] = [];
      const right: number[] = [];
      for (let i = 0; i < d; i++) {
        for (let j = i + 1; j < d; j++) {
          left.push(i);
          right.push(j);
        }
      }
      const idx = (v: number[]) => np.array(v, { dtype: np.int32 });
      parts.push(np.take(x.ref, idx(left), 1).mul(np.take(x.ref, idx(right), 1)));
    }
    x.dispose();
    const [only] = parts;
    return parts.length === 1 && only ? only : np.concatenate(parts, 1);
  },
};
