import { expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import { grad, jit, numpy as np } from '@jax-js/jax';
import { logLossFromLogit } from '../src/pieces/log_loss.js';

await createEngine();

test.each([
  [0, 0, 0.5],
  [0, 1, -0.5],
  [1000, 1, 0],
  [1000, 0, 1],
  [-1000, 1, -1],
  [-1000, 0, 0],
  [2, 1, 1 / (1 + Math.exp(-2)) - 1],
])('fused log loss gradient at z=%d, y=%d is sigmoid(z) - y', (z, y, want) => {
  const f = jit(grad((a: np.Array, b: np.Array) => np.sum(logLossFromLogit(a, b))));
  const g = f(np.array([[z]]), np.array([[y]]));
  const got = g.dataSync()[0] as number;
  expect(got).toBeCloseTo(want, 5);
});
