import { numpy as np } from '@jax-js/jax';

// jax-js nn.softmax / nn.logSoftmax / nn.logsumexp capture an array constant per trace, and jax-js never
// frees captured constants, so every compile would leak buffers. These are written out with np ops. Both consume `z` (`[n, k]`).

/** z - max(z) - log(sum(exp(z - max(z)))) along axis 1. */
export function logSoftmax(z: np.Array): np.Array {
  const shifted = z.ref.sub(np.max(z, -1, { keepdims: true }));
  return shifted.ref.sub(np.log(np.sum(np.exp(shifted.ref), -1, { keepdims: true })));
}

export function softmax(z: np.Array): np.Array {
  return np.exp(logSoftmax(z));
}
