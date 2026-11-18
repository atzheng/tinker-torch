import { numpy as np } from '@jax-js/jax';
import { expect, test } from 'vitest';
import { ENGINE_CORE, ENGINE_NAME } from '../src/index.js';

test('engine exports its name and sees core', () => {
  expect(ENGINE_NAME).toBe('@tinker/engine');
  expect(ENGINE_CORE).toBe('@tinker/core');
});

test('jax-js runs in Node', () => {
  const x = np.array([1, 2, 3]);
  const y = x.mul(2);
  expect(Array.from(y.dataSync())).toEqual([2, 4, 6]);
});
