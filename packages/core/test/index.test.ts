import { expect, test } from 'vitest';
import { CORE_NAME } from '../src/index.js';

test('core exports its name', () => {
  expect(CORE_NAME).toBe('@tinker/core');
});
