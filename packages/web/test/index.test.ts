import { expect, test } from 'vitest';
import { REGIONS, WEB_DEPS } from '../src/index.js';

test('web imports from core and engine', () => {
  expect(WEB_DEPS).toEqual(['@tinker/core', '@tinker/engine']);
  expect(REGIONS).toHaveLength(5);
});
