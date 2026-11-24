import { expect, test } from 'vitest';
import { NODE_TYPES, PALETTE } from '@tinker/core';
import { REGIONS, WEB_DEPS } from '../src/index.js';
import { WEB_PIECES } from '../src/canvas/drawings/index.js';

test('web imports from core and engine', () => {
  expect(WEB_DEPS).toEqual(['@tinker/core', '@tinker/engine']);
  expect(REGIONS).toHaveLength(5);
});

test('WEB_PIECES has an entry for every piece type used by the palette', () => {
  expect(Object.keys(WEB_PIECES).sort()).toEqual([...NODE_TYPES].sort());
  for (const e of PALETTE) {
    if (e.kind === 'piece') expect(Object.keys(WEB_PIECES)).toContain(e.make().type);
  }
  expect(PALETTE).toHaveLength(33);
});
