import { expect, test } from 'vitest';
import { PALETTE } from '@tinker/core';
import { REGIONS, WEB_DEPS } from '../src/index.js';
import { WEB_PIECES } from '../src/canvas/drawings/index.js';

test('web imports from core and engine', () => {
  expect(WEB_DEPS).toEqual(['@tinker/core', '@tinker/engine']);
  expect(REGIONS).toHaveLength(5);
});

test('WEB_PIECES has an entry for every piece type used by the palette', () => {
  expect(Object.keys(WEB_PIECES).sort()).toEqual([
    'activation',
    'add',
    'dropout',
    'feature',
    'input',
    'linear',
    'loss',
    'output',
    'target',
  ]);
  expect(PALETTE).toHaveLength(14);
});
