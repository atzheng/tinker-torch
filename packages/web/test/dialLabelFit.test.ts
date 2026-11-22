import { LAYOUT } from '@tinker/core';
import { describe, expect, it } from 'vitest';
import { estimateLabelWidth, fitDialName } from '../src/canvas/dialLabelFit.js';

describe('fitDialName', () => {
  it('keeps the delivery zone names', () => {
    for (const n of ['centre', 'north', 'east', 'south']) expect(fitDialName(n)).toBe(n);
  });
  it('cuts long names to the unit column', () => {
    for (const n of ['Northeastern suburbs', 'WWWWWWWW', 'international']) {
      const t = fitDialName(n);
      expect(t.endsWith('…')).toBe(true);
      expect(estimateLabelWidth(t)).toBeLessThanOrEqual(LAYOUT.unitW);
    }
  });
});
