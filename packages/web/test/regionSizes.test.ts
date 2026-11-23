import { expect, test } from 'vitest';
import {
  KEY_STEP,
  KEY_STEP_BIG,
  REGION_LIMITS,
  clampSize,
  dragSize,
  pctCss,
  percent,
  stepSize,
} from '../src/layout/regionSizes.js';

test('dragSize: pixels over the extent, signed, clamped to the limits', () => {
  expect(dragSize('resultsWidth', 0.3, -100, 1000, -1)).toBeCloseTo(0.4);
  expect(dragSize('resultsWidth', 0.3, 100, 1000, -1)).toBeCloseTo(0.2);
  expect(dragSize('instructions', 0.28, 100, 1000, 1)).toBeCloseTo(0.38);
  expect(dragSize('resultsWidth', 0.3, -1e6, 1000, -1)).toBe(REGION_LIMITS.resultsWidth.max);
  expect(dragSize('resultsBand', 0.3, 1e6, 1000, -1)).toBe(REGION_LIMITS.resultsBand.min);
  expect(dragSize('instructions', 0.3, 5, 0, 1)).toBe(REGION_LIMITS.instructions.max);
});

test('stepSize: arrow and Shift+arrow steps, clamped', () => {
  expect(stepSize('resultsBand', 0.3, -1, false, -1)).toBeCloseTo(0.3 + KEY_STEP);
  expect(stepSize('resultsBand', 0.3, 1, true, -1)).toBeCloseTo(0.3 - KEY_STEP_BIG);
  expect(stepSize('instructions', 0.3, 1, false, 1)).toBeCloseTo(0.3 + KEY_STEP);
  let v = 0.3;
  for (let i = 0; i < 100; i++) v = stepSize('resultsWidth', v, 1, true, -1);
  expect(v).toBe(REGION_LIMITS.resultsWidth.min);
});

test('limits are ordered and contain the defaults', () => {
  for (const l of Object.values(REGION_LIMITS)) {
    expect(l.min).toBeLessThan(l.def);
    expect(l.def).toBeLessThan(l.max);
  }
  expect(clampSize('resultsWidth', 0.9)).toBe(0.5);
});

test('formatting helpers', () => {
  expect(percent(0.304)).toBe(30);
  expect(pctCss(0.4)).toBe('40%');
  expect(pctCss(0.12345)).toBe('12.35%');
});
