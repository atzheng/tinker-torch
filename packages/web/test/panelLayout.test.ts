import { describe, expect, test } from 'vitest';
import {
  DEFAULT_ORDER,
  DEFAULT_WEIGHTS,
  KEY_STEP_SHARE,
  KEY_STEP_SHARE_BIG,
  MIN_SHARE,
  pairValues,
  stepPair,
  type PanelWeights,
  dropIndex,
  movePanel,
  resizePair,
  type PanelId,
} from '../src/results/panelLayout.js';

const THREE: readonly PanelId[] = ['fit', 'loss', 'surface'];
const all = new Set<PanelId>(['fit', 'loss', 'surface']);
const two = new Set<PanelId>(['fit', 'loss']);

test('the histogram panel is part of the default order', () => {
  expect(DEFAULT_ORDER).toContain('histogram');
  expect(DEFAULT_WEIGHTS.histogram).toBe(1);
});

describe('movePanel', () => {
  test('moves a panel to an index among the others', () => {
    expect(movePanel(THREE, all, 'fit', 1)).toEqual(['loss', 'fit', 'surface']);
    expect(movePanel(THREE, all, 'fit', 2)).toEqual(['loss', 'surface', 'fit']);
    expect(movePanel(THREE, all, 'surface', 0)).toEqual(['surface', 'fit', 'loss']);
    expect(movePanel(THREE, all, 'loss', 1)).toEqual(['fit', 'loss', 'surface']);
  });
  test('a hidden panel keeps its place beside its neighbours', () => {
    expect(movePanel(['fit', 'surface', 'loss'], two, 'fit', 1)).toEqual([
      'surface',
      'loss',
      'fit',
    ]);
    expect(movePanel(['fit', 'surface', 'loss'], two, 'loss', 0)).toEqual([
      'loss',
      'fit',
      'surface',
    ]);
  });
  test('clamps the index', () => {
    expect(movePanel(THREE, all, 'fit', 99)).toEqual(['loss', 'surface', 'fit']);
    expect(movePanel(THREE, all, 'surface', -3)).toEqual(['surface', 'fit', 'loss']);
  });
});

describe('dropIndex', () => {
  test('counts the other panels whose midpoint is above the pointer', () => {
    expect(dropIndex([100, 300], 50)).toBe(0);
    expect(dropIndex([100, 300], 200)).toBe(1);
    expect(dropIndex([100, 300], 500)).toBe(2);
  });
});

describe('resizePair', () => {
  const vis: PanelId[] = ['fit', 'loss'];
  test('moves weight in proportion to the pixels and keeps the pair sum', () => {
    const w = resizePair(DEFAULT_WEIGHTS, vis, 'fit', 'loss', 100, 400);
    expect(w.fit).toBeCloseTo(1.5);
    expect(w.loss).toBeCloseTo(0.5);
    expect(w.surface).toBe(1);
  });
  test('neither neighbour falls below the minimum share', () => {
    const down = resizePair(DEFAULT_WEIGHTS, vis, 'fit', 'loss', 10_000, 400);
    expect(down.loss).toBeCloseTo(MIN_SHARE * 2);
    const up = resizePair(DEFAULT_WEIGHTS, vis, 'fit', 'loss', -10_000, 400);
    expect(up.fit).toBeCloseTo(MIN_SHARE * 2);
    expect(up.fit + up.loss).toBeCloseTo(2);
  });
  test('never mutates its input', () => {
    const start = { ...DEFAULT_WEIGHTS };
    resizePair(start, vis, 'fit', 'loss', 50, 400);
    expect(start).toEqual(DEFAULT_WEIGHTS);
  });
});

describe('stepPair and pairValues (keyboard resize, aria values)', () => {
  const vis: PanelId[] = ['fit', 'loss'];
  test('an arrow key moves a fixed share of the total, Shift a bigger one', () => {
    const down = stepPair(DEFAULT_WEIGHTS, vis, 'fit', 'loss', 1, false);
    expect(down.fit).toBeCloseTo(1 + KEY_STEP_SHARE * 2);
    expect(down.fit + down.loss).toBeCloseTo(2);
    const bigUp = stepPair(DEFAULT_WEIGHTS, vis, 'fit', 'loss', -1, true);
    expect(bigUp.fit).toBeCloseTo(1 - KEY_STEP_SHARE_BIG * 2);
    expect(bigUp.surface).toBe(1);
  });
  test('repeated key presses stop at the minimum share', () => {
    let w: PanelWeights = { ...DEFAULT_WEIGHTS };
    for (let i = 0; i < 40; i++) w = stepPair(w, vis, 'fit', 'loss', -1, true);
    expect(w.fit).toBeCloseTo(MIN_SHARE * 2);
    for (let i = 0; i < 40; i++) w = stepPair(w, vis, 'fit', 'loss', 1, true);
    expect(w.loss).toBeCloseTo(MIN_SHARE * 2);
  });
  test('pairValues gives the upper share and the bounds the minimum allows', () => {
    expect(pairValues(DEFAULT_WEIGHTS, vis, 'fit', 'loss')).toEqual({ now: 50, min: 15, max: 85 });
  });
});

test('panelListed and lossChartShown: panel-score shows the Loss panel without its chart', async () => {
  const { panelListed, lossChartShown } = await import('../src/results/panelLayout.js');
  expect(panelListed(undefined, 'loss')).toBe(true);
  expect(lossChartShown(undefined)).toBe(true);
  expect(panelListed(['panel-loss'], 'loss')).toBe(true);
  expect(lossChartShown(['panel-loss'])).toBe(true);
  expect(panelListed(['panel-score'], 'loss')).toBe(true);
  expect(lossChartShown(['panel-score'])).toBe(false);
  expect(panelListed(['panel-score'], 'fit')).toBe(false);
  expect(panelListed(['panel-fit'], 'loss')).toBe(false);
});
