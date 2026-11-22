import { expect, test } from 'vitest';
import { gridFor } from '../src/App.js';
import { layoutFor } from '../src/layout/layoutMode.js';
import {
  deltaAlong,
  dropIndex,
  extentAlong,
  midAlong,
  posAlong,
  resizePair,
  DEFAULT_WEIGHTS,
} from '../src/results/panelLayout.js';

test('the section 10 landscape viewports keep the side-by-side layout', () => {
  for (const [w, h] of [
    [1280, 720],
    [1920, 1080],
    [1024, 768],
    [1180, 820],
  ] as const) {
    expect(layoutFor(w, h)).toBe('landscape');
  }
});

test('portrait and narrow windows stack the regions', () => {
  for (const [w, h] of [
    [768, 1024],
    [820, 1180],
    [800, 900],
    [1000, 700], // narrower than 1024
    [1100, 1200], // wide enough but taller than wide
  ] as const) {
    expect(layoutFor(w, h)).toBe('portrait');
  }
  expect(layoutFor(1024, 1024)).toBe('landscape'); // square counts as landscape
});

test('the notice shows only below 600 px in either direction', () => {
  expect(layoutFor(599, 1000)).toBe('notice');
  expect(layoutFor(1200, 599)).toBe('notice');
  expect(layoutFor(600, 600)).toBe('notice'); // stacked layout needs 700 px of height
  expect(layoutFor(900, 699)).toBe('notice');
  expect(layoutFor(900, 700)).toBe('portrait');
  expect(layoutFor(1024, 600)).toBe('landscape');
  expect(layoutFor(390, 844)).toBe('notice');
});

test('gridFor in portrait: one column, hidden regions leave no row', () => {
  expect(
    gridFor({ top: true, palette: true, results: true, train: true }, 'portrait'),
  ).toBeUndefined();
  const g = gridFor({ top: true, palette: false, results: false, train: true }, 'portrait');
  expect(g?.gridTemplateAreas).toBe("'top' 'canvas' 'train'");
  expect(g?.gridTemplateColumns).toBe('minmax(0, 1fr)');
  expect(
    gridFor({ top: false, palette: true, results: true, train: false }, 'portrait')
      ?.gridTemplateAreas,
  ).toBe("'palette' 'canvas' 'results'");
});

test('panel axis maths: a row uses x, a column uses y', () => {
  const r = { left: 10, right: 30, top: 100, bottom: 160 };
  expect(midAlong('row', r)).toBe(20);
  expect(midAlong('column', r)).toBe(130);
  expect(posAlong('row', { x: 5, y: 9 })).toBe(5);
  expect(posAlong('column', { x: 5, y: 9 })).toBe(9);
  expect(deltaAlong('row', { dx: 3, dy: 4 })).toBe(3);
  expect(deltaAlong('column', { dx: 3, dy: 4 })).toBe(4);
  expect(extentAlong('row', { width: 700, height: 200 })).toBe(700);
  expect(extentAlong('column', { width: 700, height: 200 })).toBe(200);
});

test('side-by-side resize and reorder use widths and x positions', () => {
  // Two panels in a 600 px wide band: dragging the splitter 150 px right moves a quarter of the total.
  const w = resizePair(DEFAULT_WEIGHTS, ['fit', 'loss'], 'fit', 'loss', 150, 600);
  expect(w.fit).toBeCloseTo(1.5);
  expect(w.loss).toBeCloseTo(0.5);
  // Pointer at x = 350, other panels' midpoints at x = 100, 300 and 500: it lands at index 2.
  const mids = [100, 300, 500].map((m) =>
    midAlong('row', { left: m, right: m, top: 0, bottom: 0 }),
  );
  expect(dropIndex(mids, posAlong('row', { x: 350, y: 0 }))).toBe(2);
});
