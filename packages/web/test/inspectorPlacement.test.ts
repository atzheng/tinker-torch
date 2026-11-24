import { expect, test } from 'vitest';
import { fitInspector } from '../src/canvas/inspectorPlacement';

const overlaps = (
  p: { top: number; maxHeight: number },
  h: number,
  t: { top: number; bottom: number },
) => {
  const bottom = p.top + Math.min(h, p.maxHeight);
  return p.top < t.bottom && bottom > t.top;
};

test('goes below when the whole panel fits, centred and kept in view', () => {
  expect(
    fitInspector({ cx: 100, top: 100, bottom: 130 }, { w: 700, h: 600 }, 248, 300),
  ).toMatchObject({
    left: 4,
    top: 146,
  });
  expect(fitInspector({ cx: 690, top: 100, bottom: 130 }, { w: 700, h: 600 }, 248, 300).left).toBe(
    448,
  );
});

test('goes above when only there it fits, flush with the target', () => {
  const p = fitInspector({ cx: 300, top: 450, bottom: 480 }, { w: 700, h: 600 }, 248, 300);
  expect(p.top + 300).toBe(450 - 16);
});

test('a tall measured panel near the bottom is placed whole above, not cut off', () => {
  const view = { w: 768, h: 508 };
  const target = { cx: 300, top: 300, bottom: 420 };
  const p = fitInspector(target, view, 248, 280);
  expect(p.top + 280).toBeLessThanOrEqual(view.h);
  expect(overlaps(p, 280, target)).toBe(false);
});

test('never covers the target: when nothing fits it scrolls on the larger side', () => {
  const view = { w: 768, h: 400 };
  for (const [top = 0, bottom = 0] of [
    [150, 250],
    [20, 120],
    [280, 380],
    [200, 205],
    [0, 400],
  ]) {
    const target = { cx: 300, top, bottom };
    const p = fitInspector(target, view, 248, 500);
    expect(p.maxHeight).toBeLessThan(500);
    if (top > 0 || bottom < 400) expect(overlaps(p, 500, target), `${top}-${bottom}`).toBe(false);
    if (top > 0 || bottom < 400) expect(p.top).toBeGreaterThanOrEqual(4);
  }
});

test('never reaches above minTop (the header strip)', () => {
  const p = fitInspector({ cx: 300, top: 200, bottom: 380 }, { w: 768, h: 420 }, 248, 400, 16, 44);
  expect(p.top).toBeGreaterThanOrEqual(44);
  expect(p.top + p.maxHeight).toBeLessThanOrEqual(200 - 16);
});

test('no room above or below and no x extent: clamps to the larger side, never over the target', () => {
  // view 768x200, target 80..120, minTop 44, height 300
  const a = fitInspector({ cx: 300, top: 80, bottom: 120 }, { w: 768, h: 200 }, 248, 300, 16, 44);
  expect(overlaps(a, 300, { top: 80, bottom: 120 })).toBe(false);
  // view 768x440, target 100..400
  const t = { top: 100, bottom: 400 };
  const b = fitInspector({ cx: 300, ...t }, { w: 768, h: 440 }, 248, 300, 16, 44);
  expect(overlaps(b, 300, t)).toBe(false);
  expect(b.maxHeight).toBeGreaterThanOrEqual(0);
});

test('goes beside the target when neither side has room', () => {
  const t = { cx: 200, top: 100, bottom: 400, x0: 150, x1: 250 };
  const r = fitInspector(t, { w: 768, h: 440 }, 248, 300, 16, 44);
  expect(r.left).toBe(266);
  expect(r.top).toBeGreaterThanOrEqual(44);
  expect(r.top + Math.min(300, r.maxHeight)).toBeLessThanOrEqual(436);
  // right edge too close: goes left
  const l = fitInspector(
    { cx: 600, top: 100, bottom: 400, x0: 550, x1: 650 },
    { w: 768, h: 440 },
    248,
    300,
    16,
    44,
  );
  expect(l.left).toBe(550 - 16 - 248);
});
