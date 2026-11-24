import { expect, test } from 'vitest';
import {
  formatRoute,
  parseRoute,
  routePreset,
  routeStep,
  stageNeighbours,
} from '../src/routes/lectureRoute.js';
import { LECTURES, lecturePresetNames, stagePresetName } from '../src/presets/lectures.js';
import { MENU_PRESET_NAMES, PRESET_NAMES } from '../src/presets/index.js';

test('parses the landing and stage routes', () => {
  expect(parseRoute('#/lectures')).toEqual({ kind: 'lectures' });
  expect(parseRoute('#/lecture1/stage2')).toEqual({ kind: 'stage', lecture: 'lecture1', stage: 2 });
  expect(parseRoute('#/bias-variance/stage1')).toEqual({
    kind: 'stage',
    lecture: 'bias-variance',
    stage: 1,
  });
});

test('anything else is the free canvas', () => {
  for (const h of [
    '',
    '#',
    '#/',
    '#x=1.abc',
    '#/lecture1/stage9',
    '#/nope/stage1',
    '#/bias-variance/stage0',
  ])
    expect(parseRoute(h), h).toEqual({ kind: 'free' });
});

test('format round-trips and names the preset', () => {
  for (const l of LECTURES)
    for (const k of l.stages) {
      const r = { kind: 'stage', lecture: l.slug, stage: k } as const;
      expect(parseRoute(formatRoute(r))).toEqual(r);
      expect(routePreset(r)).toBe(stagePresetName(l.slug, k));
      expect(PRESET_NAMES).toContain(stagePresetName(l.slug, k));
    }
  expect(formatRoute({ kind: 'lectures' })).toBe('#/lectures');
});

test('previous and next stay inside the lecture', () => {
  const n = stageNeighbours({ kind: 'stage', lecture: 'lecture1', stage: 0 });
  expect(n.prev).toBeNull();
  expect(n.next).toEqual({ kind: 'stage', lecture: 'lecture1', stage: 1 });
  const e = stageNeighbours({ kind: 'stage', lecture: 'bias-variance', stage: 3 });
  expect(e.next).toBeNull();
  expect(e.prev).toEqual({ kind: 'stage', lecture: 'bias-variance', stage: 2 });
});

test('the presets menu lists no lecture stage, and every other preset', () => {
  const lec = lecturePresetNames();
  expect(MENU_PRESET_NAMES.some((n) => lec.has(n))).toBe(false);
  expect(MENU_PRESET_NAMES).toHaveLength(PRESET_NAMES.length - lec.size);
  expect(MENU_PRESET_NAMES).toContain('scalar');
});

test('routeStep loads once, then clears the hash when the stage is replaced', () => {
  const stage = { kind: 'stage', lecture: 'lecture1', stage: 2 } as const;
  expect(routeStep(stage, false, false)).toEqual({ load: true, clear: false, armed: false });
  expect(routeStep(stage, true, false)).toEqual({ load: false, clear: false, armed: true });
  expect(routeStep(stage, false, true)).toEqual({ load: false, clear: true, armed: false });
  expect(routeStep({ kind: 'lectures' }, false, true).armed).toBe(false);
  expect(routeStep({ kind: 'free' }, false, true).clear).toBe(false);
});
