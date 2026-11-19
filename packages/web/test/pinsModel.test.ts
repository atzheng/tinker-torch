import { expect, test } from 'vitest';
import { swapPiece } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import type { HistoryPoint } from '../src/state/trainSlice.js';
import {
  PIN_LIMIT,
  PIN_POINTS,
  decimateHistory,
  freeSlot,
  makePin,
  mergeBounds,
  pinLabel,
  type Pin,
} from '../src/results/pinsModel.js';

const hist = (n: number, valEvery = 1): HistoryPoint[] =>
  Array.from({ length: n }, (_, i) => ({
    step: i + 1,
    train: 1 / (i + 1),
    val: (i + 1) % valEvery === 0 ? 2 / (i + 1) : null,
  }));

test('a short history is copied as it is, without the batch series', () => {
  const h: HistoryPoint[] = [{ step: 1, train: 1, val: null, trainFull: 5 }];
  expect(decimateHistory(h)).toEqual([{ step: 1, train: 1, val: null }]);
});

test('a long history is cut to at most PIN_POINTS, keeping the first and last point', () => {
  const h = hist(20000, 3);
  const d = decimateHistory(h);
  expect(d.length).toBeLessThanOrEqual(PIN_POINTS);
  expect(d[0]?.step).toBe(1);
  expect(d[d.length - 1]?.step).toBe(20000);
  const steps = d.map((p) => p.step);
  expect(steps).toEqual([...steps].sort((a, b) => a - b));
  expect(new Set(steps).size).toBe(steps.length);
});

test('decimation keeps a validation value in every run when validation is sparse', () => {
  const d = decimateHistory(hist(5000, 3));
  expect(d.slice(1).every((p) => p.val !== null)).toBe(true);
});

test('the label names activation, hidden width, optimizer and learning rate', () => {
  const g = initialGraph();
  expect(pinLabel(g)).toBe('relu · 4 · sgd $\\eta=0.03$');
  const r = swapPiece(g, 'act1', { type: 'activation', fn: 'tanh' });
  if (!r.ok) throw new Error(r.error);
  expect(pinLabel(r.graph)).toBe('tanh · 4 · sgd $\\eta=0.03$');
});

test('makePin needs a run and a free slot; slots are reused after a removal', () => {
  const g = initialGraph();
  const host = {};
  expect(makePin(g, host, [], [])).toBeNull();
  const pins: Pin[] = [];
  for (let i = 0; i < PIN_LIMIT; i++) {
    const p = makePin(g, host, hist(10), pins);
    expect(p?.slot).toBe(i);
    if (p) pins.push(p);
  }
  expect(makePin(g, host, hist(10), pins)).toBeNull();
  expect(freeSlot(pins)).toBeNull();
  const rest = pins.filter((p) => p.slot !== 1);
  expect(makePin(g, host, hist(10), rest)?.slot).toBe(1);
});

test('final loss is the last validation loss', () => {
  const p = makePin(initialGraph(), {}, hist(10, 4), []);
  expect(p?.finalLoss).toBeCloseTo(2 / 8);
});

test('mergeBounds widens the range', () => {
  const a = { stepMax: 10, lo: 0, hi: 1 };
  const b = { stepMax: 50, lo: 0, hi: 0.5 };
  expect(mergeBounds(a, b, false)).toEqual({ stepMax: 50, lo: 0, hi: 1 });
  expect(mergeBounds({ ...a, lo: 0.1 }, { ...b, lo: 0.01 }, true)).toEqual({
    stepMax: 50,
    lo: 0.01,
    hi: 1,
  });
});
