// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { parseFile } from '@tinker/core';
import interceptMean from '../src/presets/intercept-mean.json';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';
import { loadParsed } from '../src/topbar/FileMenu.js';
import { mlpGraph } from './mlpGraph.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(mlpGraph());
  s().reset();
});

function load(extra: Record<string, unknown>) {
  const r = parseFile({ ...interceptMean, params: { bias: 3 }, ...extra });
  if (!r.ok) throw new Error(r.error);
  loadParsed(r.file);
}
const SHOW = { show: { ui: ['panel-step-log'] } };

test('each Step adds a row: b_next = b - lr * slope, slope and loss from before the step', () => {
  load(SHOW);
  const lr = 0.2;
  const ys = Array.from(s().dataset.yTrain);
  const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
  let b = 3;
  for (let i = 1; i <= 3; i++) {
    s().stepOnce();
    const row = s().stepLog[i - 1];
    expect(row?.step).toBe(i);
    const slope = 2 * (b - mean); // d/db of mean((b - y)^2)
    const loss = ys.reduce((a, y) => a + (b - y) ** 2, 0) / ys.length;
    expect(row?.slopes?.[0]).toBeCloseTo(slope, 3);
    expect(row?.loss).toBeCloseTo(loss, 3);
    b = b - lr * slope;
    expect(row?.values[0]).toBeCloseTo(b, 3);
    expect(row?.prediction).toBeCloseTo(b, 3);
  }
  expect(s().stepLog).toHaveLength(3);
});

test('Reset clears; the log keeps the last 50 rows', () => {
  load(SHOW);
  for (let i = 0; i < 55; i++) s().stepOnce();
  expect(s().stepLog).toHaveLength(50);
  expect(s().stepLog[0]?.step).toBe(6);
  s().reset();
  expect(s().stepLog).toHaveLength(0);
});

test('without panel-step-log in show.ui nothing is logged', () => {
  load({});
  s().stepOnce();
  expect(s().stepLog).toHaveLength(0);
});

test('a mini-batch run logs no slope', () => {
  load({
    ...SHOW,
    graph: { ...interceptMean.graph, train: { ...interceptMean.graph.train, batch: 8 } },
  });
  s().stepOnce();
  expect(s().stepLog[0]?.slopes).toBeNull();
});
