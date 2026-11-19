// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from 'vitest';
import { buildCsv, parseCsv, sample, type DataInfo } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { axisLabel, featureName, inputLine } from '../src/state/names.js';
import { fetchCsvText } from '../src/topbar/csvSource.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(initialGraph());
});

const info: DataInfo = {
  columns: ['x1', 'x2', 'x3'],
  featureNames: ['sepal length', 'sepal width', 'petal length'],
  kind: 'regression',
  numClasses: 1,
};

test('feature names: display, axis titles, one-line input subtitle', () => {
  expect(featureName(info, 'x2')).toBe('sepal width');
  expect(featureName({ ...info, featureNames: undefined }, 'x2')).toBe('x2');
  expect(axisLabel('x1')).toBe('$x_1$');
  expect(axisLabel('cost $')).toBe('cost ');
  expect(inputLine(info, ['x1', 'x2', 'x3'], 400)).toBe('sepal length, sepal width, petal length');
  const narrow = inputLine(info, ['x1', 'x2', 'x3'], 120);
  expect(narrow.length).toBeLessThan(20);
  expect(narrow).toMatch(/sepal length/);
});

test('choosing a bundled set loads its rows then switches; columns reset to x1..xd', async () => {
  await s().chooseSet('wine');
  expect(s().graph.data.set).toBe('wine');
  expect(s().dataInfo.columns).toHaveLength(13);
  expect(s().dataInfo.featureNames?.[0]).toBe('alcohol');
  expect(s().dataInfo.targetName).toBe('cultivar');
});

test('the iris preset loads its dataset, then the graph', async () => {
  expect(loadBundled('iris')).toBe(true);
  await vi.waitFor(() => expect(s().graph.data.set).toBe('iris'));
  expect(s().dataset.nTrain).toBe(120);
  expect(s().analysis.issues).toEqual([]);
});

test('Open CSV makes the csv set; opening again re-samples', () => {
  const rows = Array.from({ length: 20 }, (_, i) => `${i},${(i * 7) % 5},${i % 2 ? 'a' : 'b'}`);
  const p = parseCsv(['u,v,kind', ...rows].join('\n'));
  if (!p.ok) throw new Error(p.error);
  const b = buildCsv(p.csv, { target: 2, inputs: [0, 1], kind: 'binary' });
  if (!b.ok) throw new Error(b.error);
  s().openCsv(b, 'toy.csv');
  expect(s().graph.data.set).toBe('csv');
  expect(s().csvName).toBe('toy.csv');
  expect(s().dataInfo.featureNames).toEqual(['u', 'v']);
  const seed = s().graph.data.seed;
  s().openCsv(b, 'toy2.csv');
  expect(s().graph.data.seed).not.toBe(seed);
  expect(sample(s().graph.data).nTrain + sample(s().graph.data).nVal).toBe(20);
});

test('fetchCsvText: fake fetch only, one-sentence errors', async () => {
  const ok = vi.fn(async () => new Response('a,b\n1,2\n', { status: 200 }));
  const r = await fetchCsvText('https://example.org/data/iris.csv', ok as unknown as typeof fetch);
  expect(r).toEqual({ ok: true, text: 'a,b\n1,2\n', name: 'iris.csv' });
  expect(ok).toHaveBeenCalledTimes(1);

  const down = vi.fn(async () => {
    throw new TypeError('Failed to fetch');
  });
  const e1 = await fetchCsvText('https://example.org/x.csv', down as unknown as typeof fetch);
  expect(!e1.ok && e1.error).toMatch(/could not be fetched/);

  const nf = vi.fn(async () => new Response('no', { status: 404 }));
  const e2 = await fetchCsvText('https://example.org/x.csv', nf as unknown as typeof fetch);
  expect(!e2.ok && e2.error).toBe('The server answered 404 for that address.');

  const never = vi.fn();
  const e3 = await fetchCsvText('ftp://example.org/x.csv', never as unknown as typeof fetch);
  expect(!e3.ok && e3.error).toMatch(/http/);
  expect(never).not.toHaveBeenCalled();
});
