// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from 'vitest';
import { SETS, buildCsv, parseCsv, sample, type DataInfo } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { axisLabel, featureName, inputLine } from '../src/state/names.js';
import { fetchCsvText } from '../src/topbar/csvSource.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';

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
  expect(s().graph.data.standardize).toBe(true); // raw columns on different scales: on by default
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

test('a saved file carries the opened CSV and reopens without it', () => {
  const rows = Array.from({ length: 20 }, (_, i) => `${i},${(i * 7) % 5},${i % 2 ? 'a' : 'b'}`);
  const p = parseCsv(['u,v,kind', ...rows].join('\n'));
  if (!p.ok) throw new Error(p.error);
  const b = buildCsv(p.csv, { target: 2, inputs: [0, 1], kind: 'binary' });
  if (!b.ok) throw new Error(b.error);
  s().openCsv(b, 'toy.csv');
  const before = sample(s().graph.data);
  const text = saveText(false);
  expect(JSON.parse(text).csv.name).toBe('toy.csv');
  // a fresh session: another CSV is open, and the file's own rows replace it
  SETS.csv = { ...SETS.csv, featureNames: ['p', 'q'] };
  useStore.setState({ csvName: null });
  expect(loadText(text)).toEqual({ ok: true, message: null });
  expect(s().graph.data.set).toBe('csv');
  expect(s().csvName).toBe('toy.csv');
  expect(s().dataInfo.featureNames).toEqual(['u', 'v']);
  expect(s().dataInfo.classNames).toEqual(before.info.classNames);
  expect(sample(s().graph.data).xTrain).toEqual(before.xTrain);
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

test('fetchCsvText: byte cap by Content-Length and by streamed size', async () => {
  const big = vi.fn(
    async () =>
      new Response('a,b\n1,2\n', { status: 200, headers: { 'content-length': '5000000' } }),
  );
  const r1 = await fetchCsvText('https://example.org/big.csv', big as unknown as typeof fetch);
  expect(!r1.ok && r1.error).toMatch(/larger than 2 MB/);
  // no Content-Length: the stream is cut at the cap
  let sent = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(c) {
      sent += 500_000;
      c.enqueue(new Uint8Array(500_000).fill(97));
      if (sent > 20_000_000) c.close();
    },
  });
  const endless = vi.fn(async () => new Response(stream, { status: 200 }));
  const r2 = await fetchCsvText('https://example.org/s.csv', endless as unknown as typeof fetch);
  expect(!r2.ok && r2.error).toMatch(/larger than 2 MB/);
  expect(sent).toBeLessThan(5_000_000);
  // multi-byte text is counted in bytes, not characters
  const text = 'é'.repeat(1_200_000);
  const wide = vi.fn(async () => new Response(text, { status: 200 }));
  const r3 = await fetchCsvText('https://example.org/w.csv', wide as unknown as typeof fetch);
  expect(r3.ok).toBe(false);
});
