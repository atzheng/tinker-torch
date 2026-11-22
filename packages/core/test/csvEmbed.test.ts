import { describe, expect, test } from 'vitest';
import {
  CSV_LIMITS,
  SETS,
  buildCsv,
  embedCsv,
  parseCsv,
  parseFileText,
  parseGraph,
  registerCsv,
  registerEmbeddedCsv,
  sample,
  serializeFile,
  type Graph,
} from '../src/index.js';

function graph(set: string): Graph {
  const r = parseGraph({
    version: 1,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1', 'x2'] },
      { id: 'lin', type: 'linear', out: 1, bias: true },
      { id: 'yhat', type: 'output', link: 'identity' },
      { id: 'y', type: 'target' },
      { id: 'loss', type: 'loss', fn: 'squared_error' },
    ],
    edges: [
      ['x', 'lin'],
      ['lin', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ],
    data: { set, n: 40, noise: 0, val: 0.2, seed: 3 },
    train: { optimizer: 'sgd', lr: 0.01, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const TEXT = [
  'a,b,color,t',
  ...Array.from({ length: 24 }, (_, i) => `${i * 0.5},${i % 5},${['r', 'g', 'b'][i % 3]},${i * 2}`),
].join('\n');

function openSample(): void {
  const p = parseCsv(TEXT);
  if (!p.ok) throw new Error(p.error);
  const b = buildCsv(p.csv, { target: 3, inputs: [0, 1, 2], kind: 'regression' });
  if (!b.ok) throw new Error(b.error);
  registerCsv(b);
}

describe('CSV rows inside a file', () => {
  test('save, open in a fresh session: same rows, columns and kind', () => {
    openSample();
    const before = sample(graph('csv').data);
    const text = serializeFile(graph('csv'), undefined, embedCsv('data.csv'));
    expect(serializeFile(graph('csv'), undefined, embedCsv('data.csv'))).toBe(text);

    // a fresh session: only the placeholder spec, no rows of its own
    SETS.csv = { ...SETS.csv, featureNames: ['other'], d: 1 };
    const r = parseFileText(text);
    expect(r.ok).toBe(true);
    if (!r.ok || !r.file.csv) throw new Error('no csv');
    expect(r.file.csv.name).toBe('data.csv');
    registerEmbeddedCsv(r.file.csv);
    const after = sample(r.file.graph.data);
    expect(after.info).toEqual(before.info);
    expect(after.xTrain).toEqual(before.xTrain);
    expect(after.yTrain).toEqual(before.yTrain);
    expect(after.info.categories).toEqual({ x3: ['b', 'g', 'r'] });
    expect(after.info.targetName).toBe('t');
  });

  test('a preset carries a CSV the same way', () => {
    openSample();
    const doc = JSON.parse(serializeFile(graph('csv'), undefined, embedCsv())) as Record<
      string,
      unknown
    >;
    doc['title'] = 'Exercise';
    const r = parseFileText(JSON.stringify(doc));
    expect(r.ok && 'title' in r.file && r.file.csv?.features).toEqual(['a', 'b', 'color']);
  });

  test('an old file with the csv set and no rows still parses; rows are not written for other sets', () => {
    const old = JSON.parse(serializeFile(graph('csv'))) as Record<string, unknown>;
    expect('csv' in old).toBe(false);
    expect(parseFileText(JSON.stringify(old)).ok).toBe(true);
    openSample();
    expect('csv' in JSON.parse(serializeFile(graph('sine'), undefined, embedCsv()))).toBe(false);
  });

  test('bad embedded data gives one plain sentence', () => {
    openSample();
    const good = JSON.parse(serializeFile(graph('csv'), undefined, embedCsv())) as {
      csv: { x: number[][]; y: number[]; features: string[] };
    };
    const err = (mut: (c: typeof good.csv) => void): string => {
      const d = structuredClone(good);
      mut(d.csv);
      const r = parseFileText(JSON.stringify(d));
      return r.ok ? '' : r.error;
    };
    expect(err((c) => (c.x[2] = [1]))).toMatch(/Row 3 .* does not have 3 values/);
    expect(err((c) => (c.x[1] = [1, null as unknown as number, 2]))).toMatch(/Row 2 .* finite/);
    expect(err((c) => (c.x = c.x.slice(0, 5)))).toMatch(/at least 10/);
    expect(err((c) => c.y.pop())).toMatch(/rows but/);
    expect(
      err((c) => {
        c.x = Array.from({ length: CSV_LIMITS.rows + 1 }, () => [0, 0, 0]);
        c.y = c.x.map(() => 0);
      }),
    ).toMatch(/at most 5000 rows/);
    expect(err((c) => (c.features = Array.from({ length: 51 }, (_, i) => `c${i}`)))).toMatch(
      /at most 50 columns/,
    );
    expect(err((c) => (c.x[0] = [1, 2, 7]))).toMatch(/category number/);
    const sine = JSON.parse(serializeFile(graph('sine'))) as Record<string, unknown>;
    sine['csv'] = good.csv;
    const r = parseFileText(JSON.stringify(sine));
    expect(!r.ok && r.error).toMatch(/does not use the csv data set/);
  });
});
