import { describe, expect, test } from 'vitest';
import {
  analyse,
  initParams,
  paramsFromJson,
  paramsToJson,
  parseFile,
  parseFileText,
  parseGraph,
  sample,
  serializeFile,
  serializeGraph,
  type Graph,
} from '../src/index.js';

/** A version 1 document on purpose: opening it goes through the migration. */
function graph(): Graph {
  const r = parseGraph({
    version: 1,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'lin1', type: 'linear', out: 3, bias: true },
      { id: 'act1', type: 'activation', fn: 'tanh' },
      { id: 'lin2', type: 'linear', out: 1, bias: true },
      { id: 'yhat', type: 'output', link: 'identity' },
      { id: 'y', type: 'target' },
      { id: 'loss', type: 'loss', fn: 'squared_error', penalty: { kind: 'l2', lambda: 0.1 } },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ],
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

describe('save and load round trip', () => {
  test('graph and params come back the same, and the text is byte-stable', () => {
    const g = graph();
    const a = analyse(g, sample(g.data).info);
    const host = initParams(a, 5);
    const text = serializeFile(g, paramsToJson(host));
    const parsed = parseFileText(text);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(serializeGraph(parsed.file.graph)).toBe(serializeGraph(g));
    expect(parsed.file.params).toBeDefined();
    const back = paramsFromJson(parsed.file.params, a);
    expect(back.dropped).toEqual([]);
    for (const id of Object.keys(host)) {
      expect(Array.from(back.params[id]?.W ?? [])).toEqual(Array.from(host[id]?.W ?? []));
      expect(Array.from(back.params[id]?.b ?? [])).toEqual(Array.from(host[id]?.b ?? []));
    }
    expect(serializeFile(parsed.file.graph, parsed.file.params)).toBe(text);
  });

  test('a file without params is fine', () => {
    const r = parseFileText(serializeFile(graph()));
    expect(r.ok && r.file.params).toBeFalsy();
  });

  test('params for a resized layer are dropped, not an error', () => {
    const g = graph();
    const a = analyse(g, sample(g.data).info);
    const json = paramsToJson(initParams(a, 1));
    const bigger = parseGraph(JSON.parse(serializeGraph(g).replace('"out": 3', '"out": 4')));
    if (!bigger.ok) throw new Error(bigger.error);
    const text = serializeFile(bigger.graph, json);
    const r = parseFileText(text);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const a2 = analyse(r.file.graph, sample(r.file.graph.data).info);
    const back = paramsFromJson(r.file.params, a2);
    expect(back.dropped.sort()).toEqual(['lin1', 'lin2']);
  });
});

describe('malformed files give one sentence', () => {
  const good = JSON.parse(serializeFile(graph())) as Record<string, unknown>;
  const cases: [string, unknown][] = [
    ['null', null],
    ['an array', []],
    ['a string', 'hi'],
    ['no kind', { ...good, kind: undefined }],
    ['wrong kind', { ...good, kind: 'other' }],
    ['wrong version', { ...good, version: 3 }],
    ['no version', { ...good, version: undefined }],
    ['no graph', { ...good, graph: undefined }],
    ['invalid graph', { ...good, graph: { version: 1, nodes: 'x' } }],
    ['params not an object', { ...good, params: 5 }],
    ['preset title not text', { ...good, title: 3 }],
    ['preset locked not a list', { ...good, title: 't', locked: 'x' }],
  ];
  test.each(cases)('%s', (_name, doc) => {
    const r = parseFile(doc);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toMatch(/^[A-Z].*\.$/);
    expect(r.error).not.toContain('\n');
  });

  test('bad JSON text', () => {
    expect(parseFileText('{nope')).toEqual({ ok: false, error: 'This file is not valid JSON.' });
    expect(parseFileText('')).toEqual({ ok: false, error: 'This file is not valid JSON.' });
  });
});

test('a preset keeps its title, prompt and locked list', () => {
  const doc = { ...JSON.parse(serializeFile(graph())), title: 'T', prompt: 'P', locked: ['lin1'] };
  const r = parseFile(doc);
  expect(r.ok && 'title' in r.file && r.file.title).toBe('T');
  expect(r.ok && 'locked' in r.file && r.file.locked).toEqual(['lin1']);
});
