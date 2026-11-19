import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { analyse, parseFile, parseFileText, sample, serializeFile } from '../src/index.js';

const NAMES = ['linear', 'logistic', 'sine', 'xor', 'skip', 'scalar', 'mlp'];
const raw = (name: string): Record<string, unknown> =>
  JSON.parse(
    readFileSync(new URL(`../../web/src/presets/${name}.json`, import.meta.url), 'utf8'),
  ) as Record<string, unknown>;

describe('bundled presets are native graph v2 files', () => {
  test.each(NAMES)('%s: raw JSON is version 2, trainable, ids are tidy', (name) => {
    const json = raw(name);
    expect(json['version']).toBe(2);
    expect((json['graph'] as { version: number }).version).toBe(2);
    const r = parseFile(json);
    if (!r.ok) throw new Error(r.error);
    const g = r.file.graph;
    expect(g.version).toBe(2);
    const ids = g.nodes.map((n) => n.id);
    expect(ids.filter((i) => i.startsWith('loss_'))).toEqual([]);
    expect(g.loss).toBe('loss');
    expect(g.predictions).toHaveLength(1);
    expect(analyse(g, sample(g.data).info).trainable).toBe(true);
    // Already v2: re-serialising the parsed graph changes nothing (no migration happened).
    const again = JSON.parse(serializeFile(g)) as { graph: unknown };
    expect(again.graph).toEqual(json['graph']);
  });
});

describe('a version 1 file opens migrated and round-trips as version 2', () => {
  const v1 = {
    kind: 'tinker',
    version: 1,
    title: 'Old',
    prompt: 'P',
    locked: ['lin1', 'yhat', 'loss'],
    graph: {
      version: 1,
      nodes: [
        { id: 'x', type: 'input', columns: ['x1'] },
        { id: 'lin1', type: 'linear', out: 1, bias: true },
        { id: 'yhat', type: 'output', link: 'identity' },
        { id: 'y', type: 'target' },
        { id: 'loss', type: 'loss', fn: 'squared_error' },
      ],
      edges: [
        ['x', 'lin1'],
        ['lin1', 'yhat'],
        ['yhat', 'loss.pred'],
        ['y', 'loss.target'],
      ],
      data: { set: 'line', n: 40, noise: 0.2, val: 0.3, seed: 7 },
      train: { optimizer: 'sgd', lr: 0.05, batch: 'full', seed: 11 },
    },
  };

  test('pins, locked ids and Linear ids survive', () => {
    const r = parseFile(v1);
    if (!r.ok) throw new Error(r.error);
    const f = r.file as { locked?: string[] };
    expect(r.file.version).toBe(2);
    expect(r.file.graph.predictions).toEqual(['lin1']);
    expect(r.file.graph.loss).toBe('loss');
    // lin1 stays; the identity Output covers nothing; the Loss covers its whole group.
    expect(f.locked).toEqual(['lin1', 'loss_diff', 'loss_err', 'loss']);
    const text = serializeFile(r.file.graph);
    const back = parseFileText(text);
    if (!back.ok) throw new Error(back.error);
    expect(back.file.graph).toEqual(r.file.graph);
    expect(serializeFile(back.file.graph)).toBe(text);
    expect((JSON.parse(text) as { version: number }).version).toBe(2);
  });
});
