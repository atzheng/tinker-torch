import { describe, expect, test } from 'vitest';
import { parseGraph, parsePort, resolveEdge, serializeGraph, type Graph } from '../src/index.js';

const example = {
  version: 1,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    { id: 'lin1', type: 'linear', out: 4, bias: true },
    { id: 'act1', type: 'activation', fn: 'relu' },
    { id: 'lin2', type: 'linear', out: 1, bias: true },
    { id: 'yhat', type: 'output', link: 'identity' },
    { id: 'y', type: 'target' },
    { id: 'loss', type: 'loss', fn: 'squared_error', penalty: { kind: 'l2', lambda: 0.0 } },
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
  train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
};

function parseOk(json: unknown): Graph {
  const r = parseGraph(json);
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

// The edit functions poke at loosely typed JSON on purpose.
type Loose = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any -- test input is deliberately malformed

function mutate(fn: (g: Loose) => void): unknown {
  const g = JSON.parse(JSON.stringify(example)) as Loose;
  fn(g);
  return g;
}

describe('parseGraph', () => {
  test('parses the SPEC example and fills defaults', () => {
    const g = parseOk(example);
    expect(g.data.standardize).toBe(false);
    expect(g.nodes).toHaveLength(7);
  });

  test('round-trips through serializeGraph unchanged', () => {
    const g = parseOk(example);
    const text = serializeGraph(g);
    expect(JSON.parse(text)).toEqual({ ...example, data: { ...example.data, standardize: false } });
    expect(serializeGraph(parseOk(JSON.parse(text)))).toBe(text);
  });

  test('serializeGraph key order is stable regardless of input key order', () => {
    const shuffled = {
      train: { seed: 11, batch: 'full', lr: 0.03, optimizer: 'sgd' },
      edges: example.edges,
      nodes: [
        example.nodes[0],
        { type: 'linear', bias: true, out: 4, id: 'lin1' },
        ...example.nodes.slice(2),
      ],
      data: example.data,
      version: 1,
    };
    const text = serializeGraph(parseOk(shuffled));
    expect(text.indexOf('"version"')).toBeLessThan(text.indexOf('"nodes"'));
    expect(text).toContain(
      '"id": "lin1",\n      "type": "linear",\n      "out": 4,\n      "bias": true',
    );
  });

  test('fills huber delta and node defaults', () => {
    const g = parseOk(
      mutate((m) => {
        m['nodes'][6] = { id: 'loss', type: 'loss', fn: 'huber' };
        m['nodes'][1] = { id: 'lin1', type: 'linear', out: 4 };
      }),
    );
    expect(g.nodes[6]).toEqual({ id: 'loss', type: 'loss', fn: 'huber', delta: 1 });
    expect(g.nodes[1]).toEqual({ id: 'lin1', type: 'linear', out: 4, bias: true });
  });

  const bad: [string, (g: Loose) => void, RegExp][] = [
    ['unknown type', (g) => (g['nodes'][1].type = 'conv'), /unknown type "conv"/],
    [
      'unknown node field',
      (g) => (g['nodes'][1].color = 'red'),
      /Node "lin1" has an unknown field "color"\./,
    ],
    ['unknown top-level field', (g) => (g['extra'] = 1), /unknown field "extra"/],
    [
      'edge to a missing node',
      (g) => (g['edges'][0] = ['x', 'nope']),
      /"nope", which is not a node/,
    ],
    [
      'bad port on loss',
      (g) => (g['edges'][4] = ['yhat', 'loss.zzz']),
      /has the inputs pred, target/,
    ],
    [
      'bare loss destination',
      (g) => (g['edges'][4] = ['yhat', 'loss']),
      /does not say which input/,
    ],
    ['edge from a loss', (g) => (g['edges'][0] = ['loss', 'lin1']), /has no output/],
    ['edge into an input', (g) => (g['edges'][0] = ['lin1', 'x']), /takes no inputs/],
    ['bad source port', (g) => (g['edges'][0] = ['x.in', 'lin1']), /only has the output "out"/],
    [
      'linear out too big',
      (g) => (g['nodes'][1].out = 65),
      /field "out" must be a whole number from 1 to 64/,
    ],
    ['linear out fractional', (g) => (g['nodes'][1].out = 2.5), /field "out"/],
    [
      'negative lambda',
      (g) => (g['nodes'][6].penalty.lambda = -1),
      /penalty\.lambda" must be a number of at least 0/,
    ],
    [
      'bad activation',
      (g) => (g['nodes'][2].fn = 'swish'),
      /must be one of sigmoid, tanh, relu, gelu/,
    ],
    [
      'dropout p = 1',
      (g) => (g['nodes'][2] = { id: 'd', type: 'dropout', p: 1 }),
      /Node "d" field "p"/,
    ],
    ['delta on non-huber', (g) => (g['nodes'][6].delta = 2), /Huber/],
    [
      'n out of range',
      (g) => (g['data'].n = 5),
      /data settings field "n" must be a whole number from 10 to 500/,
    ],
    ['val out of range', (g) => (g['data'].val = 0.9), /field "val"/],
    ['unknown dataset', (g) => (g['data'].set = 'spiral'), /field "set" must be one of/],
    ['bad batch', (g) => (g['train'].batch = 'half'), /training settings field "batch"/],
    ['lr zero', (g) => (g['train'].lr = 0), /field "lr"/],
    ['wrong version', (g) => (g['version'] = 2), /must be 1/],
    [
      'duplicate ids',
      (g) => (g['nodes'][2] = { id: 'x', type: 'target' }),
      /Two nodes are both called "x"/,
    ],
    [
      'bad id',
      (g) => (g['nodes'][1].id = 'Lin 1'),
      /field "id" must start with a lowercase letter/,
    ],
  ];
  test.each(bad)('rejects %s with one sentence', (_name, edit, pattern) => {
    const r = parseGraph(mutate(edit));
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toMatch(pattern);
      expect(r.error).toMatch(/\.$/);
      expect(r.error).not.toContain('\n');
    }
  });

  test('rejects non-objects', () => {
    expect(parseGraph(null)).toEqual({ ok: false, error: 'The graph must be a JSON object.' });
    expect(parseGraph([])).toMatchObject({ ok: false });
  });
});

describe('ports', () => {
  test('parsePort', () => {
    expect(parsePort('loss.pred')).toEqual({ node: 'loss', port: 'pred' });
    expect(parsePort('lin1')).toEqual({ node: 'lin1', port: null });
  });

  test('resolveEdge fills default ports', () => {
    const g = parseOk(example);
    expect(resolveEdge(g.nodes, ['x', 'lin1'])).toEqual({
      ok: true,
      from: { node: 'x', port: 'out' },
      to: { node: 'lin1', port: 'in' },
    });
    expect(resolveEdge(g.nodes, ['yhat', 'loss.pred'])).toMatchObject({
      ok: true,
      to: { node: 'loss', port: 'pred' },
    });
  });
});
