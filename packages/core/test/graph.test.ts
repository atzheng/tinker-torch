/// <reference types="node" />
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';
import {
  CORE_PIECES,
  type CorePiece,
  type Graph,
  lossSubgraph,
  NODE_TYPES,
  type NodeOf,
  PALETTE,
  parseFile,
  parseGraph,
  type PieceType,
  resolveEdge,
  resolvePin,
  serializeFile,
  serializeGraph,
  type Shape,
  weightPenaltySubgraph,
} from '../src/index.js';

const fixturesDir = fileURLToPath(new URL('../../engine/test/fixtures/', import.meta.url));
const presetsDir = fileURLToPath(new URL('../../web/src/presets/', import.meta.url));

function ok(json: unknown): Graph {
  const r = parseGraph(json);
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

type V1Source = { name: string; json: unknown };
function v1Sources(): V1Source[] {
  const out: V1Source[] = [];
  for (const f of readdirSync(fixturesDir).filter((x) => x.endsWith('.json'))) {
    const fx = JSON.parse(readFileSync(fixturesDir + f, 'utf8')) as { graph: { version?: number } };
    if (fx.graph.version === 2) continue; // prim_* fixtures are already version 2
    out.push({ name: `fixture ${f}`, json: fx.graph });
  }
  for (const f of readdirSync(presetsDir).filter((x) => x.endsWith('.json'))) {
    const p = JSON.parse(readFileSync(presetsDir + f, 'utf8')) as { graph: unknown };
    out.push({ name: `preset ${f}`, json: p.graph });
  }
  return out;
}

const example = {
  version: 1,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    { id: 'lin1', type: 'linear', out: 4, bias: true },
    { id: 'act1', type: 'activation', fn: 'relu' },
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
  train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
};
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;

describe('v1 migration', () => {
  test('identity output is spliced out; penalty becomes graph structure', () => {
    const g = ok(example);
    expect(g.version).toBe(2);
    expect(g.predictions).toEqual(['lin2']);
    expect(g.loss).toBe('loss');
    expect(g.nodes.map((n) => `${n.id}:${n.type}`)).toEqual([
      'x:input',
      'lin1:linear',
      'act1:activation',
      'lin2:linear',
      'y:target',
      'loss_diff:subtract',
      'loss_err:square',
      'loss_mean:mean',
      'loss_w_lin1:weights',
      'loss_n_lin1:norm',
      'loss_w_lin2:weights',
      'loss_n_lin2:norm',
      'loss_sum1:add',
      'loss_pen:scale',
      'loss:add',
    ]);
    const find = (id: string) => g.nodes.find((n) => n.id === id);
    expect(find('loss_pen')).toMatchObject({ c: 0.1 });
    expect(find('loss_n_lin1')).toMatchObject({ kind: 'l2' });
    expect(find('loss_w_lin2')).toMatchObject({ of: 'lin2' });
    expect(g.edges).toContainEqual(['lin2', 'loss_diff.a']);
    expect(g.edges).toContainEqual(['y', 'loss_diff.b']);
    expect(g.edges).toContainEqual(['loss_mean', 'loss.a']);
    expect(g.edges).toContainEqual(['loss_pen', 'loss.b']);
  });

  test('sigmoid and softmax outputs become link pieces with the same id; cross-entropy and log loss have no subtract', () => {
    const j = clone(example);
    j.nodes[4] = { id: 'yhat', type: 'output', link: 'sigmoid' } as never;
    j.nodes[6] = { id: 'loss', type: 'loss', fn: 'log_loss' } as never;
    const g = ok(j);
    expect(g.nodes).toContainEqual({ id: 'yhat', type: 'link', fn: 'sigmoid' });
    expect(g.predictions).toEqual(['yhat']);
    expect(g.nodes.some((n) => n.type === 'subtract')).toBe(false);
    expect(g.edges).toContainEqual(['yhat', 'loss_err.pred']);
    expect(g.nodes.find((n) => n.id === 'loss')).toEqual({ id: 'loss', type: 'mean' });
  });

  test('huber keeps its delta', () => {
    const j = clone(example);
    j.nodes[6] = { id: 'loss', type: 'loss', fn: 'huber', delta: 0.5 } as never;
    const g = ok(j);
    expect(g.nodes.find((n) => n.id === 'loss_err')).toEqual({
      id: 'loss_err',
      type: 'huber',
      delta: 0.5,
    });
  });

  test('a penalty with no Linear feeding the loss is dropped; generated ids never collide', () => {
    const j = clone(example);
    j.nodes.push({ id: 'loss_diff', type: 'target' } as never);
    const g = ok(j);
    const ids = g.nodes.map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain('loss_diff_2');
  });

  test('graphs with no loss, or an unconnected output, still migrate', () => {
    const j = clone(example);
    j.nodes = j.nodes.filter((n) => n.type !== 'loss');
    j.edges = j.edges.filter((e) => !(e[1] ?? '').startsWith('loss'));
    const g = ok(j);
    expect(g.loss).toBeNull();
    expect(g.predictions).toEqual(['lin2']);
    const k = clone(example);
    k.edges = k.edges.filter((e) => e[1] !== 'yhat');
    expect(ok(k).predictions).toEqual([]);
  });

  test.each(v1Sources().map((s) => [s.name, s.json] as const))(
    '%s: re-serialises stably as version 2',
    (_name, json) => {
      const g = ok(json);
      const text = serializeGraph(g);
      expect(JSON.parse(text)).toMatchObject({ version: 2 });
      const again = ok(JSON.parse(text));
      expect(again).toEqual(g);
      expect(serializeGraph(again)).toBe(text);
    },
  );
});

describe('saved files', () => {
  test('a version 1 file opens as version 2 and re-saves stably', () => {
    const v1 = JSON.stringify({
      kind: 'tinker',
      version: 1,
      graph: example,
      params: { lin1: { W: [1], b: [0] } },
    });
    const r = parseFile(JSON.parse(v1));
    if (!r.ok) throw new Error(r.error);
    expect(r.file.version).toBe(2);
    expect(r.file.params).toEqual({ lin1: { W: [1], b: [0] } });
    const text = serializeFile(r.file.graph, r.file.params);
    const r2 = parseFile(JSON.parse(text));
    if (!r2.ok) throw new Error(r2.error);
    expect(serializeFile(r2.file.graph, r2.file.params)).toBe(text);
    expect(JSON.parse(text)).toMatchObject({ kind: 'tinker', version: 2 });
  });

  test('every bundled preset migrates and keeps its title, prompt and locks', () => {
    for (const f of readdirSync(presetsDir).filter((x) => x.endsWith('.json'))) {
      const json = JSON.parse(readFileSync(presetsDir + f, 'utf8')) as Record<string, unknown>;
      const r = parseFile(json);
      if (!r.ok) throw new Error(`${f}: ${r.error}`);
      expect(r.file.version).toBe(2);
      expect('title' in r.file && r.file.title).toBe(json['title']);
      // the legacy reader still gives the same preset, for consumers that have not moved yet
      expect(parseFile(json).ok).toBe(true);
    }
  });

  test('unknown file versions are refused with a sentence', () => {
    const r = parseFile({ kind: 'tinker', version: 3, graph: example });
    expect(r).toEqual({ ok: false, error: expect.stringContaining('version 3') });
  });
});

describe('graph v2 schema', () => {
  const v2json = () => {
    const g = ok(example);
    return JSON.parse(serializeGraph(g)) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any -- test input is deliberately malformed
  };

  test('role fields default to none', () => {
    const j = v2json();
    delete j['predictions'];
    delete j['loss'];
    const g = ok(j);
    expect(g.predictions).toEqual([]);
    expect(g.loss).toBeNull();
  });

  test('pins must point at nodes and outputs; no duplicate ŷ', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- test input is deliberately malformed
    const bad = (f: (j: Record<string, any>) => void) => {
      const j = v2json();
      f(j);
      const r = parseGraph(j);
      if (r.ok) throw new Error('expected a failure');
      return r.error;
    };
    expect(bad((j) => (j['predictions'] = ['nope']))).toMatch(/not a node/);
    expect(bad((j) => (j['loss'] = 'loss.in'))).toMatch(/output "out"/);
    expect(bad((j) => (j['predictions'] = ['lin2', 'lin2.out']))).toMatch(/twice/);
    expect(
      ok({ ...v2json(), predictions: ['lin2.out', 'act1'], loss: 'loss.out' }).predictions,
    ).toHaveLength(2);
  });

  test('weights must name a Linear; unknown and v1-only types get a sentence', () => {
    const j = v2json();
    j['nodes'].push({ id: 'w', type: 'weights', of: 'act1' });
    expect(parseGraph(j)).toEqual({
      ok: false,
      error: expect.stringContaining('must name a Linear'),
    });
    const k = v2json();
    k['nodes'].push({ id: 'o', type: 'output', link: 'identity' });
    expect(parseGraph(k)).toEqual({
      ok: false,
      error: expect.stringContaining('only exists in version 1'),
    });
  });

  test('every piece appears in the schema, the registry and the palette', () => {
    expect(Object.keys(CORE_PIECES).sort()).toEqual([...NODE_TYPES].sort());
    for (const t of NODE_TYPES) expect(CORE_PIECES[t].defaults()).toMatchObject({ type: t });
    const used = new Set(PALETTE.flatMap((p) => (p.kind === 'piece' ? [p.make().type] : [])));
    for (const t of NODE_TYPES) expect(used.has(t), `palette has ${t}`).toBe(true);
    const keys = PALETTE.map((p) => p.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  test('palette macros insert primitives that parse into a valid graph', () => {
    for (const p of PALETTE) {
      if (p.kind !== 'macro') continue;
      const used = new Set(['x']);
      const name = (s: string) => {
        let id = `m_${s}`;
        for (let k = 2; used.has(id); k++) id = `m_${s}_${k}`;
        used.add(id);
        return id;
      };
      const sub = p.make(name, { linears: ['lin1'] });
      const g = ok({
        ...JSON.parse(serializeGraph(ok(example))),
        nodes: [
          { id: 'lin1', type: 'linear', out: 1, bias: true },
          { id: 'x', type: 'input', columns: ['x1'] },
          ...sub.nodes,
        ],
        edges: sub.edges,
        predictions: [],
        loss: sub.out,
      });
      expect(g.nodes.length).toBe(sub.nodes.length + 2);
    }
  });

  test('ports: every piece has out; ports of the new pieces', () => {
    expect(CORE_PIECES.subtract.inputPorts).toEqual(['a', 'b']);
    expect(CORE_PIECES.log_loss.inputPorts).toEqual(['pred', 'target']);
    expect(CORE_PIECES.weights.inputPorts).toEqual([]);
    expect(CORE_PIECES.norm.inputPorts).toEqual(['in']);
    const g = ok(example);
    expect(resolvePin(g.nodes, 'lin2.out')).toEqual({
      ok: true,
      at: { node: 'lin2', port: 'out' },
    });
    expect(resolveEdge(g.nodes, ['x', 'lin1'])).toMatchObject({ ok: true });
    expect(resolveEdge(g.nodes, ['lin1', 'loss.a'])).toMatchObject({ ok: true });
    expect(resolveEdge(g.nodes, ['lin1', 'loss'])).toMatchObject({ ok: false });
  });
});

describe('shape rules', () => {
  const ctx = { dataColumns: 2, numClasses: 3, weightShape: () => [2, 3] as [number, number] };
  const ex = (width: number): Shape => ({ kind: 'example', width });
  const sc: Shape = { kind: 'scalar' };
  const out = <T extends PieceType>(type: T, node: object, inShapes: Record<string, Shape>) =>
    (CORE_PIECES[type] as CorePiece<NodeOf<T>>).outShape(
      { id: 'n', type, ...node } as NodeOf<T>,
      inShapes,
      ctx,
    );

  test('per-example versus single number', () => {
    expect(out('mean', {}, { in: ex(1) })).toEqual(sc);
    expect(out('mean', {}, { in: sc })).toHaveProperty('error');
    expect(out('mean', {}, {})).toHaveProperty('error');
    expect(out('subtract', {}, { a: ex(1), b: ex(1) })).toEqual(ex(1));
    expect(out('subtract', {}, { a: ex(2), b: ex(1) })).toHaveProperty('error');
    expect(out('add', {}, { a: sc, b: sc })).toEqual(sc);
    expect(out('add', {}, { a: sc, b: ex(1) })).toHaveProperty('error');
    expect(out('square', {}, { in: ex(4) })).toEqual(ex(4));
    expect(out('scale', { c: 2 }, { in: sc })).toEqual(sc);
    expect(out('linear', { out: 2, bias: true }, { in: sc })).toHaveProperty('error');
  });

  test('weights and norms', () => {
    expect(out('weights', { of: 'lin1' }, {})).toEqual({ kind: 'param', rows: 2, cols: 3 });
    expect(out('norm', { kind: 'l1' }, { in: { kind: 'param', rows: 2, cols: 3 } })).toEqual(sc);
    expect(out('norm', { kind: 'l1' }, { in: ex(1) })).toHaveProperty('error');
    expect(
      CORE_PIECES.weights.outShape(
        { id: 'w', type: 'weights', of: 'q' },
        {},
        { ...ctx, weightShape: () => undefined },
      ),
    ).toHaveProperty('error');
  });

  test('losses give one number per example; log loss needs width 1', () => {
    expect(out('log_loss', {}, { pred: ex(1), target: ex(1) })).toEqual(ex(1));
    expect(out('log_loss', {}, { pred: ex(2), target: ex(1) })).toHaveProperty('error');
    expect(out('cross_entropy', {}, { pred: ex(3), target: ex(1) })).toEqual(ex(1));
    expect(out('cross_entropy', {}, { pred: ex(3) })).toHaveProperty('error');
  });

  test('links keep the width; the old width rules still hold', () => {
    expect(out('link', { fn: 'softmax' }, { in: ex(3) })).toEqual(ex(3));
    expect(
      out('feature', { degree: 3, sin: true, cos: true, products: true }, { in: ex(2) }),
    ).toEqual(ex(11));
    expect(out('input', { columns: ['x1', 'x2'] }, {})).toEqual(ex(2));
    expect(out('target', {}, {})).toEqual(ex(1));
  });

  test('macro builders', () => {
    const name = (s: string) => `l_${s}`;
    const sub = lossSubgraph('absolute_error', { name });
    expect(sub.nodes.map((n) => n.type)).toEqual(['subtract', 'abs', 'mean']);
    expect(sub.inputs).toEqual({ pred: 'l_diff.a', target: 'l_diff.b' });
    expect(sub.out).toBe('l_loss');
    expect(weightPenaltySubgraph({ kind: 'l1', lambda: 1, linears: [], name }).out).toBeNull();
  });
});

describe('locked ids through migration', () => {
  test('a locked Loss covers its whole group; a spliced identity Output covers nothing', () => {
    const r = parseFile({
      kind: 'tinker',
      version: 1,
      title: 't',
      graph: example,
      locked: ['lin1', 'yhat', 'loss'],
    });
    if (!r.ok) throw new Error(r.error);
    const locked = 'locked' in r.file ? r.file.locked : undefined;
    expect(locked).toContain('lin1');
    expect(locked).not.toContain('yhat');
    for (const id of ['loss', 'loss_diff', 'loss_err', 'loss_mean', 'loss_pen', 'loss_w_lin1']) {
      expect(locked).toContain(id);
    }
    const g = r.file.graph;
    for (const id of locked ?? []) expect(g.nodes.some((n) => n.id === id)).toBe(true);
  });
});
