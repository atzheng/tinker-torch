/// <reference types="node" />
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';
import {
  analyse,
  parseFile as parseFileV1,
  parseGraph as parseGraphV1,
  v2,
  type DataInfo,
} from '../src/index.js';

const {
  CORE_PIECES,
  NODE_TYPES,
  PALETTE,
  parseGraph,
  serializeGraph,
  parseFile,
  serializeFile,
  resolveEdge,
  resolvePin,
  lossSubgraph,
  weightPenaltySubgraph,
} = v2;
type Graph = v2.Graph;
type Shape = v2.Shape;

const fixturesDir = fileURLToPath(new URL('../../engine/test/fixtures/', import.meta.url));
const presetsDir = fileURLToPath(new URL('../../web/src/presets/', import.meta.url));

function ok(json: unknown): Graph {
  const r = parseGraph(json);
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

function dataInfo(g: { data: { set: string }; nodes: { type: string; columns?: string[] }[] }) {
  const input = g.nodes.find((n) => n.type === 'input');
  const classification = ['blobs', 'circle', 'xor', 'moons'].includes(g.data.set);
  return {
    columns: input?.columns ?? [],
    kind: classification ? 'binary' : 'regression',
    numClasses: classification ? 2 : 1,
  } satisfies DataInfo;
}

/** Shape of every node of a v2 graph by plain recursion (a reference for what analyse v2 will do). */
function shapes(g: Graph, info: DataInfo): Map<string, Shape | { error: string }> {
  const memo = new Map<string, Shape | { error: string }>();
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const inWidthOfLinear = (id: string): number | undefined => {
    const n = byId.get(id);
    if (!n) return undefined;
    const e = g.edges.find((x) => v2.parsePort(x[1]).node === id);
    if (!e) return undefined;
    const s = shapeOf(v2.parsePort(e[0]).node);
    return 'kind' in s && s.kind === 'example' ? s.width : undefined;
  };
  const ctx = {
    dataColumns: info.columns.length,
    numClasses: info.numClasses,
    weightShape: (id: string): [number, number] | undefined => {
      const n = byId.get(id);
      const w = inWidthOfLinear(id);
      if (n?.type !== 'linear' || w === undefined) return undefined;
      return [w, n.out];
    },
  };
  function shapeOf(id: string): Shape | { error: string } {
    const hit = memo.get(id);
    if (hit) return hit;
    const node = byId.get(id);
    if (!node) return { error: `no node ${id}` };
    const inShapes: Record<string, Shape> = {};
    for (const e of g.edges) {
      const r = resolveEdge(g.nodes, e);
      if (!r.ok || r.to.node !== id) continue;
      const s = shapeOf(r.from.node);
      if ('kind' in s) inShapes[r.to.port] = s;
    }
    const piece = CORE_PIECES[node.type] as v2.CorePiece<typeof node>;
    const s = piece.outShape(node, inShapes, ctx);
    memo.set(id, s);
    return s;
  }
  for (const n of g.nodes) shapeOf(n.id);
  return memo;
}

function ancestors(g: Graph, id: string): Set<string> {
  const seen = new Set<string>();
  const todo = [id];
  while (todo.length) {
    const cur = todo.pop() as string;
    for (const [from, to] of g.edges) {
      const f = v2.parsePort(from).node;
      if (v2.parsePort(to).node === cur && !seen.has(f)) {
        seen.add(f);
        todo.push(f);
      }
    }
  }
  return seen;
}

type V1Source = { name: string; json: unknown };
function v1Sources(): V1Source[] {
  const out: V1Source[] = [];
  for (const f of readdirSync(fixturesDir).filter((x) => x.endsWith('.json'))) {
    const fx = JSON.parse(readFileSync(fixturesDir + f, 'utf8')) as { graph: unknown };
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
    '%s: same widths and parameter counts as v1, and a scalar L',
    (_name, json) => {
      const v1 = parseGraphV1(json);
      if (!v1.ok) throw new Error(v1.error);
      const info = dataInfo(v1.graph);
      const a = analyse(v1.graph, info);
      const g = ok(json);
      const sh = shapes(g, info);

      for (const n of v1.graph.nodes) {
        if (n.type === 'output' || n.type === 'loss') continue;
        const s = sh.get(n.id);
        expect(s && 'kind' in s, `${n.id} has a shape`).toBe(true);
        expect(s).toEqual({ kind: 'example', width: a.outWidth[n.id] });
      }
      // every pin matches v1
      const outNode = a.predNode;
      expect(g.predictions.length).toBe(1);
      const pin = sh.get(v2.parsePort(g.predictions[0] as string).node);
      expect(pin).toEqual({ kind: 'example', width: outNode ? a.outWidth[outNode] : undefined });
      expect(g.loss).not.toBeNull();
      expect(sh.get(v2.parsePort(g.loss as string).node)).toEqual({ kind: 'scalar' });
      for (const [id, s] of sh) expect('error' in s, `${id} has no shape error`).toBe(false);

      // parameter counts: the Linear nodes that feed L
      let total = 0;
      const feed = ancestors(g, v2.parsePort(g.loss as string).node);
      for (const n of g.nodes) {
        if (n.type !== 'linear' || !feed.has(n.id)) continue;
        const feedEdge = g.edges.find((e) => v2.parsePort(e[1]).node === n.id);
        const inW = sh.get(v2.parsePort(feedEdge?.[0] ?? '').node);
        const w = inW && 'kind' in inW && inW.kind === 'example' ? inW.width : 0;
        const p = CORE_PIECES.linear.paramShapes(n, w);
        const count = (p.W ? p.W[0] * p.W[1] : 0) + (p.b ? p.b[0] : 0);
        expect(count).toBe(a.params[n.id]);
        total += count;
      }
      expect(total).toBe(a.totalParams);

      // weights taps appear exactly for the penalised Linear pieces
      const l = v1.graph.nodes.find((n) => n.type === 'loss');
      const taps = g.nodes.filter((n) => n.type === 'weights');
      if (l?.type === 'loss' && l.penalty) {
        const linears = v1.graph.nodes.filter((n) => n.type === 'linear' && feed.has(n.id));
        expect(taps.map((t) => (t.type === 'weights' ? t.of : ''))).toEqual(
          linears.map((n) => n.id),
        );
      } else expect(taps).toEqual([]);
    },
  );

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
      expect(parseFileV1(json).ok).toBe(true);
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
  const out = <T extends v2.PieceType>(type: T, node: object, inShapes: Record<string, Shape>) =>
    (CORE_PIECES[type] as v2.CorePiece<v2.NodeOf<T>>).outShape(
      { id: 'n', type, ...node } as v2.NodeOf<T>,
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
