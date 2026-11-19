import { describe, expect, test } from 'vitest';
import fixture from '../../engine/test/fixtures/linear_bias_d2.json';
import mlp from '../../engine/test/fixtures/mlp_tanh_sgd_5steps.json';
import tapOnly from '../../engine/test/fixtures/prim_weights_tap_only.json';
import {
  analyse,
  countParams,
  initParams,
  paramsFromJson,
  parseGraph,
  paramsToJson,
  reconcileParams,
  type DataInfo,
  type Edge,
  type Graph,
  type Node,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 };

function net(widths: number[], opts: { extra?: boolean; bias?: boolean } = {}): Graph {
  const nodes: Node[] = [{ id: 'in', type: 'input', columns: ['x1', 'x2'] }];
  const edges: Edge[] = [];
  let prev = 'in';
  widths.forEach((w, i) => {
    const id = `l${i + 1}`;
    nodes.push({ id, type: 'linear', out: w, bias: opts.bias ?? true });
    edges.push([prev, id]);
    prev = id;
  });
  if (opts.extra) {
    nodes.push({ id: 'zz', type: 'linear', out: 3, bias: true });
    edges.push([prev, 'zz']);
    prev = 'zz';
  }
  nodes.push(
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'err', type: 'square' },
    { id: 'loss', type: 'mean' },
  );
  edges.push([prev, 'diff.a'], ['y', 'diff.b'], ['diff', 'err'], ['err', 'loss']);
  return {
    version: 2,
    nodes,
    edges,
    predictions: [prev],
    loss: 'loss',
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
  };
}
const an = (g: Graph) => analyse(g, info);

describe('initParams', () => {
  test('shapes and bounds for several widths', () => {
    for (const widths of [[1], [4, 1], [16, 8, 1], [64, 1]]) {
      const g = net(widths);
      const a = an(g);
      const h = initParams(a, 3);
      let inW = 2;
      widths.forEach((w, i) => {
        const p = h[`l${i + 1}`];
        expect(p?.W?.length).toBe(inW * w);
        expect(p?.b?.length).toBe(w);
        const bound = 1 / Math.sqrt(inW);
        for (const v of [...(p?.W ?? []), ...(p?.b ?? [])])
          expect(Math.abs(v)).toBeLessThanOrEqual(bound);
        inW = w;
      });
      expect(Object.keys(h)).toEqual(widths.map((_, i) => `l${i + 1}`));
      expect(countParams(h)).toBe(a.totalParams);
    }
  });

  test('values spread over the range', () => {
    const g = net([64]);
    const p = initParams(an(g), 1)['l1']?.W ?? new Float32Array(0);
    expect(Math.max(...p)).toBeGreaterThan(0.5 / Math.SQRT2);
    expect(Math.min(...p)).toBeLessThan(-0.5 / Math.SQRT2);
  });

  test('no bias gives no b', () => {
    const g = net([3], { bias: false });
    expect(initParams(an(g), 1)['l1']?.b).toBeUndefined();
  });

  test('same seed same values, different seed different', () => {
    const g = net([4, 1]);
    const a = an(g);
    expect(paramsToJson(initParams(a, 5))).toEqual(paramsToJson(initParams(a, 5)));
    expect(paramsToJson(initParams(a, 5))).not.toEqual(paramsToJson(initParams(a, 6)));
  });

  test('adding a node leaves the other nodes unchanged', () => {
    const g1 = net([4, 1]);
    const g2 = net([4, 1], { extra: true });
    const h1 = initParams(an(g1), 9);
    const h2 = initParams(an(g2), 9);
    expect(h2['zz']).toBeDefined();
    expect(h2['l1']).toEqual(h1['l1']);
    expect(h2['l2']).toEqual(h1['l2']);
  });

  test('layout matches a torch fixture', () => {
    const fx = fixture as unknown as {
      graph: unknown;
      params: Record<string, { W: number[]; b?: number[] }>;
    };
    const parsed = parseGraph(fx.graph);
    if (!parsed.ok) throw new Error(parsed.error);
    const g = parsed.graph;
    const h = initParams(
      analyse(g, { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 }),
      1,
    );
    for (const [id, p] of Object.entries(fx.params)) {
      expect(h[id]?.W?.length).toBe(p.W?.length);
      expect(h[id]?.b?.length).toBe(p.b?.length);
    }
    expect(Object.keys(h)).toEqual(Object.keys(fx.params));
  });
});

describe('reconcileParams', () => {
  test('keeps unchanged nodes bit-identical and re-inits a resized Linear', () => {
    const g1 = net([4, 1]);
    const h1 = initParams(an(g1), 1);
    const g2 = net([5, 1]); // l1 resized, so l2 input width changes too
    const h2 = reconcileParams(h1, an(g1), an(g2), 1);
    expect(h2['l1']?.W?.length).toBe(10);
    expect(h2['l2']?.W?.length).toBe(5);

    const g3 = net([4, 1], { extra: true }); // l1, l2 unchanged
    const h3 = reconcileParams(h1, an(g1), an(g3), 1);
    expect(Array.from(h3['l1']?.W ?? [])).toEqual(Array.from(h1['l1']?.W ?? []));
    expect(Array.from(h3['l2']?.b ?? [])).toEqual(Array.from(h1['l2']?.b ?? []));
    expect(h3['zz']?.W?.length).toBe(3);
  });

  test('keeps trained values, re-inits on bias toggle, does not mutate old', () => {
    const g1 = net([3]);
    const h1 = initParams(an(g1), 1);
    const w1 = h1['l1']?.W;
    if (w1) w1[0] = 42;
    const same = reconcileParams(h1, an(g1), an(g1), 1);
    expect(same['l1']?.W?.[0]).toBe(42);
    const g2 = net([3], { bias: false });
    const h2 = reconcileParams(h1, an(g1), an(g2), 1);
    expect(h2['l1']?.b).toBeUndefined();
    expect(h2['l1']?.W?.[0]).toBe(42); // W kept when only bias was toggled
    const back = reconcileParams(h2, an(g2), an(g1), 7);
    expect(back['l1']?.W).toBe(h1['l1']?.W);
    expect(back['l1']?.b?.length).toBe(3);
    expect(h1['l1']?.b).toBeDefined();
  });

  test('re-inits when the shape changes but the element count does not', () => {
    const g1 = net([2, 3]); // l2 is [2, 3]
    const h1 = initParams(an(g1), 1);
    const g2 = net([3, 2]); // l2 is [3, 2]
    const h2 = reconcileParams(h1, an(g1), an(g2), 1);
    expect(h2['l2']).not.toBe(h1['l2']);
    expect(h2['l2']?.W?.length).toBe(6);
    expect(Array.from(h2['l2']?.W ?? [])).not.toEqual(Array.from(h1['l2']?.W ?? []));
  });

  test('drops nodes that left the graph', () => {
    const g1 = net([4, 1]);
    const h1 = initParams(an(g1), 1);
    const g2 = net([4]);
    expect(Object.keys(reconcileParams(h1, an(g1), an(g2), 1))).toEqual(['l1']);
  });
});

describe('json', () => {
  test('NaN, Infinity, holes and strings are dropped', () => {
    const g = net([2]);
    const a = an(g);
    const bad = [
      [NaN, 0, 0, 0],
      [Infinity, 0, 0, 0],
      [1, , 0, 0], // eslint-disable-line no-sparse-arrays
      ['1', 0, 0, 0],
    ];
    for (const W of bad) {
      const r = paramsFromJson({ l1: { W, b: [0, 0] } }, a);
      expect(r.dropped).toEqual(['l1']);
      expect(Object.keys(r.params)).toEqual([]);
    }
    expect(paramsFromJson({ l1: { W: [0, 0, 0, 0], b: [0, NaN] } }, a).dropped).toEqual(['l1']);
  });

  test('paramsToJson has no prototype', () => {
    const g = net([2]);
    expect(Object.getPrototypeOf(paramsToJson(initParams(an(g), 1)))).toBeNull();
  });

  test('round trip', () => {
    const g = net([4, 1]);
    const a = an(g);
    const h = initParams(a, 2);
    const json = JSON.parse(JSON.stringify(paramsToJson(h))) as unknown;
    const { params, dropped } = paramsFromJson(json, a);
    expect(dropped).toEqual([]);
    expect(params).toEqual(h);
  });

  test('mismatched shapes are dropped and reported', () => {
    const g1 = net([4, 1]);
    const json = paramsToJson(initParams(an(g1), 2));
    const g2 = net([5, 1]);
    const { params, dropped } = paramsFromJson(json, an(g2));
    expect(dropped.sort()).toEqual(['l1', 'l2']);
    expect(Object.keys(params)).toEqual([]);
    const g3 = net([4, 2]);
    const r = paramsFromJson(json, an(g3));
    expect(r.dropped).toEqual(['l2']);
    expect(Object.keys(r.params)).toEqual(['l1']);
  });

  test('malformed and unknown entries are dropped', () => {
    const g = net([2]);
    const r = paramsFromJson({ l1: { W: 'no' }, ghost: { W: [1] } }, an(g));
    expect(r.dropped.sort()).toEqual(['ghost', 'l1']);
  });
});

describe('live parameters', () => {
  test('a Linear read only through a Weights tap is initialised', () => {
    const g = parseGraph(tapOnly.graph);
    if (!g.ok) throw new Error('parse');
    const a = analyse(g.graph, info);
    expect(a.order).not.toContain('tap');
    const host = initParams(a, 3);
    expect(Object.keys(host).sort()).toEqual(['lin1', 'lin2', 'tap']);
    expect(host['tap']?.W?.length).toBe(2 * 2);
    expect(host['tap']?.b?.length).toBe(2);
  });

  test('a migrated version 1 graph initialises its Linears under their old ids', () => {
    const g = parseGraph(mlp.graph);
    if (!g.ok) throw new Error('parse');
    const a = analyse(g.graph, info);
    const host = initParams(a, 7);
    expect(Object.keys(host).sort()).toEqual(Object.keys(mlp.params).sort());
  });
});

describe('a non-Linear parameter owner (stub)', () => {
  // No piece owns parameters besides Linear yet, so the owner is added to a real analysis by hand.
  type A = ReturnType<typeof analyse>;
  function withStub(
    shapes: Record<string, number[]>,
    init: A['paramInit'][string],
    type = 'stub',
  ): A {
    const a = analyse(net([2]), info);
    return {
      ...a,
      paramNodes: [...a.paramNodes, 'stub'],
      paramShapes: { ...a.paramShapes, stub: shapes },
      paramInit: { ...a.paramInit, stub: init },
      paramType: { ...a.paramType, stub: type },
    };
  }
  const uni = { kind: 'uniform', bound: 0.5 } as const;

  test('init follows the shapes and the init spec; Linear is unchanged', () => {
    const a = withStub({ g: [2, 3], s: [1] }, uni);
    const h = initParams(a, 5);
    expect(Object.keys(h['stub'] ?? {})).toEqual(['g', 's']);
    expect(h['stub']?.['g']?.length).toBe(6);
    expect(h['stub']?.['s']?.length).toBe(1);
    for (const v of h['stub']?.['g'] ?? []) expect(Math.abs(v)).toBeLessThanOrEqual(0.5);
    expect(initParams(a, 5)['stub']).toEqual(h['stub']);
    expect(h['l1']).toEqual(initParams(analyse(net([2]), info), 5)['l1']);
    const c = initParams(withStub({ W: [1, 1] }, { kind: 'const', value: 2 }), 5);
    expect(Array.from(c['stub']?.['W'] ?? [])).toEqual([2]);
    expect(countParams(h)).toBe(countParams(initParams(analyse(net([2]), info), 5)) + 7);
  });

  test('reconcile keeps unchanged tensors, re-inits a changed owner, adds and drops tensors', () => {
    const a1 = withStub({ g: [2, 3] }, uni);
    const h1 = initParams(a1, 5);
    const same = reconcileParams(h1, a1, withStub({ g: [2, 3] }, uni), 9);
    expect(same['stub']?.['g']).toBe(h1['stub']?.['g']);
    const a2 = withStub({ g: [2, 3], s: [1] }, uni);
    const grown = reconcileParams(h1, a1, a2, 9);
    expect(grown['stub']?.['g']).toBe(h1['stub']?.['g']);
    expect(grown['stub']?.['s']?.length).toBe(1);
    const shrunk = reconcileParams(grown, a2, a1, 9);
    expect(Object.keys(shrunk['stub'] ?? {})).toEqual(['g']);
    const reshaped = reconcileParams(h1, a1, withStub({ g: [3, 3] }, uni), 9);
    expect(reshaped['stub']?.['g']?.length).toBe(9);
    expect(reshaped['stub']?.['g']).not.toBe(h1['stub']?.['g']);
    // the same id and tensors under another piece type (a reused id): fresh values
    const retyped = reconcileParams(h1, a1, withStub({ g: [2, 3] }, uni, 'other'), 9);
    expect(retyped['stub']?.['g']).not.toBe(h1['stub']?.['g']);
    // the owner is not in the old analysis: fresh
    const fresh = reconcileParams(h1, analyse(net([2]), info), a1, 9);
    expect(fresh['stub']?.['g']).not.toBe(h1['stub']?.['g']);
  });

  test('toJson / fromJson round trip, and a mismatched owner is dropped', () => {
    const a = withStub({ g: [2, 3], s: [1] }, uni);
    const h = initParams(a, 5);
    const json = JSON.parse(JSON.stringify(paramsToJson(h))) as unknown;
    const back = paramsFromJson(json, a);
    expect(back.dropped).toEqual([]);
    expect(Array.from(back.params['stub']?.['g'] ?? [])).toEqual(
      Array.from(h['stub']?.['g'] ?? []),
    );
    expect(Array.from(back.params['l1']?.['b'] ?? [])).toEqual(Array.from(h['l1']?.['b'] ?? []));
    const other = paramsFromJson(json, withStub({ g: [2, 4], s: [1] }, uni));
    expect(other.dropped).toEqual(['stub']);
    expect(other.params['l1']).toBeDefined();
  });

  test('an extra key on a Linear without bias is dropped, not thrown', () => {
    const a = analyse(net([2], { bias: false }), info);
    const w = Array.from(initParams(a, 1)['l1']?.['W'] ?? []);
    const r = paramsFromJson({ l1: { W: w, b: [0, 0] } }, a);
    expect(r.dropped).toEqual(['l1']);
    expect(r.params['l1']).toBeUndefined();
  });
});
