import { describe, expect, test } from 'vitest';
import fixture from '../../engine/test/fixtures/linear_bias_d2.json';
import {
  analyse,
  countParams,
  initParams,
  paramsFromJson,
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
  nodes.push({ id: 'out', type: 'output', link: 'identity' }, { id: 'y', type: 'target' });
  nodes.push({ id: 'loss', type: 'loss', fn: 'squared_error' });
  edges.push([prev, 'out'], ['out', 'loss.pred'], ['y', 'loss.target']);
  return {
    version: 1,
    nodes,
    edges,
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
      const h = initParams(g, a, 3);
      let inW = 2;
      widths.forEach((w, i) => {
        const p = h[`l${i + 1}`];
        expect(p?.W.length).toBe(inW * w);
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
    const p = initParams(g, an(g), 1)['l1']!.W; // eslint-disable-line @typescript-eslint/no-non-null-assertion
    expect(Math.max(...p)).toBeGreaterThan(0.5 / Math.SQRT2);
    expect(Math.min(...p)).toBeLessThan(-0.5 / Math.SQRT2);
  });

  test('no bias gives no b', () => {
    const g = net([3], { bias: false });
    expect(initParams(g, an(g), 1)['l1']?.b).toBeUndefined();
  });

  test('same seed same values, different seed different', () => {
    const g = net([4, 1]);
    const a = an(g);
    expect(paramsToJson(initParams(g, a, 5))).toEqual(paramsToJson(initParams(g, a, 5)));
    expect(paramsToJson(initParams(g, a, 5))).not.toEqual(paramsToJson(initParams(g, a, 6)));
  });

  test('adding a node leaves the other nodes unchanged', () => {
    const g1 = net([4, 1]);
    const g2 = net([4, 1], { extra: true });
    const h1 = initParams(g1, an(g1), 9);
    const h2 = initParams(g2, an(g2), 9);
    expect(h2['zz']).toBeDefined();
    expect(h2['l1']).toEqual(h1['l1']);
    expect(h2['l2']).toEqual(h1['l2']);
  });

  test('layout matches a torch fixture', () => {
    const fx = fixture as unknown as {
      graph: Graph;
      params: Record<string, { W: number[]; b?: number[] }>;
    };
    const g = fx.graph;
    const h = initParams(
      g,
      analyse(g, { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 }),
      1,
    );
    for (const [id, p] of Object.entries(fx.params)) {
      expect(h[id]?.W.length).toBe(p.W.length);
      expect(h[id]?.b?.length).toBe(p.b?.length);
    }
    expect(Object.keys(h)).toEqual(Object.keys(fx.params));
  });
});

describe('reconcileParams', () => {
  test('keeps unchanged nodes bit-identical and re-inits a resized Linear', () => {
    const g1 = net([4, 1]);
    const h1 = initParams(g1, an(g1), 1);
    const g2 = net([5, 1]); // l1 resized, so l2 input width changes too
    const h2 = reconcileParams(h1, g2, an(g2), 1);
    expect(h2['l1']?.W.length).toBe(10);
    expect(h2['l2']?.W.length).toBe(5);

    const g3 = net([4, 1], { extra: true }); // l1, l2 unchanged
    const h3 = reconcileParams(h1, g3, an(g3), 1);
    expect(Array.from(h3['l1']?.W ?? [])).toEqual(Array.from(h1['l1']?.W ?? []));
    expect(Array.from(h3['l2']?.b ?? [])).toEqual(Array.from(h1['l2']?.b ?? []));
    expect(h3['zz']?.W.length).toBe(3);
  });

  test('keeps trained values, re-inits on bias toggle, does not mutate old', () => {
    const g1 = net([3]);
    const h1 = initParams(g1, an(g1), 1);
    h1['l1']!.W[0] = 42; // eslint-disable-line @typescript-eslint/no-non-null-assertion
    const same = reconcileParams(h1, g1, an(g1), 1);
    expect(same['l1']?.W[0]).toBe(42);
    const g2 = net([3], { bias: false });
    const h2 = reconcileParams(h1, g2, an(g2), 1);
    expect(h2['l1']?.b).toBeUndefined();
    expect(h2['l1']?.W[0]).not.toBe(42);
    expect(h1['l1']?.b).toBeDefined();
  });

  test('drops nodes that left the graph', () => {
    const g1 = net([4, 1]);
    const h1 = initParams(g1, an(g1), 1);
    const g2 = net([4]);
    expect(Object.keys(reconcileParams(h1, g2, an(g2), 1))).toEqual(['l1']);
  });
});

describe('json', () => {
  test('round trip', () => {
    const g = net([4, 1]);
    const a = an(g);
    const h = initParams(g, a, 2);
    const json = JSON.parse(JSON.stringify(paramsToJson(h))) as unknown;
    const { params, dropped } = paramsFromJson(json, g, a);
    expect(dropped).toEqual([]);
    expect(params).toEqual(h);
  });

  test('mismatched shapes are dropped and reported', () => {
    const g1 = net([4, 1]);
    const json = paramsToJson(initParams(g1, an(g1), 2));
    const g2 = net([5, 1]);
    const { params, dropped } = paramsFromJson(json, g2, an(g2));
    expect(dropped.sort()).toEqual(['l1', 'l2']);
    expect(Object.keys(params)).toEqual([]);
    const g3 = net([4, 2]);
    const r = paramsFromJson(json, g3, an(g3));
    expect(r.dropped).toEqual(['l2']);
    expect(Object.keys(r.params)).toEqual(['l1']);
  });

  test('malformed and unknown entries are dropped', () => {
    const g = net([2]);
    const r = paramsFromJson({ l1: { W: 'no' }, ghost: { W: [1] } }, g, an(g));
    expect(r.dropped.sort()).toEqual(['ghost', 'l1']);
  });
});
