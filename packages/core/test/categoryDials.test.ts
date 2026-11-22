import { describe, expect, it } from 'vitest';
import {
  analyse,
  categoryDials,
  layoutNeurons,
  parseGraph,
  sample,
  type DataConfig,
  type Graph,
} from '../src/index.js';

const cfg: DataConfig = {
  set: 'delivery_late',
  n: 100,
  noise: 0.3,
  val: 0.3,
  seed: 2,
  standardize: false,
};
const ZONE = sample(cfg).info;

function graph(nodes: object[], edges: [string, string][]): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'z', type: 'input', columns: ['x4'] },
      ...nodes,
      { id: 'p', type: 'link', fn: 'sigmoid' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [...edges, ['p', 'err.pred'], ['y', 'err.target'], ['err', 'loss']],
    predictions: ['p'],
    loss: 'loss',
    data: cfg,
    train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const lookup = (width: number) =>
  graph(
    [{ id: 'look', type: 'lookup', width }],
    [
      ['z', 'look'],
      ['look', 'p'],
    ],
  );
const onehot = (out: number) =>
  graph(
    [
      { id: 'oh', type: 'onehot' },
      { id: 'lin', type: 'linear', out, bias: true },
    ],
    [
      ['z', 'oh'],
      ['oh', 'lin'],
      ['lin', 'p'],
    ],
  );

describe('category dials', () => {
  it('Lookup of width 1: one dial per category, each its own table row', () => {
    const g = lookup(1);
    const a = analyse(g, ZONE);
    const d = categoryDials(g, a, ZONE);
    expect(d).toHaveLength(4);
    expect(d.map((x) => x.index)).toEqual([0, 1, 2, 3]);
    expect(d.map((x) => x.name)).toEqual(ZONE.categories?.['x4']);
    expect(d.map((x) => x.target)).toEqual([0, 1, 2, 3].map((i) => ({ node: 'look', i, j: 0 })));
    const l = layoutNeurons(g, a);
    expect(l.units.filter((u) => u.node === 'look').map((u) => u.kind)).toEqual(
      Array<string>(4).fill('cat'),
    );
  });

  it('One-hot into Linear(out 1): dials point at the Linear weight rows', () => {
    const g = onehot(1);
    const a = analyse(g, ZONE);
    const d = categoryDials(g, a, ZONE);
    expect(d.map((x) => x.target)).toEqual([0, 1, 2, 3].map((i) => ({ node: 'lin', i, j: 0 })));
    const l = layoutNeurons(g, a);
    expect(l.units.filter((u) => u.node === 'oh')).toHaveLength(4);
    expect(l.edges.filter((e) => e.node === 'lin')).toHaveLength(4);
  });

  it('One-hot into a wider Linear: dials drawn but without a scalar target', () => {
    const g = onehot(2);
    const d = categoryDials(g, analyse(g, ZONE), ZONE);
    expect(d).toHaveLength(4);
    expect(d.every((x) => x.target === null)).toBe(true);
  });

  it('Lookup wider than 1 stays a box with no dials', () => {
    const g = lookup(3);
    const a = analyse(g, ZONE);
    expect(categoryDials(g, a, ZONE).filter((x) => x.node === 'look')).toEqual([]);
    expect(layoutNeurons(g, a).units.some((u) => u.node === 'look')).toBe(false);
  });
});
