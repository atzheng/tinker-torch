/** Weight wires (CONTRACTS.md section 9): a Weight with one wire in and one out has no box in layer view. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  collapsedWeights,
  layoutLayers,
  layoutNeurons,
  moveNodes,
  parseGraph,
  serializeGraph,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 };

function build(): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x1', type: 'column', name: 'x1' },
      { id: 'x2', type: 'column', name: 'x2' },
      { id: 'w1', type: 'scalew' },
      { id: 'w2', type: 'scalew' },
      { id: 'sum', type: 'sum' },
      { id: 'b', type: 'bias' },
    ],
    edges: [
      ['x1', 'w1'],
      ['x2', 'w2'],
      ['w1', 'sum'],
      ['w2', 'sum'],
      ['sum', 'b'],
    ],
    predictions: ['b'],
    loss: null,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}
const lay = (g: Graph) => layoutLayers(g, analyse(g, info));

describe('weight wires', () => {
  test('a Weight with one wire in and one out collapses into its wire', () => {
    const g = build();
    expect([...collapsedWeights(g)].sort()).toEqual(['w1', 'w2']);
    const l = lay(g);
    expect(l.nodes['w1']).toBeUndefined();
    expect(l.nodes['w2']).toBeUndefined();
    const weights = l.wires.filter((w) => w.weightNode !== undefined);
    expect(weights.map((w) => [w.weightNode, w.edge, w.viaEdge])).toEqual([
      ['w1', 2, 0],
      ['w2', 3, 1],
    ]);
    expect(l.wires.find((w) => w.edge === 4)?.weightNode).toBeUndefined();
  });

  test('a Weight is zero-width: the column and the Sum are in adjacent columns', () => {
    const l = lay(build());
    expect(l.nodes['x1']?.col).toBe(0);
    expect(l.nodes['sum']?.col).toBe(1);
    expect(l.nodes['b']?.col).toBe(2);
    const w = l.wires.find((x) => x.weightNode === 'w1');
    const a = l.nodes['x1'];
    const b = l.nodes['sum'];
    expect(w?.points[0]).toEqual([(a?.x ?? 0) + (a?.w ?? 0), (a?.y ?? 0) + (a?.h ?? 0) / 2]);
    expect(w?.points.at(-1)?.[0]).toBe(b?.x);
  });

  test('wires into the Sum take different slots along its port', () => {
    const l = lay(build());
    const ys = l.wires.filter((w) => w.weightNode).map((w) => w.points.at(-1)?.[1]);
    expect(new Set(ys).size).toBe(2);
  });

  test('a Weight that is not between a piece and a consumer keeps its box', () => {
    const g = build();
    const dangling: Graph = { ...g, edges: g.edges.filter((e) => e[0] !== 'w1') };
    expect(collapsedWeights(dangling).has('w1')).toBe(false);
    expect(lay(dangling).nodes['w1']).toBeDefined();
    const chained: Graph = { ...g, edges: [['x1', 'w1'], ['w1', 'w2'], ['w2', 'sum']] };
    expect([...collapsedWeights(chained)]).toEqual(['w1']);
    expect(lay(chained).nodes['w2']).toBeDefined();
  });

  test('a Weight with two consumers draws one wire per consumer', () => {
    const g = build();
    const fan: Graph = { ...g, edges: [...g.edges, ['w1', 'b']] };
    expect(lay(fan).wires.filter((w) => w.weightNode === 'w1')).toHaveLength(2);
  });

  test('neuron view keeps the Weight as a box', () => {
    const g = build();
    expect(layoutNeurons(g, analyse(g, info)).boxes['w1']).toBeDefined();
  });

  test('a collapsed Weight cannot be moved and its position is not saved', () => {
    const g = build();
    expect(moveNodes(g, { w1: [100, 100] }).ok).toBe(false);
    const stale: Graph = { ...g, positions: { w1: [100, 100], sum: [300, 40] } };
    expect(JSON.parse(serializeGraph(stale)).positions).toEqual({ sum: [300, 40] });
    expect(lay(stale).nodes['sum']?.x).toBe(300);
  });
});
