/** A Weight is an ordinary piece box in layer view (CONTRACTS.md section 9): no wire carries a weight. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  layoutLayers,
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
    ],
    edges: [
      ['x1', 'w1'],
      ['x2', 'w2'],
      ['w1', 'sum'],
      ['w2', 'sum'],
    ],
    predictions: ['sum'],
    loss: null,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}
const lay = (g: Graph) => layoutLayers(g, analyse(g, info));

describe('weight boxes', () => {
  test('every Weight has a box and every edge a wire', () => {
    const l = lay(build());
    expect(l.nodes.w1).toBeDefined();
    expect(l.nodes.w2).toBeDefined();
    expect(l.wires.map((w) => w.edge)).toEqual([0, 1, 2, 3]);
    expect(l.nodes.w1?.col).toBe((l.nodes.x1?.col ?? 0) + 1);
  });

  test('a Weight can be moved and keeps its position in the file', () => {
    const m = moveNodes(build(), { w1: [300, 200] });
    if (!m.ok) throw new Error(m.error);
    expect(lay(m.graph).nodes.w1).toMatchObject({ x: 300, y: 200 });
    expect(JSON.parse(serializeGraph(m.graph)).positions).toEqual({ w1: [300, 200] });
  });
});
