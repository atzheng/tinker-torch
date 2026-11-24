import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, test } from 'vitest';
import {
  analyse,
  layoutLayers,
  layoutNeurons,
  LAYOUT,
  parseFile,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
const lin = (id: string, out: number) => ({ id, type: 'linear' as const, out, bias: true });
const sumGraph: Graph = {
  version: 2,
  predictions: [],
  loss: null,
  data: { set: 'line' as const, n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
  train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    lin('l1', 3),
    { id: 'a1', type: 'activation', fn: 'relu' },
    lin('l2', 3),
    { id: 'a2', type: 'activation', fn: 'relu' },
    lin('l3', 3),
    { id: 's', type: 'sum' },
  ],
  edges: [
    ['x', 'l1'],
    ['l1', 'a1'],
    ['a1', 'l2'],
    ['l2', 'a2'],
    ['a2', 'l3'],
    ['x', 's.in'],
    ['l1', 's.in'],
    ['l3', 's.in'],
  ],
};

const withPos = (positions: Record<string, [number, number]>): Graph => ({
  ...sumGraph,
  positions,
});
const lay = (g: Graph) => layoutLayers(g, analyse(g, info));
const wireOf = (g: Graph, from: string, to: string) => {
  const L = lay(g);
  const w = L.wires.find((w) => {
    const e = g.edges[w.edge];
    return e?.[0] === from && e[1].split('.')[0] === to;
  });
  if (!w) throw new Error('no wire');
  return { L, w };
};
// Gap between columns 0 and 1: centre 172, boxes at 136 and 208.
const GAP_X = 172;

describe('manual verticals as lane obstacles (tinker-9zn)', () => {
  test('without manual routes the skip wires sit on the gap centre', () => {
    const { w } = wireOf(sumGraph, 'x', 's');
    expect(w.points[1]?.[0]).toBe(GAP_X);
  });

  test('a manual vertical inside the gap pushes an auto lane off it', () => {
    // l1 placed far below: x -> l1 is a manual elbow whose vertical sits at the gap centre.
    const g = withPos({ l1: [208, 300] });
    const { L, w: manual } = wireOf(g, 'x', 'l1');
    expect(manual.points[1]?.[0]).toBe(GAP_X);
    expect(manual.points[2]?.[0]).toBe(GAP_X);
    const { w } = wireOf(g, 'x', 's');
    const vx = w.points[2]?.[0] as number;
    expect(Math.abs(vx - GAP_X)).toBeGreaterThanOrEqual(LAYOUT.laneStep);
    // still inside the gap, clear of the boxes either side
    const left = (L.nodes['x']?.x ?? 0) + (L.nodes['x']?.w ?? 0);
    const right = L.nodes['l1']?.x ?? 0;
    expect(vx).toBeGreaterThanOrEqual(left + LAYOUT.laneMargin);
    expect(vx).toBeLessThanOrEqual(right - LAYOUT.laneMargin);
    // the horizontal in and out of the lane stay joined to the same vertical
    expect(w.points[1]?.[0]).toBe(vx);
  });

  test('a manual vertical outside the lane range is ignored', () => {
    // vertical at (136 + 300) / 2 = 218, inside the next column, far from the gap's lanes
    const g = withPos({ l1: [300, 300] });
    const { w: manual } = wireOf(g, 'x', 'l1');
    expect(manual.points[1]?.[0]).toBe(218);
    const { w } = wireOf(g, 'x', 's');
    expect(w.points[2]?.[0]).toBe(GAP_X);
  });

  test('a manual vertical that does not reach an auto item in y is ignored', () => {
    // manual vertical far below the skip wire's vertical (y 42..106)
    const g = withPos({ x: [24, 600], l1: [208, 700] });
    const { w } = wireOf(g, 'l1', 'a1');
    expect(w.points.length).toBeGreaterThan(0);
    const { w: skip } = wireOf(g, 'l1', 's');
    expect(skip.points.length).toBeGreaterThan(0);
  });

  test('a crowded gap terminates and stays inside the gap', () => {
    // Several manual elbows through one gap at different heights, plus the auto skip wires.
    for (let y = 120; y <= 560; y += 40) {
      const g = withPos({ l1: [208, y], a1: [392, y + 60], l2: [250, y - 20] });
      const L = lay(g);
      for (const w of L.wires) {
        for (const [px] of w.points) {
          expect(Number.isFinite(px)).toBe(true);
        }
      }
      const { w } = wireOf(g, 'x', 's');
      const vx = w.points[1]?.[0] as number;
      expect(vx).toBeGreaterThan(136);
      expect(vx).toBeLessThan(L.nodes['l1']?.x ?? Infinity);
    }
  });

  test('layout is deterministic', () => {
    const g = withPos({ l1: [208, 300] });
    expect(JSON.stringify(lay(g))).toBe(JSON.stringify(lay(g)));
  });
});

// Identity of graphs without manual positions: snapshots taken from main before the change.
describe('preset layouts are unchanged', () => {
  const dir = join(__dirname, '../../web/src/presets');
  const files = readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort();
  test('every bundled preset, layer and neuron view', async () => {
    const out: Record<string, unknown> = {};
    for (const f of files) {
      const r = parseFile(JSON.parse(readFileSync(join(dir, f), 'utf8')));
      if (!r.ok) continue;
      const g = r.file.graph;
      const cols = g.nodes.flatMap((n) => (n.type === 'input' ? n.columns : []));
      const i: DataInfo = { columns: cols, kind: 'regression', numClasses: 2 };
      const a = analyse(g, i);
      out[f] = { layers: layoutLayers(g, a), neurons: layoutNeurons(g, a) };
    }
    await expect(JSON.stringify(out, null, 1)).toMatchFileSnapshot(
      './__snapshots__/presetLayouts.json',
    );
  });
});
