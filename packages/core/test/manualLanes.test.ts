import { createHash } from 'node:crypto';
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
    // A second pair placed far below: its manual elbow sits at the gap centre too, but its
    // y-range (628..728) is nowhere near the skip wire's vertical there (42..106).
    const g: Graph = {
      ...sumGraph,
      nodes: [...sumGraph.nodes, { id: 'p', type: 'input', columns: ['x1'] }, lin('q', 2)],
      edges: [...sumGraph.edges, ['p', 'q']],
      positions: { p: [24, 600], q: [208, 700] },
    };
    const { w: manual } = wireOf(g, 'p', 'q');
    expect(manual.points[1]?.[0]).toBe(GAP_X);
    const { w } = wireOf(g, 'x', 's');
    expect(w.points[2]?.[0]).toBe(GAP_X);
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

// Identity of graphs without manual routes: sha256 of each preset's layouts (numbers rounded to
// 1e-6), taken from main before tinker-9zn. A changed hash means a preset's layout changed; if that
// is intended, run with PRINT_HASHES=1 (`PRINT_HASHES=1 pnpm --filter @tinker/core exec vitest run
// test/manualLanes.test.ts --disable-console-intercept`) and paste the printed object here.
const EXPECTED: Record<string, string> = {
  'bias-variance-stage1.json:layers': '62ae74f67f260222',
  'bias-variance-stage1.json:neurons': 'd553631695df9e04',
  'bias-variance-stage2.json:layers': '62ae74f67f260222',
  'bias-variance-stage2.json:neurons': 'd553631695df9e04',
  'bias-variance-stage3.json:layers': '62ae74f67f260222',
  'bias-variance-stage3.json:neurons': 'd553631695df9e04',
  'intercept-mean.json:layers': '9b717aeea21640fc',
  'intercept-mean.json:neurons': '473f267d3487ff3b',
  'intercept-prob.json:layers': '9d2378947f16836b',
  'intercept-prob.json:neurons': 'b134e1e770f209d9',
  'iris.json:layers': 'bc614812d0e70cf7',
  'iris.json:neurons': '03ec1735a543f44b',
  'lecture1-stage0.json:layers': '5b6d140faee9d098',
  'lecture1-stage0.json:neurons': '9e79b6db2b3f2d6d',
  'lecture1-stage1.json:layers': '5b6d140faee9d098',
  'lecture1-stage1.json:neurons': '9e79b6db2b3f2d6d',
  'lecture1-stage2.json:layers': '5b6d140faee9d098',
  'lecture1-stage2.json:neurons': '9e79b6db2b3f2d6d',
  'lecture1-stage3.json:layers': '94cb91e19ad7fa41',
  'lecture1-stage3.json:neurons': '56911c65509bcdf8',
  'lecture1-stage4.json:layers': '51250e2195e5e571',
  'lecture1-stage4.json:neurons': 'afdc241cebc854e9',
  'lecture1-stage5.json:layers': 'bc614812d0e70cf7',
  'lecture1-stage5.json:neurons': '0e1c86a5e80884f2',
  'linear.json:layers': '5a7f9a56459310d1',
  'linear.json:neurons': '127794c6d3e50d9e',
  'logistic.json:layers': 'bc614812d0e70cf7',
  'logistic.json:neurons': 'fd8ca53474014479',
  'mlp.json:layers': '22aca25af872e3fc',
  'mlp.json:neurons': '7266fa05fcc97b65',
  'scalar.json:layers': 'b05b3882ec3b2a2c',
  'scalar.json:neurons': '52128aeafa868974',
  'sine.json:layers': '22aca25af872e3fc',
  'sine.json:neurons': '1a2acaffdcbb303e',
  'skip.json:layers': '09062541e8424d63',
  'skip.json:neurons': 'a5c59d9eaf2438b6',
  'softmax.json:layers': 'bc614812d0e70cf7',
  'softmax.json:neurons': 'b65a186957d3d187',
  'tree-class-stage1.json:layers': '5b15ec1aad6b30ea',
  'tree-class-stage1.json:neurons': '7c1e48d6eb8a63ea',
  'tree-class-stage2.json:layers': '5b15ec1aad6b30ea',
  'tree-class-stage2.json:neurons': '7c1e48d6eb8a63ea',
  'tree-class-stage3.json:layers': '5b15ec1aad6b30ea',
  'tree-class-stage3.json:neurons': '7c1e48d6eb8a63ea',
  'xor.json:layers': '5d336cb6020758c1',
  'xor.json:neurons': '55f38d4117244559',
};
describe('preset layouts are unchanged', () => {
  const dir = join(__dirname, '../../web/src/presets');
  const files = readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort();
  const hash = (v: unknown) =>
    createHash('sha256')
      .update(
        JSON.stringify(v, (_k, x: unknown) =>
          typeof x === 'number' ? Math.round(x * 1e6) / 1e6 : x,
        ),
      )
      .digest('hex')
      .slice(0, 16);
  test('every bundled preset, layer and neuron view', () => {
    const got: Record<string, string> = {};
    for (const f of files) {
      const r = parseFile(JSON.parse(readFileSync(join(dir, f), 'utf8')));
      if (!r.ok) continue;
      const g = r.file.graph;
      const cols = g.nodes.flatMap((n) => (n.type === 'input' ? n.columns : []));
      const i: DataInfo = { columns: cols, kind: 'regression', numClasses: 2 };
      const a = analyse(g, i);
      got[`${f}:layers`] = hash(layoutLayers(g, a));
      got[`${f}:neurons`] = hash(layoutNeurons(g, a));
    }
    if (process.env['PRINT_HASHES']) console.log(JSON.stringify(got, null, 2));
    expect(got).toEqual(EXPECTED);
  });
});
