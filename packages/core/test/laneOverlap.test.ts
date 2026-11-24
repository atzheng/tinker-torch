import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import {
  analyse,
  layoutLayers,
  LAYOUT,
  parseFile,
  sample,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const dir = new URL('../../web/src/presets/', import.meta.url);
const names = readdirSync(dir)
  .filter((f) => f.endsWith('.json'))
  .map((f) => f.slice(0, -5))
  .sort();

type Seg = [number, number, number, number];

/** Overlap length of two axis-aligned collinear segments (-Infinity when not collinear; negative = the gap between them). */
function overlap(p: Seg, q: Seg): number {
  const hp = p[1] === p[3];
  const hq = q[1] === q[3];
  const span = (s: Seg, h: boolean): [number, number] =>
    h ? [Math.min(s[0], s[2]), Math.max(s[0], s[2])] : [Math.min(s[1], s[3]), Math.max(s[1], s[3])];
  if (hp && hq && Math.abs(p[1] - q[1]) < 1e-6) {
    const [a0, a1] = span(p, true);
    const [b0, b1] = span(q, true);
    return Math.min(a1, b1) - Math.max(a0, b0);
  }
  if (!hp && !hq && Math.abs(p[0] - q[0]) < 1e-6) {
    const [a0, a1] = span(p, false);
    const [b0, b1] = span(q, false);
    return Math.min(a1, b1) - Math.max(a0, b0);
  }
  return -Infinity;
}

function segs(points: [number, number][]): Seg[] {
  const out: Seg[] = [];
  for (let i = 0; i + 1 < points.length; i++) {
    const a = points[i] as [number, number];
    const b = points[i + 1] as [number, number];
    out.push([a[0], a[1], b[0], b[1]]);
  }
  return out;
}

const fallback: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
function infoOf(g: Graph): DataInfo {
  try {
    return sample(g.data).info;
  } catch {
    return fallback; // rows of a real dataset are not loaded in this test; geometry does not depend on them
  }
}

function overlaps(g: Graph): string[] {
  const L = layoutLayers(g, analyse(g, infoOf(g)));
  const found: string[] = [];
  for (let i = 0; i < L.wires.length; i++) {
    for (let j = i + 1; j < L.wires.length; j++) {
      const wi = L.wires[i];
      const wj = L.wires[j];
      if (!wi || !wj) continue;
      if (g.edges[wi.edge]?.[0] === g.edges[wj.edge]?.[0]) continue; // fan-out may share the trunk
      for (const p of segs(wi.points))
        for (const q of segs(wj.points)) {
          const o = overlap(p, q);
          // Vertical runs must also keep `laneClear` apart end to end; horizontal ones only must not overlap.
          const vertical = p[0] === p[2];
          if (o > (vertical ? -LAYOUT.laneClear : 0)) found.push(`${wi.edge}/${wj.edge}: ${p.join(',')} ~ ${q.join(',')} (${o})`);
        }
    }
  }
  return found;
}

type Before = Record<string, { size: [number, number]; wires: [number, [number, number][], [number, number]][] }>;
const before = JSON.parse(
  readFileSync(new URL('./fixtures/preset-wires-before-lanes.json', import.meta.url), 'utf8'),
) as Before;

describe('no two non-fan wires share a collinear run (tinker-fsv)', () => {
  test.each(names)('%s', (name) => {
    const r = parseFile(JSON.parse(readFileSync(new URL(`${name}.json`, dir), 'utf8')));
    if (!r.ok) throw new Error(r.error);
    expect(overlaps(r.file.graph)).toEqual([]);
  });
});

describe('layouts without overlaps are unchanged', () => {
  // `fixtures/preset-wires-before-lanes.json` holds every bundled preset's wires from before lanes
  // existed. A wire moves only if its old route had a run that clashed with another source's.
  test.each(Object.keys(before))('%s', (name) => {
    const r = parseFile(JSON.parse(readFileSync(new URL(`${name}.json`, dir), 'utf8')));
    if (!r.ok) throw new Error(r.error);
    const g = r.file.graph;
    const old = before[name];
    if (!old) throw new Error(name);
    const L = layoutLayers(g, analyse(g, infoOf(g)));
    expect(L.size).toEqual(old.size);
    const oldSegs = old.wires.map(([edge, pts]) => ({ edge, segs: segs(pts) }));
    for (const [edge, pts, label] of old.wires) {
      const w = L.wires.find((x) => x.edge === edge);
      expect(w).toBeDefined();
      if (JSON.stringify(w?.points) === JSON.stringify(pts)) {
        expect(w?.labelAt).toEqual(label);
        continue;
      }
      // Moved: some other source's old run must have been within `laneClear` of one of its runs.
      const from = g.edges[edge]?.[0];
      const clashed = oldSegs.some(
        (o) =>
          g.edges[o.edge]?.[0] !== from &&
          o.segs.some((p) => segs(pts).some((q) => overlap(p, q) > -LAYOUT.laneClear)),
      );
      expect(clashed, `${name} wire ${edge} moved without a clash`).toBe(true);
    }
  });

  test('the skip preset: lin1 -> add.b no longer continues lin2 -> add.a\'s vertical', () => {
    const r = parseFile(JSON.parse(readFileSync(new URL('skip.json', dir), 'utf8')));
    if (!r.ok) throw new Error(r.error);
    const g = r.file.graph;
    const L = layoutLayers(g, analyse(g, infoOf(g)));
    const xs = (edge: number) => new Set(segs(L.wires.find((w) => w.edge === edge)?.points ?? []).filter((s) => s[0] === s[2] && s[1] !== s[3]).map((s) => s[0]));
    const skipEdge = g.edges.findIndex((e) => e[1] === 'add1.a');
    const nearEdge = g.edges.findIndex((e) => e[1] === 'add1.b');
    const common = [...xs(skipEdge)].filter((x) => xs(nearEdge).has(x));
    expect(common).toEqual([]);
  });
});

describe('lanes with several skip wires into one variadic port', () => {
  const lin = (id: string, out: number) => ({ id, type: 'linear' as const, out, bias: true });
  const graph: Graph = {
    version: 2,
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
    predictions: [],
    loss: null,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
  test('no collinear overlaps, every wire still ends where it did, labels sit on a run', () => {
    expect(overlaps(graph)).toEqual([]);
    const L = layoutLayers(graph, analyse(graph, fallback));
    for (const w of L.wires) {
      const b = L.nodes[graph.edges[w.edge]?.[1].split('.')[0] ?? ''];
      const last = w.points[w.points.length - 1];
      expect(last?.[0]).toBe(b?.x);
      const onRun = segs(w.points).some((s) => s[1] === s[3] && w.labelAt[1] === s[1] && w.labelAt[0] >= Math.min(s[0], s[2]) && w.labelAt[0] <= Math.max(s[0], s[2]));
      expect(onRun).toBe(true);
    }
    const lanes = new Set(L.wires.flatMap((w) => segs(w.points).filter((s) => s[0] === s[2] && s[1] !== s[3]).map((s) => s[0])));
    expect(lanes.size).toBeGreaterThan(3);
  });
});
