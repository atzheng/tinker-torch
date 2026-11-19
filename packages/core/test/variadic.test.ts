/**
 * Variadic input ports (CONTRACTS.md section 3). No real piece has one yet, so the registry's
 * `add` entry is swapped for a stub with the single variadic input `in` while this file runs.
 */
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import {
  analyse,
  connect,
  CORE_PIECES,
  deleteEdge,
  deleteNode,
  insertOnWire,
  isVariadicPort,
  layoutLayers,
  layoutNeurons,
  PALETTE,
  resolveEdge,
  rewireEdge,
  swapPiece,
  variadicPortsOf,
  variadicSlotY,
  type DataInfo,
  type Edge,
  type Graph,
  type Node,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
const original = CORE_PIECES.add;
beforeAll(() => {
  (CORE_PIECES as Record<string, unknown>)['add'] = {
    family: 'math',
    inputPorts: ['in'],
    variadicPorts: ['in'],
    defaults: () => ({ type: 'add' }),
    outShape: (_n: unknown, s: Record<string, unknown>) =>
      s['in'] ?? { error: 'Add needs an input.' },
    paramShapes: () => ({}),
    paramInit: () => ({ kind: 'const', value: 0 }),
    torch: () => 'sum',
  };
});
afterAll(() => {
  (CORE_PIECES as Record<string, unknown>)['add'] = original;
});

const lin = (id: string, out: number): Node => ({ id, type: 'linear', out, bias: true });
function g(
  nodes: Node[],
  edges: Edge[],
  predictions: string[] = [],
  loss: string | null = null,
): Graph {
  return {
    version: 2,
    nodes,
    edges,
    predictions,
    loss,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
}
const base = (): Node[] => [
  { id: 'x', type: 'input', columns: ['x1'] },
  lin('a', 1),
  lin('b', 1),
  lin('c', 3),
  { id: 's', type: 'add' },
];
function ok<T extends { ok: boolean }>(r: T): Extract<T, { ok: true }> {
  expect(r.ok).toBe(true);
  return r as Extract<T, { ok: true }>;
}
const into = (gr: Graph, dest: string): string[] =>
  gr.edges.filter((e) => e[1] === dest || e[1].startsWith(`${dest}.`)).map((e) => e[0]);

describe('registry flag', () => {
  test('stub is variadic, real pieces are not', () => {
    expect(variadicPortsOf('add')).toEqual(['in']);
    expect(isVariadicPort('add', 'in')).toBe(true);
    expect(isVariadicPort('linear', 'in')).toBe(false);
    expect(variadicPortsOf('subtract')).toEqual([]);
  });
  test('not in the palette', () => {
    expect(PALETTE.some((p) => p.kind === 'piece' && p.key === 'sum')).toBe(false);
  });
  test('resolveEdge accepts several wires into the port', () => {
    const nodes = base();
    expect(resolveEdge(nodes, ['a', 's']).ok).toBe(true);
    expect(resolveEdge(nodes, ['b', 's.in']).ok).toBe(true);
  });
});

describe('analyse', () => {
  const edges: Edge[] = [
    ['x', 'a'],
    ['x', 'b'],
    ['b', 's'],
    ['a', 's'],
  ];
  test('many wires into one port: no double-feed issue, edge order kept', () => {
    const a = analyse(g(base(), edges, ['s']), info);
    expect(a.issues.filter((i) => i.message.includes('two wires'))).toEqual([]);
    expect(a.inputs['s']?.['in']).toEqual([
      { edge: 2, from: 'b' },
      { edge: 3, from: 'a' },
    ]);
    expect(a.shape['s']).toEqual({ kind: 'example', width: 1 });
    expect(a.edgeShape[2]).toEqual({ kind: 'example', width: 1 });
    expect(a.order.indexOf('s')).toBeGreaterThan(a.order.indexOf('a'));
    expect(a.order.indexOf('s')).toBeGreaterThan(a.order.indexOf('b'));
  });
  test('region follows every wire', () => {
    const a = analyse(g(base(), edges, ['s']), info);
    expect(a.region['a']).toBe('model');
    expect(a.region['b']).toBe('model');
  });
  test('plain ports still allow one wire', () => {
    const a = analyse(
      g(
        base(),
        [
          ['x', 'a'],
          ['x', 'b'],
          ['x', 'c'],
          ['a', 'c'],
        ],
        [],
      ),
      info,
    );
    expect(a.issues.some((i) => i.message.includes('two wires into its in input'))).toBe(true);
  });
  test('shape mismatch is one issue on the offending edge', () => {
    const e: Edge[] = [
      ['x', 'a'],
      ['x', 'c'],
      ['a', 's'],
      ['c', 's'],
    ];
    const a = analyse(g(base(), e, ['s']), info);
    const bad = a.issues.filter((i) => i.at.edge !== undefined);
    expect(bad).toHaveLength(1);
    expect(bad[0]?.at.edge).toBe(3);
    expect(bad[0]?.message).toContain('same kind of value');
    expect(a.shape['s']).toBeUndefined();
    expect(a.trainable).toBe(false);
  });
  test('three wires, only the third mismatching', () => {
    const e: Edge[] = [
      ['x', 'a'],
      ['x', 'b'],
      ['x', 'c'],
      ['a', 's'],
      ['b', 's'],
      ['c', 's'],
    ];
    const a = analyse(g(base(), e, ['s']), info);
    expect(a.issues.filter((i) => i.at.edge === 5)).toHaveLength(1);
    expect(a.issues.filter((i) => i.at.edge === 4)).toHaveLength(0);
  });
  test('zero wires is a dangling input', () => {
    const a = analyse(g(base(), [['x', 'a']], ['s']), info);
    expect(a.issues.some((i) => i.at.node === 's' && i.message.includes('needs an input'))).toBe(
      true,
    );
  });
  test('an unknown upstream shape gives no extra issue at the node', () => {
    const a = analyse(
      g(
        base(),
        [
          ['a', 's'],
          ['b', 's'],
        ],
        ['s'],
      ),
      info,
    );
    expect(a.issues.every((i) => i.at.node !== 's')).toBe(true);
  });
  test('a loop through a variadic port is reported', () => {
    const e: Edge[] = [
      ['x', 'a'],
      ['a', 's'],
      ['s', 'b'],
      ['b', 's'],
    ];
    const a = analyse(g(base(), e, []), info);
    expect(a.issues.some((i) => i.message.includes('loop'))).toBe(true);
  });
});

describe('duplicate wire from a file', () => {
  test('the same source twice into a variadic port is one issue on the second edge', () => {
    const gr = g(
      base(),
      [
        ['x', 'a'],
        ['a', 's'],
        ['a', 's'],
      ],
      ['s'],
    );
    const a = analyse(gr, info);
    const bad = a.issues.filter((i) => i.at.edge !== undefined);
    expect(bad).toHaveLength(1);
    expect(bad[0]?.at.edge).toBe(2);
    expect(a.inputs['s']?.['in']).toEqual([{ edge: 1, from: 'a' }]);
  });
});

describe('edits', () => {
  test('rewireEdge onto the same variadic port changes nothing', () => {
    const gr = g(
      base(),
      [
        ['x', 'a'],
        ['x', 'b'],
        ['a', 's'],
        ['b', 's'],
      ],
      ['s'],
    );
    const r = ok(rewireEdge(gr, 2, 'to', 's.in'));
    expect(r.graph.edges).toEqual(gr.edges);
  });
  test('variadicSlotY matches the layout', () => {
    expect(variadicSlotY({ y: 10, h: 60 }, 0, 1, 0, 1)).toBe(40);
    expect(variadicSlotY({ y: 10, h: 60 }, 0, 1, 0, 3)).toBeCloseTo(10 + (60 * 0.75) / 2);
  });
  const wired = (): Graph =>
    g(
      base(),
      [
        ['x', 'a'],
        ['x', 'b'],
        ['a', 's'],
        ['b', 's'],
      ],
      ['s'],
    );

  test('connect appends and never replaces', () => {
    const r = ok(
      connect(
        g(
          base(),
          [
            ['x', 'a'],
            ['x', 'b'],
            ['a', 's'],
          ],
          ['s'],
        ),
        'b',
        's',
      ),
    );
    expect(into(r.graph, 's')).toEqual(['a', 'b']);
    const more = ok(connect(r.graph, 'c', 's.in'));
    expect(into(more.graph, 's')).toEqual(['a', 'b', 'c']);
    expect(analyse(more.graph, info).inputs['s']?.['in']).toHaveLength(3);
  });
  test('connect refuses the same source twice', () => {
    expect(connect(wired(), 'a', 's').ok).toBe(false);
  });
  test('connect onto a plain port still replaces', () => {
    const r = ok(
      connect(
        g(
          [...base(), lin('d', 1)],
          [
            ['x', 'd'],
            ['a', 'd'],
          ],
          [],
        ),
        'b',
        'd',
      ),
    );
    expect(into(r.graph, 'd')).toEqual(['b']);
  });
  test('connect refuses a loop through the port', () => {
    const gr = g(
      base(),
      [
        ['a', 's'],
        ['s', 'b'],
      ],
      [],
    );
    expect(connect(gr, 'b', 's').ok).toBe(false);
  });
  test('deleteEdge removes just that wire', () => {
    const r = ok(deleteEdge(wired(), 2));
    expect(into(r.graph, 's')).toEqual(['b']);
  });
  test('insertOnWire splices one wire in place', () => {
    const r = ok(insertOnWire(wired(), 2, { type: 'activation', fn: 'relu' }));
    expect(r.graph.edges).toEqual([
      ['x', 'a'],
      ['x', 'b'],
      ['a', 'activation1'],
      ['activation1', 's'],
      ['b', 's'],
    ]);
    const a = analyse(r.graph, info);
    expect(a.inputs['s']?.['in']?.map((w) => w.from)).toEqual(['activation1', 'b']);
  });
  test('insertOnWire of a variadic piece goes in on its port', () => {
    const r = ok(
      insertOnWire(
        g(
          base(),
          [
            ['x', 'a'],
            ['a', 'b'],
          ],
          [],
        ),
        1,
        { type: 'add' },
      ),
    );
    expect(r.graph.edges[1]).toEqual(['a', 'add1']);
  });
  test('rewireEdge onto a variadic port appends; off it removes only that wire', () => {
    const start = g(
      base(),
      [
        ['x', 'a'],
        ['x', 'b'],
        ['a', 's'],
        ['b', 's'],
        ['x', 'c'],
        ['x', 'lone'],
      ],
      [],
    );
    const withLone = { ...start, nodes: [...start.nodes, lin('lone', 1)] };
    const r0 = ok(rewireEdge(withLone, 5, 'to', 's'));
    expect(into(r0.graph, 's')).toEqual(['a', 'b', 'x']);
    const off = ok(rewireEdge(r0.graph, 2, 'to', 'c'));
    expect(into(off.graph, 's')).toEqual(['b', 'x']);
    expect(into(off.graph, 'c')).toEqual(['a']);
  });
  test('rewire the source end keeps the place', () => {
    const r = ok(rewireEdge(wired(), 2, 'from', 'c'));
    expect(into(r.graph, 's')).toEqual(['c', 'b']);
  });
  test('deleteNode: several wires in, does not close', () => {
    const gr = g(
      [...base(), lin('t', 1)],
      [
        ['x', 'a'],
        ['x', 'b'],
        ['a', 's'],
        ['b', 's'],
        ['s', 't'],
      ],
      ['s'],
    );
    const r = ok(deleteNode(gr, 's'));
    expect(r.graph.edges).toEqual([
      ['x', 'a'],
      ['x', 'b'],
    ]);
    expect(r.graph.predictions).toEqual([]);
  });
  test('deleteNode: one wire into the variadic port closes the gap and moves pins', () => {
    const gr = g(
      [...base(), lin('t', 1)],
      [
        ['x', 'a'],
        ['a', 's'],
        ['s', 't'],
      ],
      ['s'],
    );
    const r = ok(deleteNode(gr, 's'));
    expect(r.graph.edges).toEqual([
      ['x', 'a'],
      ['a', 't'],
    ]);
    expect(r.graph.predictions).toEqual(['a']);
  });
  test('deleteNode: a node feeding a variadic port closes into it, once', () => {
    const nodes = [...base(), lin('m', 1)];
    const r = ok(
      deleteNode(
        g(
          nodes,
          [
            ['x', 'a'],
            ['a', 'm'],
            ['m', 's'],
            ['b', 's'],
          ],
          [],
        ),
        'm',
      ),
    );
    expect(r.graph.edges).toEqual([
      ['x', 'a'],
      ['a', 's'],
      ['b', 's'],
    ]);
    const dup = ok(
      deleteNode(
        g(
          nodes,
          [
            ['x', 'a'],
            ['a', 'm'],
            ['m', 's'],
            ['a', 's'],
          ],
          [],
        ),
        'm',
      ),
    );
    expect(dup.graph.edges).toEqual([
      ['x', 'a'],
      ['a', 's'],
    ]);
  });
  test('deleteNode: a source with no wire in just drops its wires out', () => {
    const r = ok(
      deleteNode(
        g(
          base(),
          [
            ['a', 's'],
            ['b', 's'],
          ],
          [],
        ),
        'a',
      ),
    );
    expect(r.graph.edges).toEqual([['b', 's']]);
  });
  test('swap needs identical ports and variadic flags', () => {
    expect(swapPiece(wired(), 's', { type: 'subtract' }).ok).toBe(false);
    expect(swapPiece(wired(), 's', { type: 'add' }).ok).toBe(true);
    expect(swapPiece(wired(), 'a', { type: 'linear', out: 2, bias: true }).ok).toBe(true);
  });
  test('swap between a variadic and a plain piece with the same port names is refused', () => {
    // restore the plain add (ports a, b) is not same-named; use a temporary plain one-port piece
    const saved = CORE_PIECES.square;
    (CORE_PIECES as Record<string, unknown>)['square'] = {
      ...CORE_PIECES.add,
      variadicPorts: undefined,
    };
    try {
      expect(swapPiece(wired(), 's', { type: 'square' }).ok).toBe(false);
    } finally {
      (CORE_PIECES as Record<string, unknown>)['square'] = saved;
    }
  });
});

describe('layout', () => {
  const graph = (k: number): Graph => {
    const names = ['a', 'b', 'c', 'd', 'e'].slice(0, k);
    const nodes: Node[] = [
      { id: 'x', type: 'input', columns: ['x1'] },
      ...names.map((n) => lin(n, 1)),
      { id: 's', type: 'add' },
    ];
    const edges: Edge[] = [
      ...names.map((n): Edge => ['x', n]),
      ...names.map((n): Edge => [n, 's']),
    ];
    return g(nodes, edges, ['s']);
  };
  for (const view of ['layer', 'neuron'] as const) {
    const run = (gr: Graph) => {
      const a = analyse(gr, info);
      return view === 'layer' ? layoutLayers(gr, a) : layoutNeurons(gr, a);
    };
    const boxOf = (L: ReturnType<typeof run>, id: string) =>
      'boxes' in L ? L.boxes[id] : L.nodes[id];
    test(`${view}: k wires end at distinct heights inside the box, ordered by source`, () => {
      for (const k of [2, 3, 5]) {
        const gr = graph(k);
        const L = run(gr);
        const box = boxOf(L, 's');
        expect(box).toBeDefined();
        const ys = gr.edges.flatMap((e, i) => {
          if (e[1] !== 's') return [];
          const w = L.wires.find((q) => q.edge === i);
          return w ? [{ from: e[0], y: w.points[w.points.length - 1]?.[1] ?? NaN }] : [];
        });
        expect(ys).toHaveLength(k);
        const sorted = [...ys].sort((p, q) => p.y - q.y);
        for (let i = 1; i < sorted.length; i++) {
          expect((sorted[i]?.y ?? 0) - (sorted[i - 1]?.y ?? 0)).toBeGreaterThan(2);
        }
        for (const { y } of ys) {
          expect(y).toBeGreaterThan(box?.y ?? 0);
          expect(y).toBeLessThan((box?.y ?? 0) + (box?.h ?? 0));
        }
        const srcY = (id: string) => boxOf(L, id)?.y ?? 0;
        const bySrc = [...ys].sort((p, q) => srcY(p.from) - srcY(q.from)).map((q) => q.from);
        expect(sorted.map((q) => q.from)).toEqual(bySrc);
      }
    });
    test(`${view}: one wire sits at the centre like a plain port`, () => {
      const L = run(graph(1));
      const box = boxOf(L, 's');
      const w = L.wires.find((q) => q.edge === 1);
      expect(w?.points[w.points.length - 1]?.[1]).toBeCloseTo((box?.y ?? 0) + (box?.h ?? 0) / 2);
    });
  }
});
