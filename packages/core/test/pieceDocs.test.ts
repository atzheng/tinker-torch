/** Piece reference cards (CONTRACTS.md section 15, tinker-32v). */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  CORE_PIECES,
  docView,
  docViewFor,
  PALETTE,
  paletteKeyOf,
  PIECE_DOCS,
  type DataInfo,
  type Edge,
  type Graph,
  type Node,
} from '../src/index.js';

const info = (d: number): DataInfo => ({
  columns: Array.from({ length: d }, (_, i) => `x${i + 1}`),
  kind: 'regression',
  numClasses: 2,
});

function g(nodes: Node[], edges: Edge[], predictions: string[], loss: string | null): Graph {
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

const VOICE = /\b(you|your|should|try|helps)\b/i;

/** Every string of a doc that is prose (not TeX). */
function prose(key: string): string[] {
  const d = PIECE_DOCS[key];
  if (d === undefined) return [];
  return [
    d.summary,
    ...Object.values(d.settings ?? {}),
    ...Object.values(d.variants ?? {}).flatMap((v) => (v.summary === undefined ? [] : [v.summary])),
    ...(d.notes ?? []),
  ];
}

describe('PIECE_DOCS', () => {
  test('every palette key has a doc and there are no extras', () => {
    expect(Object.keys(PIECE_DOCS).sort()).toEqual(PALETTE.map((e) => e.key).sort());
  });

  test('settings keys are fields of the piece defaults', () => {
    for (const e of PALETTE) {
      if (e.kind !== 'piece') continue;
      const fields = Object.keys(CORE_PIECES[e.make().type].defaults());
      for (const k of Object.keys(PIECE_DOCS[e.key]?.settings ?? {})) {
        expect(fields, `${e.key}.${k}`).toContain(k);
      }
    }
  });

  test('seeAlso keys exist', () => {
    const keys = new Set(PALETTE.map((e) => e.key));
    for (const [key, d] of Object.entries(PIECE_DOCS)) {
      for (const k of d.seeAlso ?? []) expect(keys.has(k), `${key} -> ${k}`).toBe(true);
    }
  });

  test('length caps', () => {
    for (const [key, d] of Object.entries(PIECE_DOCS)) {
      expect(d.summary.length, key).toBeLessThanOrEqual(100);
      expect(d.notes?.length ?? 0, key).toBeLessThanOrEqual(3);
      for (const s of prose(key)) expect(s.length, `${key}: ${s}`).toBeGreaterThan(0);
    }
  });

  test('notes and settings are one sentence each', () => {
    for (const key of Object.keys(PIECE_DOCS)) {
      for (const s of [
        ...(PIECE_DOCS[key]?.notes ?? []),
        ...Object.values(PIECE_DOCS[key]?.settings ?? {}),
      ]) {
        expect(s.trim().endsWith('.'), `${key}: ${s}`).toBe(true);
        // no sentence break inside (a '. ' followed by a capital letter)
        expect(/\.\s+[A-Z]/.test(s), `${key}: ${s}`).toBe(false);
      }
    }
  });

  test('reference voice: no you, your, should, try, helps', () => {
    for (const key of Object.keys(PIECE_DOCS)) {
      for (const s of prose(key)) expect(VOICE.test(s), `${key}: ${s}`).toBe(false);
    }
  });

  test('variants cover the activation functions', () => {
    expect(Object.keys(PIECE_DOCS['activation']?.variants ?? {}).sort()).toEqual([
      'gelu',
      'relu',
      'sigmoid',
      'tanh',
    ]);
  });
});

describe('paletteKeyOf', () => {
  test('link by fn, norm by kind, else the type', () => {
    expect(paletteKeyOf({ id: 'a', type: 'link', fn: 'softmax' })).toBe('softmax');
    expect(paletteKeyOf({ id: 'a', type: 'link', fn: 'sigmoid' })).toBe('sigmoid');
    expect(paletteKeyOf({ id: 'a', type: 'norm', kind: 'l1' })).toBe('l1');
    expect(paletteKeyOf({ id: 'a', type: 'norm', kind: 'l2' })).toBe('l2');
    expect(paletteKeyOf({ id: 'a', type: 'linear', out: 2, bias: true })).toBe('linear');
  });

  test('every piece entry key maps back from its made node', () => {
    for (const e of PALETTE) {
      if (e.kind !== 'piece') continue;
      expect(paletteKeyOf({ ...e.make(), id: 'n' } as Node)).toBe(e.key);
    }
  });
});

describe('docView (symbolic)', () => {
  test('Linear', () => {
    const v = docView('linear');
    expect(v.label).toBe('Linear');
    expect(v.params).toBe('d \\cdot m + m');
    expect(v.shapes).toEqual({ in: { in: 'n \\times d' }, out: 'n \\times m' });
    expect(v.torch).toBe('nn.Linear(d, m, bias=True)');
    expect(v.settings.map((s) => s.field)).toEqual(['out', 'bias']);
    expect(v.seeAlso).toContainEqual({ key: 'activation', label: 'Activation' });
  });

  test('every key resolves a view with a torch line, and no sentinel leaks', () => {
    for (const e of PALETTE) {
      const v = docView(e.key);
      expect(v.key).toBe(e.key);
      expect(v.torch, e.key).toBeTruthy();
      expect(v.equation, e.key).toBeTruthy();
      expect(JSON.stringify(v), e.key).not.toMatch(/\d{6,}/);
    }
  });

  test('pieces without parameters have no params; scalar pieces and Lookup have them', () => {
    expect(docView('relu' as string).params).toBeUndefined();
    expect(docView('activation').params).toBeUndefined();
    expect(docView('scalew').params).toBe('1');
    expect(docView('bias').params).toBe('1');
    expect(docView('lookup').params).toBe('K \\cdot m');
    expect(docView('mean').shapes?.out).toBe('1');
    expect(docView('weights').shapes?.out).toBe('d \\times m');
  });

  test('macros use the section 14 idiom', () => {
    expect(docView('loss_squared_error').torch).toBe('F.mse_loss(y_hat, y)');
    expect(docView('loss_log_loss').torch).toContain('binary_cross_entropy_with_logits');
    expect(docView('weight_penalty').torch).toContain('.pow(2).sum()');
  });

  test('an unknown key does not throw', () => {
    const v = docView('nope');
    expect(v.label).toBe('nope');
    expect(v.settings).toEqual([]);
  });
});

describe('docViewFor (live)', () => {
  const lin = (out: number, bias: boolean, cols = 3) => {
    const graph = g(
      [
        { id: 'in', type: 'input', columns: info(cols).columns },
        { id: 'l', type: 'linear', out, bias },
      ],
      [['in', 'l']],
      ['l'],
      null,
    );
    return { graph, analysis: analyse(graph, info(cols)) };
  };

  test('Linear 3 -> 8 with bias', () => {
    const { graph, analysis } = lin(8, true);
    const v = docViewFor(graph.nodes[1] as Node, analysis, info(3));
    expect(v.params).toBe('3 \\cdot 8 + 8 = 32');
    expect(v.torch).toBe('nn.Linear(3, 8, bias=True)');
    expect(v.shapes).toEqual({ in: { in: 'n \\times 3' }, out: 'n \\times 8' });
    expect(v.key).toBe('linear');
  });

  test('Linear without bias', () => {
    const { graph, analysis } = lin(8, false);
    const v = docViewFor(graph.nodes[1] as Node, analysis, info(3));
    expect(v.params).toBe('3 \\cdot 8 = 24');
    expect(v.torch).toBe('nn.Linear(3, 8, bias=False)');
  });

  test('activation uses variants[fn]', () => {
    const graph = g(
      [
        { id: 'in', type: 'input', columns: ['x1'] },
        { id: 'a', type: 'activation', fn: 'tanh' },
      ],
      [['in', 'a']],
      ['a'],
      null,
    );
    const v = docViewFor(graph.nodes[1] as Node, analyse(graph, info(1)), info(1));
    expect(v.equation).toBe(PIECE_DOCS['activation']?.variants?.['tanh']?.equation);
    expect(v.summary).toBe(PIECE_DOCS['activation']?.variants?.['tanh']?.summary);
    expect(v.torch).toBe('torch.tanh(in_)');
  });

  test('unknown widths fall back to symbols', () => {
    const graph = g([{ id: 'l', type: 'linear', out: 8, bias: true }], [], [], null);
    const v = docViewFor(graph.nodes[0] as Node, analyse(graph, info(3)), info(3));
    expect(v.params).toBe('d \\cdot 8 + 8');
    expect(v.torch).toBe('nn.Linear(d, 8, bias=True)');
  });

  test('never throws on broken graphs', () => {
    const nodes: Node[] = [
      { id: 'in', type: 'input', columns: [] },
      { id: 'c', type: 'column', name: 'nope' },
      { id: 'l', type: 'linear', out: Number.NaN, bias: true },
      { id: 'w', type: 'weights', of: 'missing' },
      { id: 'n', type: 'norm', kind: 'l2' },
      { id: 'oh', type: 'onehot' },
      { id: 'lk', type: 'lookup', width: 2 },
      {
        id: 't',
        type: 'tree',
        criterion: 'gini',
        depth: 3,
        minSplit: 2,
        minLeaf: 1,
        minDecrease: 0,
      },
      { id: 'a', type: 'add' },
      { id: 'a2', type: 'add' },
      { id: 'ce', type: 'cross_entropy' },
      { id: 'm', type: 'mean' },
    ];
    const edges: Edge[] = [
      ['a', 'a2.a'],
      ['a2', 'a.a'],
      ['w', 'n'],
      ['in', 'l'],
      ['c', 'oh'],
      ['in', 't'],
    ];
    const graph = g(nodes, edges, ['l', 'ghost'], 'm');
    const empty = analyse(graph, info(0));
    const normal = analyse(graph, info(3));
    for (const analysis of [empty, normal, {} as unknown as typeof normal]) {
      for (const n of nodes) {
        for (const data of [info(0), info(3)]) {
          const v = docViewFor(n, analysis, data);
          expect(v.key).toBe(paletteKeyOf(n));
          expect(typeof v.summary).toBe('string');
        }
      }
    }
  });
});
