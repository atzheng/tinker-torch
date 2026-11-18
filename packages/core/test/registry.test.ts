import { describe, expect, test } from 'vitest';
import { CORE_PIECES, NODE_TYPES, PALETTE, featureWidth, type NodeOf } from '../src/index.js';

const ctx = { dataColumns: 2, numClasses: 2 };

describe('CORE_PIECES', () => {
  test('has an entry for every piece type', () => {
    expect(Object.keys(CORE_PIECES).sort()).toEqual([...NODE_TYPES].sort());
    for (const t of NODE_TYPES) expect(CORE_PIECES[t].defaults()).toMatchObject({ type: t });
  });

  test('input ports follow section 2', () => {
    expect(CORE_PIECES.add.inputPorts).toEqual(['a', 'b']);
    expect(CORE_PIECES.loss.inputPorts).toEqual(['pred', 'target']);
    expect(CORE_PIECES.input.inputPorts).toEqual([]);
    expect(CORE_PIECES.target.inputPorts).toEqual([]);
    expect(CORE_PIECES.linear.inputPorts).toEqual(['in']);
  });

  test('input width is the number of selected columns', () => {
    const n: NodeOf<'input'> = { id: 'x', type: 'input', columns: ['x1', 'x2'] };
    expect(CORE_PIECES.input.outWidth(n, {}, ctx)).toBe(2);
    expect(CORE_PIECES.input.outWidth({ ...n, columns: [] }, {}, ctx)).toHaveProperty('error');
  });

  test('feature width formula for d = 1 and 2', () => {
    const f = (degree: number, sin: boolean, cos: boolean, products: boolean) =>
      ({ id: 'f', type: 'feature', degree, sin, cos, products }) as const;
    const w = (n: ReturnType<typeof f>, d: number) =>
      CORE_PIECES.feature.outWidth(n, { in: d }, ctx);
    expect(w(f(3, false, false, false), 1)).toBe(3);
    expect(w(f(1, true, true, true), 1)).toBe(3);
    expect(w(f(2, false, false, false), 2)).toBe(4);
    expect(w(f(2, true, true, true), 2)).toBe(4 + 2 + 2 + 1);
    expect(featureWidth(2, { degree: 5, sin: false, cos: false, products: true })).toBe(11);
    expect(CORE_PIECES.feature.outWidth(f(2, false, false, false), {}, ctx)).toHaveProperty(
      'error',
    );
  });

  test('linear width, shapes and torch', () => {
    const n = { id: 'l', type: 'linear', out: 8, bias: true } as const;
    expect(CORE_PIECES.linear.outWidth(n, { in: 4 }, ctx)).toBe(8);
    expect(CORE_PIECES.linear.outWidth(n, {}, ctx)).toEqual({ error: 'Linear needs an input.' });
    expect(CORE_PIECES.linear.paramShapes(n, 4)).toEqual({ W: [4, 8], b: [8] });
    expect(CORE_PIECES.linear.paramShapes({ ...n, bias: false }, 4)).toEqual({ W: [4, 8] });
    expect(CORE_PIECES.linear.torch(n, 4)).toBe('nn.Linear(4, 8, bias=True)');
  });

  test('activation, dropout and output pass the width through', () => {
    const act = { id: 'a', type: 'activation', fn: 'relu' } as const;
    const drop = { id: 'd', type: 'dropout', p: 0.1 } as const;
    const out = { id: 'o', type: 'output', link: 'softmax' } as const;
    expect(CORE_PIECES.activation.outWidth(act, { in: 5 }, ctx)).toBe(5);
    expect(CORE_PIECES.dropout.outWidth(drop, { in: 5 }, ctx)).toBe(5);
    expect(CORE_PIECES.output.outWidth(out, { in: 2 }, ctx)).toBe(2);
    expect(CORE_PIECES.activation.outWidth(act, {}, ctx)).toHaveProperty('error');
  });

  test('add: shared width, mismatch sentence, missing input', () => {
    const n = { id: 'a', type: 'add' } as const;
    expect(CORE_PIECES.add.outWidth(n, { a: 4, b: 4 }, ctx)).toBe(4);
    expect(CORE_PIECES.add.outWidth(n, { a: 4, b: 8 }, ctx)).toEqual({
      error: 'Add needs both inputs to have the same width: 4 and 8.',
    });
    expect(CORE_PIECES.add.outWidth(n, { a: 4 }, ctx)).toHaveProperty('error');
  });

  test('target is 1 wide and loss has no output', () => {
    expect(CORE_PIECES.target.outWidth({ id: 'y', type: 'target' }, {}, ctx)).toBe(1);
    const loss = { id: 'l', type: 'loss', fn: 'huber' } as const;
    expect(CORE_PIECES.loss.outWidth(loss, { pred: 1, target: 1 }, ctx)).toBe(0);
  });

  test('only linear has parameters', () => {
    expect(
      CORE_PIECES.activation.paramShapes({ id: 'a', type: 'activation', fn: 'tanh' }, 3),
    ).toEqual({});
    expect(CORE_PIECES.dropout.paramShapes({ id: 'd', type: 'dropout', p: 0.5 }, 3)).toEqual({});
    expect(CORE_PIECES.add.paramShapes({ id: 'a', type: 'add' }, 3)).toEqual({});
  });

  test('torch strings', () => {
    expect(CORE_PIECES.activation.torch({ id: 'a', type: 'activation', fn: 'gelu' })).toBe(
      'nn.GELU()',
    );
    expect(CORE_PIECES.dropout.torch({ id: 'd', type: 'dropout', p: 0.2 })).toBe(
      'nn.Dropout(p=0.2)',
    );
    expect(CORE_PIECES.loss.torch({ id: 'l', type: 'loss', fn: 'log_loss' })).toBe(
      'nn.BCEWithLogitsLoss()',
    );
    expect(CORE_PIECES.loss.torch({ id: 'l', type: 'loss', fn: 'huber', delta: 1 })).toBe(
      'nn.HuberLoss(delta=1.0)',
    );
    expect(CORE_PIECES.loss.torch({ id: 'l', type: 'loss', fn: 'cross_entropy' })).toBe(
      'nn.CrossEntropyLoss()',
    );
    expect(CORE_PIECES.output.torch({ id: 'o', type: 'output', link: 'sigmoid' })).toBe(
      'nn.Sigmoid()',
    );
    expect(
      CORE_PIECES.feature.torch(
        { id: 'f', type: 'feature', degree: 2, sin: true, cos: false, products: true },
        2,
      ),
    ).toBe('torch.cat([x, x**2, torch.sin(x), x[:, [0]] * x[:, [1]]], dim=-1)');
    expect(
      CORE_PIECES.feature.torch({
        id: 'f',
        type: 'feature',
        degree: 1,
        sin: false,
        cos: false,
        products: true,
      }),
    ).toBe('torch.cat([x, x[:, [?]] * x[:, [?]]], dim=-1)');
    expect(CORE_PIECES.input.torch({ id: 'x', type: 'input', columns: ['x1', 'x2'] })).toBe(
      'x[:, [0, 1]]',
    );
  });
});

describe('PALETTE', () => {
  test('has 14 entries in SPEC order with unique keys', () => {
    expect(PALETTE).toHaveLength(14);
    expect(new Set(PALETTE.map((p) => p.key)).size).toBe(14);
    expect(PALETTE.map((p) => p.label)).toEqual([
      'Input',
      'Feature',
      'Linear',
      'Activation',
      'Dropout',
      'Add',
      'Output',
      'Target',
      'Squared error',
      'Absolute error',
      'Huber',
      'Log loss',
      'Cross-entropy',
      'Weight penalty',
    ]);
  });

  test('every entry makes something and the penalty makes a Penalty', () => {
    for (const p of PALETTE) expect(p.make()).toBeTruthy();
    expect(PALETTE.at(-1)?.family).toBe('penalties');
    expect(PALETTE.at(-1)?.make()).toEqual({ kind: 'l2', lambda: 0.01 });
  });
});
