import { describe, expect, test } from 'vitest';
import type { TorchCode } from '@tinker/core';
import { buildRows, firstSpan, nodeAt, nodeOnLine } from '../src/code/codeModel.js';
import { tokenizeLine } from '../src/code/pythonTokens.js';

const text = [
  'import torch',
  'x = torch.tensor([',
  '  [1.5],',
  '  [2e-3],',
  '], dtype=torch.float32)',
  'a = F.mse_loss(p, y)  # note',
].join('\n');
const fi = text.indexOf('[\n');
const fe = text.indexOf('], dtype') + 1;
const stmt = text.indexOf('a = F');
const code: TorchCode = {
  text,
  spans: [
    { node: 'm', from: stmt, to: text.indexOf('  # note') },
    { node: 's', from: stmt, to: text.indexOf('  # note') },
    { node: 'p', from: text.indexOf('p, y'), to: text.indexOf('p, y') + 1 },
  ],
  folds: [{ from: fi, to: fe, label: '2 rows' }],
};

describe('tokenizer', () => {
  test('keywords, builtins, numbers, strings, comments', () => {
    const t = tokenizeLine('for i in range(3): print("a#b", 1.5e-3)  # c');
    expect(t.map((x) => x.kind)).toEqual([
      'keyword',
      'keyword',
      'builtin',
      'number',
      'builtin',
      'string',
      'number',
      'comment',
    ]);
  });
  test('attribute names are not keywords; f-string prefix is part of the string', () => {
    expect(tokenizeLine('x.in = f"a{b}"').map((x) => x.kind)).toEqual(['string']);
    expect(tokenizeLine('self.x').map((x) => x.kind)).toEqual(['self']);
  });
});

describe('spans', () => {
  test('innermost span wins; ties go to the first in the list', () => {
    expect(nodeAt(code.spans, stmt)).toBe('m');
    expect(nodeAt(code.spans, text.indexOf('p, y'))).toBe('p');
    expect(nodeAt(code.spans, 0)).toBeNull();
  });
  test('first span and line fallback', () => {
    expect(firstSpan(code.spans, 's')?.from).toBe(stmt);
    expect(nodeOnLine(code.spans, stmt, text.length)).toBe('m');
    expect(nodeOnLine(code.spans, 0, 10)).toBeNull();
  });
});

describe('rows', () => {
  test('a collapsed mid-line fold becomes [...] plus the label; later lines are hidden', () => {
    const rows = buildRows(code, new Set(), null);
    expect(rows.map((r) => r.n)).toEqual([1, 2, 6]);
    expect(rows[1]?.segs.map((s) => s.text).join('')).toBe(
      'x = torch.tensor([...], dtype=torch.float32)  # 2 rows',
    );
  });
  test('an expanded fold shows every line and marks the first', () => {
    const rows = buildRows(code, new Set([0]), null);
    expect(rows.map((r) => r.n)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(rows[1]?.openFold).toBe(0);
    expect(rows.map((r) => r.segs.map((s) => s.text).join('')).join('\n')).toBe(text);
  });
  test('the selected piece highlights its spans', () => {
    const rows = buildRows(code, new Set(), 'm');
    const last = rows[2];
    expect(last?.hl).toBe(true);
    expect(
      last?.segs
        .filter((s) => s.hl)
        .map((s) => s.text)
        .join(''),
    ).toBe('a = F.mse_loss(p, y)');
    expect(rows[0]?.hl).toBe(false);
  });
});
