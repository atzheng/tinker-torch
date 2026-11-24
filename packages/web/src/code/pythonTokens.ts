/**
 * A small hand-written Python tokenizer for the Code tab (tinker-img.2). It only has to colour the
 * script `torchScript` writes (one statement per line, no multi-line strings), so it works on one
 * line at a time and knows keywords, builtins, numbers, strings and comments. No dependency.
 */
export type TokKind = 'keyword' | 'builtin' | 'number' | 'string' | 'comment' | 'self';
export type Token = { from: number; to: number; kind: TokKind };

const KEYWORDS: ReadonlySet<string> = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
]);
const BUILTINS: ReadonlySet<string> = new Set([
  'abs',
  'dict',
  'enumerate',
  'float',
  'int',
  'iter',
  'len',
  'list',
  'max',
  'min',
  'next',
  'print',
  'range',
  'set',
  'sorted',
  'str',
  'sum',
  'super',
  'tuple',
  'zip',
]);

const WORD = /[A-Za-z_][A-Za-z0-9_]*/y;
const NUMBER = /(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?/y;

/** End (exclusive) of the string literal whose opening quote is at `open`. */
function stringEnd(line: string, open: number): number {
  const q = line.charAt(open);
  let j = open + 1;
  while (j < line.length && line.charAt(j) !== q) j += line.charAt(j) === '\\' ? 2 : 1;
  return Math.min(j + 1, line.length);
}

/** Tokens of one line (offsets into `line`), in order. Plain text (names, punctuation) has no token. */
export function tokenizeLine(line: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  while (i < line.length) {
    const c = line.charAt(i);
    if (c === '#') {
      out.push({ from: i, to: line.length, kind: 'comment' });
      break;
    }
    if (c === '"' || c === "'") {
      const j = stringEnd(line, i);
      out.push({ from: i, to: j, kind: 'string' });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      WORD.lastIndex = i;
      const w = WORD.exec(line)?.[0] ?? c;
      const q = line.charAt(i + w.length);
      // a string prefix (f"...", r'...') belongs to the string
      if (w.length <= 2 && /^[rbfuRBFU]+$/.test(w) && (q === '"' || q === "'")) {
        const j = stringEnd(line, i + w.length);
        out.push({ from: i, to: j, kind: 'string' });
        i = j;
        continue;
      }
      const kind: TokKind | null =
        line.charAt(i - 1) === '.'
          ? null
          : w === 'self'
            ? 'self'
            : KEYWORDS.has(w)
              ? 'keyword'
              : BUILTINS.has(w)
                ? 'builtin'
                : null;
      if (kind) out.push({ from: i, to: i + w.length, kind });
      i += w.length;
      continue;
    }
    if (/\d/.test(c) || (c === '.' && /\d/.test(line.charAt(i + 1)))) {
      NUMBER.lastIndex = i;
      const m = NUMBER.exec(line)?.[0] ?? c;
      out.push({ from: i, to: i + m.length, kind: 'number' });
      i += m.length;
      continue;
    }
    i += 1;
  }
  return out;
}
