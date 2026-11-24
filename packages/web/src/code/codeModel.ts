/**
 * Pure model of the Code tab (tinker-img.2): from a `TorchCode` and the set of expanded folds to the
 * rows to draw, with each row cut into segments by syntax token, span edge and fold. No React.
 */
import type { NodeId, TorchCode, TorchFold, TorchSpan } from '@tinker/core';
import { tokenizeLine, type TokKind } from './pythonTokens.js';

export type SegKind = TokKind | 'plain' | 'fold' | 'foldLabel';
export type Seg = {
  text: string;
  kind: SegKind;
  /** Offset into `code.text` of the segment's first character (a fold: of its `[`); null for the fold label. */
  from: number | null;
  /** Inside a span of the selected piece. */
  hl: boolean;
  /** A collapsed fold's index (kind `fold`). */
  fold?: number;
};
export type Row = {
  /** 1-based line number of the row's first source line. */
  n: number;
  /** Source offset of the row's line start. */
  start: number;
  /** Source offset of the end of the row's last source line. */
  end: number;
  segs: Seg[];
  hl: boolean;
  /** An open fold starting on this row (a toggle to collapse it). */
  openFold?: number;
};

/** Start offsets of each line of `text`. */
export function lineStarts(text: string): number[] {
  const out = [0];
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) out.push(i + 1);
  return out;
}

/** Index of the line containing `offset` (binary search over `starts`). */
export function lineOf(starts: readonly number[], offset: number): number {
  let lo = 0;
  let hi = starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if ((starts[mid] ?? 0) <= offset) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** The first span of `node` in text order (the one the view scrolls to). */
export function firstSpan(spans: readonly TorchSpan[], node: NodeId): TorchSpan | null {
  let best: TorchSpan | null = null;
  for (const s of spans) if (s.node === node && (best === null || s.from < best.from)) best = s;
  return best;
}

/**
 * The node whose span is the innermost one containing `offset`: the shortest, and on a tie the
 * first in the list (an idiom's covered pieces share the Mean's range, which comes first).
 */
export function nodeAt(spans: readonly TorchSpan[], offset: number): NodeId | null {
  let best: TorchSpan | null = null;
  for (const s of spans) {
    if (offset < s.from || offset >= s.to) continue;
    if (best === null || s.to - s.from < best.to - best.from) best = s;
  }
  return best?.node ?? null;
}

/** The node of the first span (earliest start, then list order) that starts on the line `[ls, le]`; null when none. */
export function nodeOnLine(spans: readonly TorchSpan[], ls: number, le: number): NodeId | null {
  let best: TorchSpan | null = null;
  for (const s of spans) {
    if (s.from >= ls && s.from <= le && (best === null || s.from < best.from)) best = s;
  }
  return best?.node ?? null;
}

/** Rows to draw. `expanded` holds the indices (into `code.folds`) of the folds that are open. */
export function buildRows(
  code: TorchCode,
  expanded: ReadonlySet<number>,
  selected: NodeId | null,
): Row[] {
  const { text, spans, folds } = code;
  const starts = lineStarts(text);
  const endOf = (l: number): number =>
    l + 1 < starts.length ? (starts[l + 1] ?? 0) - 1 : text.length;
  const selSpans = selected === null ? [] : spans.filter((s) => s.node === selected);
  const inSel = (o: number): boolean => selSpans.some((s) => o >= s.from && o < s.to);
  // a fold is keyed by its first line; a collapsed fold hides every line after it up to `last`
  const foldAtLine = new Map<number, { index: number; fold: TorchFold; last: number }>();
  folds.forEach((fold, index) => {
    foldAtLine.set(lineOf(starts, fold.from), {
      index,
      fold,
      last: lineOf(starts, fold.to - 1),
    });
  });

  // Source range [a, b) of line `l`, cut at token and span edges.
  const segRange = (a: number, b: number, l: number): Seg[] => {
    if (b <= a) return [];
    const ls = starts[l] ?? 0;
    const cuts = new Set<number>([a, b]);
    const toks = tokenizeLine(text.slice(ls, endOf(l))).map((t) => ({
      from: t.from + ls,
      to: t.to + ls,
      kind: t.kind,
    }));
    for (const t of toks) for (const c of [t.from, t.to]) if (c > a && c < b) cuts.add(c);
    for (const s of spans) for (const c of [s.from, s.to]) if (c > a && c < b) cuts.add(c);
    const pts = [...cuts].sort((x, y) => x - y);
    const out: Seg[] = [];
    for (let i = 0; i + 1 < pts.length; i++) {
      const from = pts[i] ?? 0;
      const to = pts[i + 1] ?? 0;
      const tok = toks.find((t) => from >= t.from && from < t.to);
      out.push({ text: text.slice(from, to), kind: tok?.kind ?? 'plain', from, hl: inSel(from) });
    }
    return out;
  };

  const rows: Row[] = [];
  for (let l = 0; l < starts.length; l++) {
    const ls = starts[l] ?? 0;
    const f = foldAtLine.get(l);
    if (f && !expanded.has(f.index)) {
      const segs: Seg[] = [
        ...segRange(ls, f.fold.from, l),
        { text: '[...]', kind: 'fold', from: f.fold.from, hl: false, fold: f.index },
        ...segRange(f.fold.to, endOf(f.last), f.last),
        { text: `  # ${f.fold.label}`, kind: 'foldLabel', from: null, hl: false },
      ];
      rows.push({ n: l + 1, start: ls, end: endOf(f.last), segs, hl: segs.some((s) => s.hl) });
      l = f.last;
      continue;
    }
    let segs = segRange(ls, endOf(l), l);
    if (segs.length === 0) segs = [{ text: '', kind: 'plain', from: ls, hl: false }];
    const row: Row = { n: l + 1, start: ls, end: endOf(l), segs, hl: segs.some((s) => s.hl) };
    if (f) row.openFold = f.index;
    rows.push(row);
  }
  return rows;
}
