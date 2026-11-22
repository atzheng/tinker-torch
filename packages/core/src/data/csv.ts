/**
 * CSV import (CONTRACTS.md section 6): a small parser, a target/kind inference and the builder that
 * turns chosen columns into the `csv` dataset. Pure text in, plain objects out; never fetches.
 */
import { registerTable, type RealTable } from './real.js';
import { SETS, type SetKind, type SetSpec } from './sets.js';

export const CSV_LIMITS = { rows: 5000, columns: 50, classes: 20, minRows: 10 } as const;

export type CsvColumn = {
  name: string;
  /** Every non-missing cell is a finite number (and there is at least one). */
  numeric: boolean;
  /** Distinct non-missing values. */
  distinct: number;
  missing: number;
};

export type Csv = { header: string[]; rows: string[][]; columns: CsvColumn[] };
export type Fail = { ok: false; error: string };

const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', '?', '-']);
export const isMissing = (cell: string): boolean => MISSING.has(cell.trim().toLowerCase());

const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** A decimal number (not hex, binary or Infinity, which `Number` would accept); NaN otherwise. */
function numberOf(cell: string): number {
  const t = cell.trim();
  return DECIMAL.test(t) ? Number(t) : NaN;
}

/** Splits text into rows of cells: quoted cells may hold the delimiter, quotes ("" inside) and newlines. */
export function parseCsvCells(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;
  let started = false; // anything seen on this row (so a blank line is skipped, `a,` is not)
  const endCell = (): void => {
    row.push(cell);
    cell = '';
  };
  const endRow = (): void => {
    if (started) {
      endCell();
      rows.push(row);
    }
    row = [];
    started = false;
  };
  for (let i = 0; i < text.length; i++) {
    const ch = text[i] as string;
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else quoted = false;
      } else cell += ch;
      continue;
    }
    if (ch === '"' && cell === '') {
      quoted = true;
      started = true;
    } else if (ch === delimiter) {
      started = true;
      endCell();
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      endRow();
    } else {
      cell += ch;
      started = true;
    }
  }
  if (quoted) throw new Error('A quoted value is never closed.');
  endRow();
  return rows;
}

function pickDelimiter(text: string): string {
  const first = text.split(/\r?\n/, 1)[0] ?? '';
  let best = ',';
  let bestCount = 0;
  for (const d of [',', ';', '\t']) {
    let inQ = false;
    let count = 0;
    for (const ch of first) {
      if (ch === '"') inQ = !inQ;
      else if (!inQ && ch === d) count++;
    }
    if (count > bestCount) {
      best = d;
      bestCount = count;
    }
  }
  return best;
}

/** Parses CSV text (first row is the header; delimiter comma, semicolon or tab) and describes its columns. */
export function parseCsv(input: string): { ok: true; csv: Csv } | Fail {
  const text = input.replace(/^\uFEFF/, '');
  if (text.trim() === '') return { ok: false, error: 'The file is empty.' };
  let cells: string[][];
  try {
    cells = parseCsvCells(text, pickDelimiter(text));
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : 'The text is not a valid CSV.' };
  }
  const head = cells[0] ?? [];
  if (head.length < 2) {
    return { ok: false, error: 'This does not look like a CSV: the first line has one column.' };
  }
  if (head.length > CSV_LIMITS.columns) {
    return {
      ok: false,
      error: `The file has ${head.length} columns; at most ${CSV_LIMITS.columns} columns are supported.`,
    };
  }
  const body = cells.slice(1);
  if (body.length === 0) return { ok: false, error: 'The file has a header but no data rows.' };
  if (body.length > CSV_LIMITS.rows) {
    return {
      ok: false,
      error: `The file has ${body.length} rows; at most ${CSV_LIMITS.rows} rows are supported.`,
    };
  }
  const seen = new Map<string, number>();
  const header = head.map((h, i) => {
    const base = h.trim() === '' ? `column ${i + 1}` : h.trim();
    const k = (seen.get(base) ?? 0) + 1;
    seen.set(base, k);
    return k === 1 ? base : `${base} ${k}`;
  });
  for (let r = 0; r < body.length; r++) {
    const row = body[r] as string[];
    // a trailing delimiter on every row leaves empty cells past the header: ignore them
    while (row.length > head.length && (row[row.length - 1] as string).trim() === '') row.pop();
    if (row.length > head.length) {
      return {
        ok: false,
        error: `Row ${r + 2} has ${row.length} values but the header has ${head.length}.`,
      };
    }
    while (row.length < head.length) row.push('');
  }
  const columns: CsvColumn[] = header.map((name, c) => {
    const vals = new Set<string>();
    let missing = 0;
    let numeric = true;
    for (const row of body) {
      const cell = row[c] as string;
      if (isMissing(cell)) {
        missing++;
        continue;
      }
      vals.add(cell.trim());
      if (!Number.isFinite(numberOf(cell))) numeric = false;
    }
    return { name, numeric: numeric && vals.size > 0, distinct: vals.size, missing };
  });
  return { ok: true, csv: { header, rows: body, columns } };
}

/** The kind a target column most likely is (text: classes; numbers: regression, or binary for exactly 0 and 1); null when it cannot be a target (many distinct text values, or empty). */
export function inferKind(csv: Csv, target: number): SetKind | null {
  const col = csv.columns[target];
  if (!col || col.distinct === 0) return null;
  if (col.numeric) {
    // numbers are regression unless they are exactly 0 and 1; the Kind choice overrides
    const zeroOne = csv.rows.every((r) => {
      const c = r[target] as string;
      return isMissing(c) || numberOf(c) === 0 || numberOf(c) === 1;
    });
    return zeroOne && col.distinct === 2 ? 'binary' : 'regression';
  }
  if (col.distinct === 2) return 'binary';
  return col.distinct <= CSV_LIMITS.classes ? 'multiclass' : null;
}

/**
 * How a column can be used as an input: numbers as they are; text with 2 to `CSV_LIMITS.classes`
 * distinct values as a category (stored as the index of its sorted value); anything else (one
 * value, or text with many values such as names or ids) not at all.
 */
export function inputKind(csv: Csv, i: number): 'numeric' | 'categorical' | null {
  const c = csv.columns[i];
  if (!c || c.distinct === 0) return null;
  if (c.numeric) return 'numeric';
  return c.distinct >= 2 && c.distinct <= CSV_LIMITS.classes ? 'categorical' : null;
}

/** Columns other than the target that can be inputs (numeric, or text with few distinct values). */
export function usableInputs(csv: Csv, target: number): number[] {
  return csv.columns.flatMap((_, i) => (i !== target && inputKind(csv, i) !== null ? [i] : []));
}

export type CsvBuild = {
  spec: SetSpec;
  table: RealTable;
  /** Plain one-sentence notes on what was dropped. */
  notes: string[];
};

/**
 * Builds the dataset from the chosen columns. A text input column with few distinct values becomes a
 * categorical column: its categories are the sorted distinct values over the whole file (so the
 * validation rows never hold a category the training rows lack) and its cells are stored as the
 * category's index. Other unusable columns are dropped with a note, as are rows with a missing value
 * in any used column. Classes are the sorted distinct target values.
 */
export function buildCsv(
  csv: Csv,
  opts: { target: number; inputs: number[]; kind: SetKind },
): ({ ok: true } & CsvBuild) | Fail {
  const { target, kind } = opts;
  const tcol = csv.columns[target];
  if (!tcol) return { ok: false, error: 'Choose a target column.' };
  const inputs = opts.inputs.filter((i) => i !== target);
  const notes: string[] = [];
  const left = inputs.filter((i) => inputKind(csv, i) === null);
  const used = inputs.filter((i) => inputKind(csv, i) !== null);
  if (left.length > 0) {
    notes.push(
      `Left out ${left.length} text column${left.length === 1 ? '' : 's'} with too many distinct values: ${left
        .map((i) => csv.header[i])
        .join(', ')}.`,
    );
  }
  if (used.length === 0) return { ok: false, error: 'Choose at least one input column.' };
  if (kind === 'regression' && !tcol.numeric) {
    return { ok: false, error: 'A regression target must be a numeric column.' };
  }

  const kept = csv.rows.filter((r) => ![target, ...used].some((c) => isMissing(r[c] as string)));
  const dropped = csv.rows.length - kept.length;
  if (dropped > 0) {
    notes.push(`Dropped ${dropped} row${dropped === 1 ? '' : 's'} with a missing value.`);
  }
  if (kept.length < CSV_LIMITS.minRows) {
    return {
      ok: false,
      error: `Only ${kept.length} complete rows are left; at least ${CSV_LIMITS.minRows} are needed.`,
    };
  }

  let y: number[];
  let classNames: string[] | undefined;
  if (kind === 'regression') {
    y = kept.map((r) => numberOf(r[target] as string));
  } else {
    const labels = [...new Set(kept.map((r) => (r[target] as string).trim()))];
    if (tcol.numeric) labels.sort((a, b) => Number(a) - Number(b));
    else labels.sort((a, b) => a.localeCompare(b));
    if (kind === 'binary' && labels.length !== 2) {
      return {
        ok: false,
        error: `Two classes are needed for binary, but the target has ${labels.length} distinct values.`,
      };
    }
    if (kind === 'multiclass' && labels.length < 3) {
      return {
        ok: false,
        error: `Multi-class needs at least 3 classes, but the target has ${labels.length}; try binary.`,
      };
    }
    if (labels.length > CSV_LIMITS.classes) {
      return {
        ok: false,
        error: `The target has ${labels.length} distinct values; at most ${CSV_LIMITS.classes} classes are supported.`,
      };
    }
    classNames = labels;
    y = kept.map((r) => labels.indexOf((r[target] as string).trim()));
  }
  // categories: sorted distinct values over every kept row, before any split
  const categories: Record<number, string[]> = {};
  used.forEach((c, j) => {
    if (inputKind(csv, c) !== 'categorical') return;
    categories[j] = [...new Set(kept.map((r) => (r[c] as string).trim()))].sort((a, b) =>
      a.localeCompare(b),
    );
  });
  const x = kept.map((r) =>
    used.map((c, j) => {
      const names = categories[j];
      return names ? names.indexOf((r[c] as string).trim()) : numberOf(r[c] as string);
    }),
  );
  const d = used.length;
  return {
    ok: true,
    notes,
    table: { x, y },
    spec: {
      name: 'csv',
      kind,
      d,
      domain: Array.from({ length: d }, () => [0, 1] as [number, number]),
      ...(kind === 'multiclass' ? { numClasses: classNames?.length ?? 3 } : {}),
      ...(Object.keys(categories).length > 0 ? { categories } : {}),
      featureNames: used.map((c) => csv.header[c] as string),
      targetName: csv.header[target] as string,
      ...(classNames ? { classNames } : {}),
      real: true,
      f: () => NaN,
    },
  };
}

/** Makes a built CSV the `csv` dataset (session only: the rows are not part of a saved file). */
export function registerCsv(b: CsvBuild): void {
  SETS.csv = b.spec;
  registerTable('csv', b.table);
}
