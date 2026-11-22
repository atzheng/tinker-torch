/**
 * The CSV dataset inside a saved file or preset (CONTRACTS.md sections 6 and 12). The file holds
 * the BUILT dataset (the used columns as numbers, categories as indices, the target as a number or
 * class index), not the raw CSV text: it is smaller, parses without guessing, and reopens exactly
 * as the dialog built it (same target, inputs, kind, categories and classes). Pure; never fetches.
 */
import { getTable, registerTable } from './real.js';
import { SETS, type SetKind } from './sets.js';
import { CSV_LIMITS, type CsvBuild } from './csv.js';

export type EmbeddedCsv = {
  /** File name or address the rows came from (a label only). */
  name?: string;
  kind: SetKind;
  /** Display names of the input columns, in order. */
  features: string[];
  target: string;
  /** Classification only: class names in label order. */
  classes?: string[];
  /** Categorical input columns: column index (as text) -> category names; the cell is the index. */
  categories?: Record<string, string[]>;
  /** One row per example, one number per feature. */
  x: number[][];
  /** Regression: the target; classification: the class index. */
  y: number[];
};

const isRec = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const isTexts = (v: unknown): v is string[] =>
  Array.isArray(v) && v.every((s) => typeof s === 'string');

/** The registered `csv` dataset as an embeddable object, or null when none is open. */
export function embedCsv(name?: string | null): EmbeddedCsv | null {
  const spec = SETS.csv;
  const table = getTable('csv');
  if (!table || !spec.featureNames) return null;
  const out: EmbeddedCsv = {
    kind: spec.kind,
    features: spec.featureNames,
    target: spec.targetName ?? 'y',
    x: table.x,
    y: table.y,
  };
  if (name) out.name = name;
  if (spec.kind !== 'regression' && spec.classNames) out.classes = spec.classNames;
  if (spec.categories) {
    const cats: Record<string, string[]> = {};
    for (const [k, names] of Object.entries(spec.categories)) cats[k] = names;
    out.categories = cats;
  }
  return out;
}

/** Validates the `csv` field of a file: the object, or one plain sentence. */
export function parseEmbeddedCsv(v: unknown): EmbeddedCsv | string {
  const bad = 'The data saved in this file is not a valid CSV table';
  if (!isRec(v)) return `${bad}.`;
  const { name, kind, features, target, classes, categories, x, y } = v;
  if (name !== undefined && (typeof name !== 'string' || name.length > 300)) {
    return `${bad}: the name must be short text.`;
  }
  if (kind !== 'regression' && kind !== 'binary' && kind !== 'multiclass') {
    return `${bad}: the kind must be regression, binary or multiclass.`;
  }
  if (!isTexts(features) || features.length < 1) {
    return `${bad}: it needs a list of input column names.`;
  }
  if (features.length > CSV_LIMITS.columns) {
    return `The data saved in this file has ${features.length} input columns; at most ${CSV_LIMITS.columns} columns are supported.`;
  }
  if (typeof target !== 'string') return `${bad}: the target needs a name.`;
  if (!Array.isArray(x) || !Array.isArray(y)) return `${bad}: it needs rows "x" and targets "y".`;
  if (x.length > CSV_LIMITS.rows) {
    return `The data saved in this file has ${x.length} rows; at most ${CSV_LIMITS.rows} rows are supported.`;
  }
  if (x.length < CSV_LIMITS.minRows) {
    return `The data saved in this file has ${x.length} rows; at least ${CSV_LIMITS.minRows} are needed.`;
  }
  if (y.length !== x.length) {
    return `${bad}: there are ${x.length} rows but ${y.length} targets.`;
  }
  const d = features.length;
  for (let r = 0; r < x.length; r++) {
    const row: unknown = x[r];
    if (!Array.isArray(row) || row.length !== d) {
      return `Row ${r + 1} of the data saved in this file does not have ${d} values.`;
    }
    if (!row.every((c) => typeof c === 'number' && Number.isFinite(c))) {
      return `Row ${r + 1} of the data saved in this file has a value that is not a finite number.`;
    }
  }
  if (!y.every((c) => typeof c === 'number' && Number.isFinite(c))) {
    return `${bad}: a target is not a finite number.`;
  }
  const out: EmbeddedCsv = { kind, features, target, x: x as number[][], y: y as number[] };
  if (name !== undefined) out.name = name as string;
  if (kind === 'regression') {
    if (classes !== undefined) return `${bad}: a regression has no classes.`;
  } else {
    const k = isTexts(classes) ? classes.length : -1;
    if (k < 0 || (kind === 'binary' ? k !== 2 : k < 3 || k > CSV_LIMITS.classes)) {
      return `${bad}: ${kind === 'binary' ? 'a binary target needs exactly 2' : `a multi-class target needs 3 to ${CSV_LIMITS.classes}`} class names.`;
    }
    const names = classes as string[];
    if (!(y as number[]).every((c) => Number.isInteger(c) && c >= 0 && c < names.length)) {
      return `${bad}: a target is not a class number.`;
    }
    out.classes = names;
  }
  if (categories !== undefined) {
    if (!isRec(categories)) return `${bad}: the categories must be an object.`;
    const cats: Record<string, string[]> = {};
    for (const [key, names] of Object.entries(categories)) {
      const j = Number(key);
      if (!Number.isInteger(j) || j < 0 || j >= d || String(j) !== key) {
        return `${bad}: the categories name a column that is not there.`;
      }
      if (!isTexts(names) || names.length < 2 || names.length > CSV_LIMITS.classes) {
        return `${bad}: each categorical column needs 2 to ${CSV_LIMITS.classes} category names.`;
      }
      for (const row of x as number[][]) {
        const c = row[j] as number;
        if (!Number.isInteger(c) || c < 0 || c >= names.length) {
          return `${bad}: a categorical value is not a category number.`;
        }
      }
      cats[key] = names;
    }
    if (Object.keys(cats).length > 0) out.categories = cats;
  }
  return out;
}

/** The dataset a validated embedded CSV stands for (the same spec `buildCsv` makes). */
export function embeddedToBuild(e: EmbeddedCsv): CsvBuild {
  const d = e.features.length;
  const categories: Record<number, string[]> = {};
  for (const [k, names] of Object.entries(e.categories ?? {})) categories[Number(k)] = names;
  return {
    notes: [],
    table: { x: e.x, y: e.y },
    spec: {
      name: 'csv',
      kind: e.kind,
      d,
      domain: Array.from({ length: d }, () => [0, 1] as [number, number]),
      ...(e.kind === 'multiclass' ? { numClasses: e.classes?.length ?? 3 } : {}),
      ...(Object.keys(categories).length > 0 ? { categories } : {}),
      featureNames: e.features,
      targetName: e.target,
      ...(e.classes ? { classNames: e.classes } : {}),
      real: true,
      f: () => NaN,
    },
  };
}

/** Makes an embedded CSV the `csv` dataset (session only, like `registerCsv`). */
export function registerEmbeddedCsv(e: EmbeddedCsv): void {
  const b = embeddedToBuild(e);
  SETS.csv = b.spec;
  registerTable('csv', b.table);
}
