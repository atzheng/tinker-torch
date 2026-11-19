/**
 * Real datasets (CONTRACTS.md section 6): the bundled classic sets and the one user-opened CSV.
 * Their rows are fixed; the metadata (names, kind, classes) is static here so analysis and the
 * Data menu know them before the rows arrive, and the rows (cleaned JSON under `data/real/`) load
 * on demand with `loadRealSet`. Nothing here touches the network: the JSON is bundled.
 */
import type { DatasetName } from '../common.js';
import type { Rng } from '../rng.js';
import type { SetKind, SetSpec } from './sets.js';

export type RealTable = {
  /** Row-major: x[i] is one row of raw feature values. */
  x: number[][];
  /** Regression: the target value; classification: the class index 0..k-1. */
  y: number[];
};

export type RealMeta = {
  /** Label in the Data menu. */
  title: string;
  kind: SetKind;
  featureNames: string[];
  targetName: string;
  classNames?: string[];
  /** Number of rows in the bundled file. */
  rows: number;
  /** One short line on what was done to the source (cleaning). */
  note: string;
  source: string;
  url: string;
  licence: string;
};

/** The bundled sets, in menu order. */
export const BUNDLED = ['iris', 'penguins', 'wine', 'diabetes'] as const;
export type BundledName = (typeof BUNDLED)[number];

export const REAL_META: Record<BundledName, RealMeta> = {
  iris: {
    title: 'Iris',
    kind: 'multiclass',
    featureNames: ['sepal length', 'sepal width', 'petal length', 'petal width'],
    targetName: 'species',
    classNames: ['setosa', 'versicolor', 'virginica'],
    rows: 150,
    note: 'Lengths in cm. As published, no rows dropped.',
    source: 'R. A. Fisher (1936), UCI Machine Learning Repository',
    url: 'https://archive.ics.uci.edu/dataset/53/iris',
    licence: 'CC BY 4.0',
  },
  penguins: {
    title: 'Palmer penguins',
    kind: 'multiclass',
    featureNames: ['bill length', 'bill depth', 'flipper length', 'body mass'],
    targetName: 'species',
    classNames: ['Adelie', 'Chinstrap', 'Gentoo'],
    rows: 342,
    note: 'Bill and flipper in mm, mass in g. Rows with a missing value dropped (344 to 342); island, sex and year left out.',
    source: 'Horst, Hill and Gorman (2020), palmerpenguins; data by K. Gorman, Palmer Station LTER',
    url: 'https://allisonhorst.github.io/palmerpenguins/',
    licence: 'CC0 1.0',
  },
  wine: {
    title: 'Wine',
    kind: 'multiclass',
    featureNames: [
      'alcohol',
      'malic acid',
      'ash',
      'alcalinity of ash',
      'magnesium',
      'total phenols',
      'flavanoids',
      'nonflavanoid phenols',
      'proanthocyanins',
      'color intensity',
      'hue',
      'OD280/OD315',
      'proline',
    ],
    targetName: 'cultivar',
    classNames: ['cultivar 1', 'cultivar 2', 'cultivar 3'],
    rows: 178,
    note: 'As published, no rows dropped. The columns have very different scales: try Standardise.',
    source: 'M. Forina et al., PARVUS, UCI Machine Learning Repository',
    url: 'https://archive.ics.uci.edu/dataset/109/wine',
    licence: 'CC BY 4.0',
  },
  diabetes: {
    title: 'Diabetes progression',
    kind: 'regression',
    featureNames: ['age', 'sex', 'BMI', 'blood pressure', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    targetName: 'progression',
    rows: 442,
    note: 'Raw (unscaled) values from the original table; the target is disease progression one year after baseline. S1 to S6 are blood serum measurements.',
    source:
      'Efron, Hastie, Johnstone and Tibshirani (2004), Least Angle Regression; the table behind scikit-learn load_diabetes',
    url: 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html',
    licence: 'No licence stated by the authors; scikit-learn (BSD-3-Clause) redistributes it',
  },
};

export function isBundled(set: DatasetName): set is BundledName {
  return (BUNDLED as readonly string[]).includes(set);
}

/** True for the sets whose rows are fixed (bundled and the CSV), false for the generators. */
export function isRealSet(set: DatasetName): boolean {
  return isBundled(set) || set === 'csv';
}

/** SetSpec of a bundled set; `domain` is a placeholder, `sample` takes the real one from the rows. */
export function bundledSpec(name: BundledName): SetSpec {
  const m = REAL_META[name];
  const d = m.featureNames.length;
  return {
    name,
    kind: m.kind,
    d,
    domain: Array.from({ length: d }, () => [0, 1] as [number, number]),
    ...(m.classNames ? { numClasses: m.classNames.length } : {}),
    featureNames: m.featureNames,
    targetName: m.targetName,
    ...(m.classNames ? { classNames: m.classNames } : {}),
    real: true,
    f: () => NaN,
  };
}

const tables = new Map<DatasetName, RealTable>();

export function registerTable(set: DatasetName, table: RealTable): void {
  tables.set(set, table);
}

export function getTable(set: DatasetName): RealTable | undefined {
  return tables.get(set);
}

export function hasTable(set: DatasetName): boolean {
  return tables.has(set);
}

const LOADERS: Record<BundledName, () => Promise<{ default: RealTable }>> = {
  iris: () => import('./real/iris.json'),
  penguins: () => import('./real/penguins.json'),
  wine: () => import('./real/wine.json'),
  diabetes: () => import('./real/diabetes.json'),
};

/**
 * Makes the rows of a bundled set available to `sample` (a dynamic import of the bundled JSON, so
 * the app fetches it from its own origin only when the set is chosen). Resolves at once when
 * loaded. The CSV set is registered by `registerCsv`, never loaded.
 */
export async function loadRealSet(set: DatasetName): Promise<void> {
  if (hasTable(set)) return;
  if (!isBundled(set)) throw new Error('There is no data for this set yet. Open a CSV file first.');
  const mod = await LOADERS[set]();
  registerTable(set, mod.default);
}

/**
 * The rows of a real set in a seeded shuffle (the train and validation split takes the last
 * round(n * val) of them), and the per-column range in raw units.
 */
export function realRows(
  set: DatasetName,
  rng: Rng,
): { rows: { x: number[]; y: number }[]; domain: [number, number][] } {
  const t = tables.get(set);
  if (!t) throw new Error('The rows of this dataset have not been loaded.');
  const n = t.y.length;
  const order = Array.from({ length: n }, (_, i) => i);
  const r = rng.fork('split');
  for (let i = n - 1; i > 0; i--) {
    const j = r.int(i + 1);
    const a = order[i] as number;
    order[i] = order[j] as number;
    order[j] = a;
  }
  const d = t.x[0]?.length ?? 0;
  const domain: [number, number][] = Array.from({ length: d }, () => [Infinity, -Infinity]);
  for (const row of t.x)
    row.forEach((v, c) => {
      const dom = domain[c] as [number, number];
      if (v < dom[0]) dom[0] = v;
      if (v > dom[1]) dom[1] = v;
    });
  const rows = order.map((i) => ({
    x: (t.x[i] as number[]).map(Math.fround),
    y: t.y[i] as number,
  }));
  return { rows, domain: domain.map(([lo, hi]) => (lo < hi ? [lo, hi] : [lo - 1, lo + 1])) };
}
