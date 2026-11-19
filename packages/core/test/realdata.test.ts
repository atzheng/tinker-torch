import { describe, expect, it } from 'vitest';
import {
  BUNDLED,
  REAL_META,
  SETS,
  buildCsv,
  grid2d,
  grid2dSlice,
  inferKind,
  loadRealSet,
  parseCsv,
  parseCsvCells,
  registerCsv,
  sample,
  usableInputs,
  type DataConfig,
} from '../src/index.js';

const cfg = (set: DataConfig['set'], extra: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 40,
  noise: 0,
  val: 0.2,
  seed: 1,
  standardize: false,
  ...extra,
});

describe('bundled datasets', () => {
  for (const name of BUNDLED) {
    it(`${name}: rows, names and classes agree with the catalogue`, async () => {
      await loadRealSet(name);
      const meta = REAL_META[name];
      const ds = sample(cfg(name));
      expect(ds.nTrain + ds.nVal).toBe(meta.rows);
      expect(ds.info.columns).toEqual(meta.featureNames.map((_, i) => `x${i + 1}`));
      expect(ds.info.featureNames).toEqual(meta.featureNames);
      expect(ds.info.targetName).toBe(meta.targetName);
      expect(ds.info.kind).toBe(meta.kind);
      expect(SETS[name].d).toBe(meta.featureNames.length);
      if (meta.classNames) {
        expect(ds.info.classNames).toEqual(meta.classNames);
        const all = [...ds.yTrain, ...ds.yVal];
        expect(new Set(all)).toEqual(new Set(meta.classNames.map((_, i) => i)));
      }
      for (const [lo, hi] of ds.domain) expect(hi).toBeGreaterThan(lo);
    });
  }

  it('the split is seeded and n does not matter', async () => {
    await loadRealSet('iris');
    const a = sample(cfg('iris', { n: 10 }));
    const b = sample(cfg('iris', { n: 500 }));
    expect(a.nTrain).toBe(120);
    expect(a.nVal).toBe(30);
    expect(Array.from(b.xTrain)).toEqual(Array.from(a.xTrain));
    const c = sample(cfg('iris', { seed: 2 }));
    expect(Array.from(c.xTrain)).not.toEqual(Array.from(a.xTrain));
  });

  it('standardises with the training mean and std', async () => {
    await loadRealSet('wine');
    const ds = sample(cfg('wine', { standardize: true }));
    const d = 13;
    for (let c = 0; c < d; c++) {
      let m = 0;
      for (let i = 0; i < ds.nTrain; i++) m += ds.xTrain[i * d + c] ?? 0;
      expect(m / ds.nTrain).toBeCloseTo(0, 4);
    }
  });
});

describe('CSV parser', () => {
  it('handles quotes, escaped quotes, embedded commas and newlines, CRLF', () => {
    const rows = parseCsvCells('a,b\r\n"x, y","say ""hi"""\r\n"two\nlines",3\r\n\r\n', ',');
    expect(rows).toEqual([
      ['a', 'b'],
      ['x, y', 'say "hi"'],
      ['two\nlines', '3'],
    ]);
  });

  it('keeps empty trailing cells and detects semicolons and tabs', () => {
    expect(parseCsvCells('a,b,\n1,,', ',')).toEqual([
      ['a', 'b', ''],
      ['1', '', ''],
    ]);
    const r = parseCsv('a;b\n1;2\n3;4');
    expect(r.ok && r.csv.rows).toEqual([
      ['1', '2'],
      ['3', '4'],
    ]);
    const t = parseCsv('a\tb\n1\t2');
    expect(t.ok && t.csv.header).toEqual(['a', 'b']);
  });

  it('gives one-sentence errors', () => {
    const msg = (s: string) => {
      const r = parseCsv(s);
      return r.ok ? 'ok' : r.error;
    };
    expect(msg('')).toBe('The file is empty.');
    expect(msg('just one column\n1\n2')).toMatch(/does not look like a CSV/);
    expect(msg('a,b\n')).toMatch(/no data rows/);
    expect(msg('a,b\n1,2,3')).toMatch(/Row 2 has 3 values but the header has 2/);
    expect(msg('a,b\n"1,2')).toMatch(/never closed/);
    expect(msg('a,b\n' + '1,2\n'.repeat(5001))).toMatch(/at most 5000 rows/);
    expect(msg(Array.from({ length: 51 }, (_, i) => `c${i}`).join(',') + '\n1')).toMatch(
      /at most 50 columns/,
    );
  });

  it('names blank and repeated headers', () => {
    const r = parseCsv('a,a,\n1,2,3');
    expect(r.ok && r.csv.header).toEqual(['a', 'a 2', 'column 3']);
  });
});

const IRISISH = [
  'len,wid,kind,label,note',
  ...Array.from(
    { length: 30 },
    (_, i) => `${i * 0.5},${(i % 7) + 1},${['a', 'b', 'c'][i % 3]},${i % 3},t${i}`,
  ),
].join('\n');

describe('CSV to dataset', () => {
  it('infers kinds and builds a trainable multi-class set, dropping text inputs', () => {
    const p = parseCsv(IRISISH);
    if (!p.ok) throw new Error(p.error);
    const { csv } = p;
    expect(inferKind(csv, 2)).toBe('multiclass'); // text with 3 values
    expect(inferKind(csv, 3)).toBe('multiclass'); // integer labels
    expect(inferKind(csv, 0)).toBe('regression');
    expect(inferKind(csv, 4)).toBe(null); // 30 distinct strings
    expect(usableInputs(csv, 2)).toEqual([0, 1, 3]);

    const b = buildCsv(csv, { target: 2, inputs: [0, 1, 4], kind: 'multiclass' });
    if (!b.ok) throw new Error(b.error);
    expect(b.notes[0]).toMatch(/Left out 1 text column: note/);
    registerCsv(b);
    const ds = sample(cfg('csv'));
    expect(ds.info.featureNames).toEqual(['len', 'wid']);
    expect(ds.info.classNames).toEqual(['a', 'b', 'c']);
    expect(ds.info.targetName).toBe('kind');
    expect(ds.info.numClasses).toBe(3);
    expect(ds.nTrain + ds.nVal).toBe(30);
  });

  it('drops incomplete rows and refuses wrong kinds with a sentence', () => {
    const rows = Array.from({ length: 12 }, (_, i) => `${i},${i * 2},${i === 1 ? '' : i + 1}`);
    const p = parseCsv(['x,z,y', ...rows].join('\n'));
    if (!p.ok) throw new Error(p.error);
    const b = buildCsv(p.csv, { target: 2, inputs: [0, 1], kind: 'regression' });
    expect(b.ok && b.notes).toEqual(['Dropped 1 row with a missing value.']);
    const bad = buildCsv(p.csv, { target: 2, inputs: [0, 1], kind: 'binary' });
    expect(!bad.ok && bad.error).toMatch(/Two classes are needed/);
    const one = buildCsv(p.csv, { target: 2, inputs: [0], kind: 'multiclass' });
    expect(!one.ok && one.error).toMatch(/at least two numeric input columns/);
    const tiny = parseCsv('x,y\n1,2\n2,3');
    if (!tiny.ok) throw new Error(tiny.error);
    const t = buildCsv(tiny.csv, { target: 1, inputs: [0], kind: 'regression' });
    expect(!t.ok && t.error).toMatch(/at least 10 are needed/);
  });
});

describe('grid2dSlice', () => {
  it('equals grid2d for d = 2 and holds the other columns fixed for d > 2', () => {
    const dom: [number, number][] = [
      [0, 1],
      [2, 4],
      [-1, 1],
      [5, 6],
    ];
    const a = grid2d(dom.slice(0, 2), 5);
    const b = grid2dSlice(dom.slice(0, 2), [0, 0], 0, 1, 5);
    expect(Array.from(b.xRaw)).toEqual(Array.from(a.xRaw));
    const g = grid2dSlice(dom, [0.5, 3, 0.25, 5.5], 3, 1, 4);
    expect(g.d).toBe(4);
    expect(g.n).toBe(16);
    for (let i = 0; i < g.n; i++) {
      expect(g.xRaw[i * 4]).toBe(0.5);
      expect(g.xRaw[i * 4 + 2]).toBe(0.25);
    }
    expect(g.xRaw[3]).toBe(5); // first row: x4 at its low end
    expect(g.xRaw[4 * 3 + 3]).toBe(6); // fastest-varying column ends at its high end
  });
});
