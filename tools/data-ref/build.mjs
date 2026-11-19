// Opt-in: downloads the classic teaching datasets and writes the cleaned JSON assets bundled in
// packages/core/src/data/real/. Network is used here only, never by tests or the app.
// Run: pnpm data-ref   (Node >= 20.19). Sources and licences: docs/datasets.md.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '../../packages/core/src/data/real');

async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.text()).split(/\r?\n/).filter((l) => l.trim() !== '');
}

const round = (v) => Math.round(v * 1e6) / 1e6;

/** Rows with a missing or unparsable cell in `cols` (or an unknown label) are dropped. */
function numericTable(lines, sep, header, cols, labelCol, labelMap) {
  const rows = lines.slice(header ? 1 : 0).map((l) => l.split(sep).map((c) => c.trim()));
  const x = [];
  const y = [];
  for (const r of rows) {
    if (cols.some((c) => r[c] === undefined || r[c] === '' || r[c] === 'NA')) continue;
    const vals = cols.map((c) => Number(r[c]));
    if (vals.some((v) => !Number.isFinite(v))) continue;
    let label;
    if (labelMap) {
      label = labelMap.indexOf(r[labelCol]);
      if (label < 0) continue;
    } else label = Number(r[labelCol]);
    x.push(vals.map(round));
    y.push(label);
  }
  return { x, y };
}

async function write(name, t) {
  await mkdir(OUT, { recursive: true });
  const body =
    '{\n"x": [\n' +
    t.x.map((r) => JSON.stringify(r)).join(',\n') +
    '\n],\n"y": ' +
    JSON.stringify(t.y) +
    '\n}\n';
  await writeFile(join(OUT, `${name}.json`), body);
  console.log(name, t.x.length, 'rows,', t.x[0].length, 'features');
}

await write(
  'iris',
  numericTable(
    await get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'),
    ',',
    false,
    [0, 1, 2, 3],
    4,
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
  ),
);

// Wine: first column is the cultivar 1..3, then 13 measurements.
await write(
  'wine',
  numericTable(
    await get('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'),
    ',',
    false,
    Array.from({ length: 13 }, (_, i) => i + 1),
    0,
    ['1', '2', '3'],
  ),
);

// Palmer penguins: the four numeric measurements; rows with any missing value are dropped.
await write(
  'penguins',
  numericTable(
    await get(
      'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv',
    ),
    ',',
    true,
    [2, 3, 4, 5],
    0,
    ['Adelie', 'Chinstrap', 'Gentoo'],
  ),
);

// Diabetes (Efron et al. 2004), raw units: AGE SEX BMI BP S1..S6, target Y.
await write(
  'diabetes',
  numericTable(
    await get('https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'),
    '\t',
    true,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    10,
    null,
  ),
);
