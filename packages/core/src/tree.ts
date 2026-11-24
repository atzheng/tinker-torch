/**
 * Decision-tree fit for the Tree piece (CONTRACTS.md section 3, "Tree"). Pure, float64, no DOM, no jax.
 *
 * Greedy CART as scikit-learn's `DecisionTreeRegressor` / `DecisionTreeClassifier` (best splitter,
 * `max_features=None`). The tree is fitted on the host from the training rows whenever the data or the
 * tree's settings change; its output is the one-hot of the leaf each row falls in.
 */
import type { DataInfo, Graph, NodeId } from './graph.js';
import { resolveEdge, type TreeCriterion } from './graph.js';
import type { Dataset } from './data/index.js';

/** The settings of a Tree node (the fields of `NodeOf<'tree'>` without `id` and `type`). */
export type TreeSettings = {
  criterion: TreeCriterion;
  /** Maximum depth (the root is depth 0): 1..8. */
  depth: number;
  /** A node with fewer examples than this is a leaf. */
  minSplit: number;
  /** Every leaf keeps at least this many examples. */
  minLeaf: number;
  /** A split must lower the weighted impurity by at least this much (sklearn's min_impurity_decrease). */
  minDecrease: number;
};

export type TreeSplitNode = {
  kind: 'split';
  /** Column of the row (index into a row of the `x` passed to `fitTree` / `treeLeafIndex`). */
  feature: number;
  /** `x[feature] <= threshold` goes left. */
  threshold: number;
  /** Indices into `TreeFit.nodes`. */
  left: number;
  right: number;
  /** Training examples reaching the node. */
  n: number;
};
export type TreeLeafNode = {
  kind: 'leaf';
  /** Leaf number, 0-based, left to right (depth-first, left child first). */
  leaf: number;
  n: number;
};
export type TreeNodeFit = TreeSplitNode | TreeLeafNode;
export type TreeFit = {
  /** `nodes[0]` is the root. A tree that never splits is one leaf. */
  nodes: TreeNodeFit[];
  /** Number of leaves, at least 1: the output width of the Tree piece. */
  leaves: number;
};

const EPS = Number.EPSILON;
/** sklearn's FEATURE_THRESHOLD: values closer than this are the same value. */
const FEATURE_THRESHOLD = 1e-7;
/** Improvements closer than this (relative) count as a tie; the earlier candidate wins. */
const TIE = 1e-12;

/**
 * Impurity bookkeeping for one criterion. `reset(rows)` puts every row of the node on the right and
 * none on the left; `moveLeft(row)` moves one row across; `left()` and `right()` give the impurity of
 * each side; `node(rows)` the impurity of a whole set of rows (two-pass, exactly 0 when pure).
 */
type Scanner = {
  node(rows: Int32Array): number;
  reset(rows: Int32Array): void;
  moveLeft(row: number): void;
  left(): number;
  right(): number;
};

function makeScanner(criterion: TreeCriterion, y: ArrayLike<number>, classes: number): Scanner {
  const yv = (r: number) => y[r] as number;
  if (criterion === 'squared_error') {
    // Running sums of the targets shifted by the node mean (keeps the variance free of cancellation).
    let shift = 0;
    let sL = 0;
    let qL = 0;
    let sR = 0;
    let qR = 0;
    let nL = 0;
    let nR = 0;
    const variance = (s: number, q: number, m: number) => Math.max(0, q / m - (s / m) * (s / m));
    return {
      node(rows) {
        let mean = 0;
        for (const r of rows) mean += yv(r);
        mean /= rows.length;
        let ss = 0;
        for (const r of rows) ss += (yv(r) - mean) ** 2;
        return ss / rows.length;
      },
      reset(rows) {
        shift = 0;
        for (const r of rows) shift += yv(r);
        shift /= rows.length;
        sL = qL = sR = qR = nL = 0;
        nR = rows.length;
        for (const r of rows) {
          const z = yv(r) - shift;
          sR += z;
          qR += z * z;
        }
      },
      moveLeft(r) {
        const z = yv(r) - shift;
        sL += z;
        qL += z * z;
        nL++;
        sR -= z;
        qR -= z * z;
        nR--;
      },
      left: () => variance(sL, qL, nL),
      right: () => variance(sR, qR, nR),
    };
  }
  if (criterion === 'gini' || criterion === 'entropy') {
    const cL = new Float64Array(classes);
    const cR = new Float64Array(classes);
    let nL = 0;
    let nR = 0;
    const of = (c: Float64Array, m: number) => {
      let s = 0;
      for (let k = 0; k < classes; k++) {
        const q = (c[k] as number) / m;
        if (criterion === 'gini') s += q * q;
        else if (q > 0) s -= q * Math.log2(q);
      }
      return criterion === 'gini' ? 1 - s : s;
    };
    return {
      node(rows) {
        const c = new Float64Array(classes);
        for (const r of rows) c[yv(r)] = (c[yv(r)] as number) + 1;
        return of(c, rows.length);
      },
      reset(rows) {
        cL.fill(0);
        cR.fill(0);
        for (const r of rows) cR[yv(r)] = (cR[yv(r)] as number) + 1;
        nL = 0;
        nR = rows.length;
      },
      moveLeft(r) {
        cL[yv(r)] = (cL[yv(r)] as number) + 1;
        cR[yv(r)] = (cR[yv(r)] as number) - 1;
        nL++;
        nR--;
      },
      left: () => of(cL, nL),
      right: () => of(cR, nR),
    };
  }
  // absolute_error: mean absolute deviation from the median. Each side keeps its values sorted.
  let L = new Float64Array(0);
  let R = new Float64Array(0);
  let nL = 0;
  let nR = 0;
  const mad = (v: Float64Array, m: number) => {
    const med =
      m % 2 === 1
        ? (v[(m - 1) / 2] as number)
        : ((v[m / 2 - 1] as number) + (v[m / 2] as number)) / 2;
    let s = 0;
    for (let i = 0; i < m; i++) s += Math.abs((v[i] as number) - med);
    return s / m;
  };
  /** First index in `v[0..m)` with `v[i] >= t`. */
  const lower = (v: Float64Array, m: number, t: number) => {
    let lo = 0;
    let hi = m;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((v[mid] as number) < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  return {
    node(rows) {
      const v = Float64Array.from(rows, yv).sort();
      return mad(v, v.length);
    },
    reset(rows) {
      L = new Float64Array(rows.length);
      R = Float64Array.from(rows, yv).sort();
      nL = 0;
      nR = rows.length;
    },
    moveLeft(r) {
      const t = yv(r);
      const at = lower(R, nR, t);
      R.copyWithin(at, at + 1, nR);
      nR--;
      const to = lower(L, nL, t);
      L.copyWithin(to + 1, to, nL);
      L[to] = t;
      nL++;
    },
    left: () => mad(L, nL),
    right: () => mad(R, nR),
  };
}

/**
 * Fits a tree on `n` rows of `x` (row-major `[n, d]`) with targets `y` (regression: the value;
 * classification: the class index 0..numClasses-1). Float64 throughout; pass the training rows only.
 * Candidate thresholds are the midpoints between consecutive distinct sorted values of a column.
 * A node is a leaf at `depth`, when it has fewer than `minSplit` examples or fewer than `2 * minLeaf`,
 * when it is pure, or when no split keeps `minLeaf` examples on each side and lowers the weighted
 * impurity by more than zero and by at least `minDecrease`. Ties pick the lowest feature, then the
 * lowest threshold. Does not modify its arguments.
 */
export function fitTree(
  x: ArrayLike<number>,
  n: number,
  d: number,
  y: ArrayLike<number>,
  settings: TreeSettings,
  data: Pick<DataInfo, 'numClasses'>,
): TreeFit {
  const classes = Math.max(2, data.numClasses);
  const nodes: TreeNodeFit[] = [];
  let leaves = 0;
  const sc = makeScanner(settings.criterion, y, classes);

  const build = (rows: Int32Array, depth: number): number => {
    const id = nodes.length;
    const m = rows.length;
    const imp = sc.node(rows);
    const leaf = (): number => {
      nodes.push({ kind: 'leaf', leaf: leaves++, n: m });
      return id;
    };
    if (
      depth >= settings.depth ||
      m < settings.minSplit ||
      m < 2 * settings.minLeaf ||
      imp <= EPS
    ) {
      return leaf();
    }
    // A gain is m * (imp - N_L/m imp_L - N_R/m imp_R): N_t times sklearn's bracket.
    const total = m * imp;
    const tol = TIE * total;
    let best: { gain: number; feature: number; threshold: number } | null = null;
    for (let f = 0; f < d; f++) {
      const order = Int32Array.from(rows);
      order.sort((a, b) => (x[a * d + f] as number) - (x[b * d + f] as number) || a - b);
      sc.reset(order);
      for (let p = 1; p < m; p++) {
        sc.moveLeft(order[p - 1] as number);
        const lo = x[(order[p - 1] as number) * d + f] as number;
        const hi = x[(order[p] as number) * d + f] as number;
        if (hi <= lo + FEATURE_THRESHOLD) continue;
        if (p < settings.minLeaf || m - p < settings.minLeaf) continue;
        const gain = total - p * sc.left() - (m - p) * sc.right();
        // Strictly better than the best so far (beyond float noise), and above zero.
        if (best === null ? gain > tol : gain > best.gain + tol) {
          let threshold = lo / 2 + hi / 2;
          if (threshold === hi || !Number.isFinite(threshold)) threshold = lo;
          best = { gain, feature: f, threshold };
        }
      }
    }
    if (best === null) return leaf();
    // sklearn's weighted decrease N_t / N * (imp - ...) is gain / N.
    if (best.gain / n + EPS < settings.minDecrease) return leaf();
    const left: number[] = [];
    const right: number[] = [];
    for (const r of rows) {
      if ((x[r * d + best.feature] as number) <= best.threshold) left.push(r);
      else right.push(r);
    }
    nodes.push({
      kind: 'split',
      feature: best.feature,
      threshold: best.threshold,
      left: -1,
      right: -1,
      n: m,
    });
    const l = build(Int32Array.from(left), depth + 1);
    const r = build(Int32Array.from(right), depth + 1);
    const self = nodes[id] as TreeSplitNode;
    self.left = l;
    self.right = r;
    return id;
  };

  const all = new Int32Array(n);
  for (let i = 0; i < n; i++) all[i] = i;
  build(all, 0);
  return { nodes, leaves };
}

/**
 * The leaf number (0-based) of each of the `n` rows of `x` (row-major, row width `d`). `x` may be any
 * rows: training, validation or a grid. Compares in float64 on the values given (float32 is exact in it).
 */
export function treeLeafIndex(
  fit: TreeFit,
  x: ArrayLike<number>,
  n: number,
  d: number,
): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let node = fit.nodes[0] as TreeNodeFit;
    while (node.kind === 'split') {
      const v = x[i * d + node.feature] as number;
      node = fit.nodes[v <= node.threshold ? node.left : node.right] as TreeNodeFit;
    }
    out[i] = node.leaf;
  }
  return out;
}

/** The one-hot of `treeLeafIndex`, row-major `[n, fit.leaves]` (what a Tree piece outputs). */
export function treeOneHot(fit: TreeFit, x: ArrayLike<number>, n: number, d: number): Float32Array {
  const idx = treeLeafIndex(fit, x, n, d);
  const out = new Float32Array(n * fit.leaves);
  for (let i = 0; i < n; i++) out[i * fit.leaves + (idx[i] as number)] = 1;
  return out;
}

/** A fit per Tree node. `columns` are the dataset column indices the tree reads (its Input's columns). */
export type TreeFits = Record<NodeId, TreeFit & { columns: number[] }>;

/** Leaf count per Tree node id (what `analyse` takes as `treeLeaves`). */
export function treeLeafCounts(fits: TreeFits): Record<NodeId, number> {
  const out: Record<NodeId, number> = {};
  for (const [id, f] of Object.entries(fits)) out[id] = f.leaves;
  return out;
}

/**
 * Fits every Tree node of `graph` whose `in` is wired straight from an Input, on the dataset's
 * training rows only (never validation), with `dataset.yTrain` as the target. Needs no analysis, so it
 * runs before `analyse`. The fit's `feature` indices are dataset column indices (positions in a full
 * dataset row), so `treeOneHot(fit, rows, n, dataset.info.columns.length)` works on any full-width
 * rows (batch, validation, grid). A Tree with no Input wired, an Input with no known or a category
 * column, or a criterion that does not match the data kind gets no entry (analyse says why).
 */
export function fitTrees(graph: Graph, dataset: Dataset): TreeFits {
  const info = dataset.info;
  const d = info.columns.length;
  const out: TreeFits = {};
  for (const node of graph.nodes) {
    if (node.type !== 'tree' || node.id in out) continue;
    const regression = info.kind === 'regression';
    const criterionOk =
      node.criterion === 'squared_error' || node.criterion === 'absolute_error'
        ? regression
        : !regression;
    if (!criterionOk) continue;
    let src = null;
    for (const e of graph.edges) {
      const r = resolveEdge(graph.nodes, e);
      if (r.ok && r.to.node === node.id && r.to.port === 'in') {
        src = graph.nodes.find((n) => n.id === r.from.node) ?? null;
        break;
      }
    }
    if (src?.type !== 'input' || src.columns.length === 0) continue;
    const cols = src.columns.map((c) => info.columns.indexOf(c));
    if (cols.some((c, i) => c < 0 || cols.indexOf(c) !== i)) continue;
    if (cols.some((c) => info.columnKinds?.[c] === 'categorical')) continue;
    const n = dataset.nTrain;
    // Features index full dataset rows, so fit on the full-width matrix restricted to the chosen columns.
    const sub = new Float64Array(n * cols.length);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < cols.length; j++) {
        sub[i * cols.length + j] = dataset.xTrain[i * d + (cols[j] as number)] as number;
      }
    }
    const fit = fitTree(sub, n, cols.length, dataset.yTrain, node, info);
    for (const nd of fit.nodes) if (nd.kind === 'split') nd.feature = cols[nd.feature] as number;
    out[node.id] = { ...fit, columns: cols };
  }
  return out;
}
