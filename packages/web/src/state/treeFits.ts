import {
  analyse,
  fitTrees,
  resolveEdge,
  treeLeafCounts,
  type Analysis,
  type Dataset,
  type Graph,
  type TreeFits,
} from '@tinker/core';

type DatasetFit = Pick<Dataset, 'info' | 'nTrain' | 'xTrain' | 'yTrain'>;

const EMPTY: TreeFits = Object.freeze({}) as TreeFits;

/** What a fit depends on besides the data: each Tree's settings and the Input wired straight into it (no other wire, node or position). */
function fitKey(graph: Graph): string | null {
  const trees = graph.nodes.filter((n) => n.type === 'tree');
  if (trees.length === 0) return null;
  return JSON.stringify(
    trees.map((t) => {
      let src: unknown = null;
      for (const e of graph.edges) {
        const r = resolveEdge(graph.nodes, e);
        if (r.ok && r.to.node === t.id && r.to.port === 'in') {
          const from = graph.nodes.find((n) => n.id === r.from.node);
          src = from?.type === 'input' ? from : (from?.id ?? null);
          break;
        }
      }
      return [t, src];
    }),
  );
}

/** Whether two fit sets are the same trees (per id: leaves, split features and thresholds). */
export function sameFits(a: TreeFits, b: TreeFits): boolean {
  const sig = (f: TreeFits) =>
    JSON.stringify(
      Object.keys(f)
        .sort()
        .map((id) => [
          id,
          f[id]?.leaves,
          f[id]?.columns,
          f[id]?.nodes.map((n) => (n.kind === 'split' ? [n.feature, n.threshold] : n.leaf)),
        ]),
    );
  return a === b || sig(a) === sig(b);
}

const cache = new WeakMap<object, { key: string; fits: TreeFits }>();

/**
 * The host-side tree fits of `graph` on `dataset`'s training rows (CONTRACTS.md section 3, "Tree").
 * Memoised on the dataset object and the tree settings, so the same object comes back until the data
 * or a tree setting changes (callers compare by identity to know a refit happened); a graph without a
 * Tree always gets the same empty object.
 */
export function treeFitsOf(graph: Graph, dataset: DatasetFit): TreeFits {
  const key = fitKey(graph);
  if (key === null) return EMPTY;
  const hit = cache.get(dataset);
  if (hit && hit.key === key) return hit.fits;
  const fits = fitTrees(graph, dataset);
  cache.set(dataset, { key, fits });
  return fits;
}

/** `analyse` with the Tree widths from the fits (what every web caller needs). */
export function analyseWithTrees(graph: Graph, dataset: DatasetFit): Analysis {
  return analyse(graph, dataset.info, treeLeafCounts(treeFitsOf(graph, dataset)));
}
