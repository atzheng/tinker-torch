import {
  analyse,
  fitTrees,
  treeLeafCounts,
  type Analysis,
  type Dataset,
  type Graph,
  type TreeFits,
} from '@tinker/core';

type DatasetFit = Pick<Dataset, 'info' | 'nTrain' | 'xTrain' | 'yTrain'>;

const EMPTY: TreeFits = Object.freeze({}) as TreeFits;

/** What a fit depends on besides the data: the Tree and Input nodes and the wires (positions and training settings are not in it). */
function fitKey(graph: Graph): string | null {
  const nodes = graph.nodes.filter((n) => n.type === 'tree' || n.type === 'input');
  if (!nodes.some((n) => n.type === 'tree')) return null;
  return JSON.stringify([nodes, graph.edges]);
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
