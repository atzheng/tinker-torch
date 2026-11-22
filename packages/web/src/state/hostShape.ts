import {
  analyse,
  lookupCategories,
  type Graph,
  type HostParams,
  type DataInfo,
} from '@tinker/core';

/**
 * `HostParams` carries flat arrays only, so [2, 3] and [3, 2] look the same. The train slice
 * publishes `host` in the same store update as the graph it belongs to, so remember that graph
 * per snapshot and derive each Linear's [in, out] from it. The neuron view draws a weight only
 * when that shape equals the current one.
 */
const graphOf = new WeakMap<HostParams, Graph>();
const shapesOf = new WeakMap<Graph, Record<string, [number, number] | undefined>>();

/** Call from a store subscription: remembers the graph that is current when a `host` appears. */
export function noteHost(host: HostParams | null | undefined, graph: Graph): void {
  if (host && !graphOf.has(host)) graphOf.set(host, graph);
}

/** [in, out] of a Linear ([categories, width] of a Lookup) in the graph a snapshot was published with; undefined when unknown. */
export function hostShape(
  host: HostParams | undefined,
  node: string,
  info: DataInfo,
): [number, number] | undefined {
  const g = host && graphOf.get(host);
  if (!g) return undefined;
  let shapes = shapesOf.get(g);
  if (!shapes) {
    const a = analyse(g, info);
    shapes = Object.create(null) as Record<string, [number, number] | undefined>;
    for (const n of g.nodes) {
      const inW = a.inWidth[n.id];
      if (n.type === 'linear' && inW !== undefined && shapes[n.id] === undefined) {
        shapes[n.id] = [inW, n.out];
      }
      if (n.type === 'lookup' && shapes[n.id] === undefined) {
        const k = lookupCategories(a, n.id);
        if (k !== undefined) shapes[n.id] = [k, n.width];
      }
    }
    shapesOf.set(g, shapes);
  }
  return shapes[node];
}
