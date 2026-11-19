import { parsePort, type Graph, type NodeId } from '@tinker/core';

/** The nodes carrying the ŷ pins, in pin order. */
export function predictionNodes(graph: Pick<Graph, 'predictions'>): NodeId[] {
  return graph.predictions.map((p) => parsePort(p).node);
}

/**
 * Which ŷ pin the results plots draw: the index into `graph.predictions` of the picked node, or 0
 * (the first pin) when nothing is picked or the picked node is no longer a ŷ. Picking is by node, so
 * moving or reordering pins keeps the choice.
 */
export function plotIndex(graph: Pick<Graph, 'predictions'>, pick: NodeId | null): number {
  const i = pick === null ? -1 : predictionNodes(graph).indexOf(pick);
  return i < 0 ? 0 : i;
}
