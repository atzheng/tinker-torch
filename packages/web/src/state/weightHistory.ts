import type { Graph, HostParams, NodeId } from '@tinker/core';
import type { StoreApi } from 'zustand';
import type { Store } from './store.js';

/**
 * Undo for hand-set weights (tinker-ran.2). The graph history holds graphs only, so a committed
 * weight edit is recorded as a content-identical copy of the graph (a new object, so it is one
 * more entry in the ordinary undo stack, in order with the structural edits) and a record here,
 * keyed by that copy. When the current graph moves from the copy back to `prev` (undo) the layer's
 * old values are written back; when it moves from `prev` to the copy (redo) the new ones.
 * Weights trained between two edits are not part of undo: undo restores the layer as it was
 * when the drag began.
 */
export type LayerValues = { W?: Float32Array; b?: Float32Array };
type Record_ = {
  prev: Graph;
  node: NodeId;
  before: LayerValues;
  after: LayerValues;
  generation: number;
};

const records = new WeakMap<Graph, Record_>();

/**
 * Bumped whenever the weights are replaced outside the history: Reset, Restore best, a loaded
 * file, a fresh start after a divergence. A record from an older generation belongs to weights
 * that no longer exist, so undo and redo skip its write-back (the graph still moves).
 */
let generation = 0;
export function bumpWeightGeneration(): void {
  generation += 1;
}
export function weightGeneration(): number {
  return generation;
}

let applying = false;
/** Run the `apply` of a new hand edit: the subscriber must not treat it as a redo. */
export function whileApplyingEdit(fn: () => void): void {
  applying = true;
  try {
    fn();
  } finally {
    applying = false;
  }
}

/** Remember an edit; `next` must be the copy of `prev` that is applied to the graph history. */
export function recordWeightEdit(
  prev: Graph,
  next: Graph,
  node: NodeId,
  before: LayerValues,
  after: LayerValues,
): void {
  records.set(next, { prev, node, before, after, generation });
}

/** Subscribe once (store.ts): applies the stored values when undo or redo crosses a record. */
export function attachWeightHistory(store: StoreApi<Store>): void {
  let last = store.getState().graph;
  store.subscribe(() => {
    const s = store.getState(); // the listener argument can be stale inside nested set() calls
    const g = s.graph;
    if (g === last) return;
    const from = last;
    last = g;
    if (applying) return; // the original apply of a hand edit, not an undo or redo
    const undone = records.get(from);
    const redone = records.get(g);
    if (undone && undone.prev === g && undone.generation === generation) {
      s.writeParams(undone.node, undone.before);
    } else if (redone && redone.prev === from && redone.generation === generation) {
      s.writeParams(redone.node, redone.after);
    }
  });
}

/**
 * Group into Linear / Expand to scalars (tinker-9io.7): the graph changes shape but the function and
 * its values stay, so the edit records the parameters on both sides. The train slice asks
 * `scalarHostFor` on every graph change; it answers when the change crosses a record (the edit itself,
 * its undo, its redo), so graph and parameters move in one step. Like hand edits, weights trained
 * between the edit and its undo are not part of undo.
 *
 * Records are keyed by graph OBJECT identity: whoever holds the graph that `recordScalarEdit` was
 * given (the history, the store) must not clone it, or the record is no longer found.
 */
type ScalarRecord = {
  prev: Graph;
  before: HostParams;
  after: HostParams;
  /** The step count at the edit: undo and redo return the run to it (the weights are those of that step). */
  step: number;
  generation: number;
};
export type ScalarCarry = { host: HostParams; step: number };
const scalarRecords = new WeakMap<Graph, ScalarRecord>();

export function recordScalarEdit(
  prev: Graph,
  next: Graph,
  before: HostParams,
  after: HostParams,
  step: number,
): void {
  scalarRecords.set(next, { prev, before, after, step, generation });
}

/** The parameters for a move of the graph from `from` to `to` when it crosses a scalar-edit record. */
export function scalarHostFor(from: Graph, to: Graph): ScalarCarry | null {
  const undone = scalarRecords.get(from);
  if (undone && undone.prev === to && undone.generation === generation) {
    return { host: undone.before, step: undone.step };
  }
  const redone = scalarRecords.get(to);
  if (redone && redone.prev === from && redone.generation === generation) {
    return { host: redone.after, step: redone.step };
  }
  return null;
}
