import type { StateCreator } from 'zustand';
import { parseGraph, type EditResult, type Graph } from '@tinker/core';
import type { ActivePreset, PresetSlice } from './presetSlice.js';

export type { EditResult };

/** Undo history is capped so a long session cannot grow without bound. */
export const HISTORY_LIMIT = 200;

export type GraphSlice = {
  graph: Graph;
  /** Earlier graphs, oldest first. */
  past: Graph[];
  /** Graphs undone, most recently undone last. */
  future: Graph[];
  /**
   * The preset (title, prompt, locked) that went with each entry of `past` and `future`, same
   * order and length: undo and redo restore the preset together with the graph.
   */
  pastPresets: (ActivePreset | null)[];
  futurePresets: (ActivePreset | null)[];
  canUndo: boolean;
  canRedo: boolean;
  /** Message of the last refused edit, for the toast. Cleared by the next ok edit or `clearError`. */
  error: string | null;
  /** Bumped when a different graph is loaded (`loadGraph`, or `apply` with a preset); not by edits. */
  loadEpoch: number;
  /**
   * Apply a pure edit. An ok result becomes the current graph and is pushed on the history
   * (redo is cleared). An error result leaves the graph alone and sets `error`.
   * Returns the result so the caller can read `select`.
   * `preset` (a file load): the preset that goes with the new graph (null for none); left out,
   * the current preset stays. Either way it is saved in the history with the graph.
   */
  apply: (edit: EditResult, preset?: ActivePreset | null) => EditResult;
  undo: () => void;
  redo: () => void;
  /** Replace the graph and clear the history and the preset (a new graph). */
  loadGraph: (graph: Graph) => void;
  clearError: () => void;
};

/** The SPEC "Graph format" example, in graph v2: a network pinned as ŷ and the squared-error group pinned as L. */
export function initialGraph(): Graph {
  const parsed = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'lin1', type: 'linear', out: 4, bias: true },
      { id: 'act1', type: 'activation', fn: 'relu' },
      { id: 'lin2', type: 'linear', out: 1, bias: true },
      { id: 'y', type: 'target' },
      { id: 'diff', type: 'subtract' },
      { id: 'err', type: 'square' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    predictions: ['lin2'],
    loss: 'loss',
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!parsed.ok) throw new Error(`initial graph is invalid: ${parsed.error}`);
  return parsed.graph;
}

export const createGraphSlice: StateCreator<GraphSlice & PresetSlice, [], [], GraphSlice> = (
  set,
  get,
) => ({
  graph: initialGraph(),
  loadEpoch: 0,
  past: [],
  future: [],
  pastPresets: [],
  futurePresets: [],
  canUndo: false,
  canRedo: false,
  error: null,
  apply: (edit, preset) => {
    if (!edit.ok) {
      set({ error: edit.error });
      return edit;
    }
    const { graph, past, pastPresets, preset: current } = get();
    const nextPast = [...past, graph].slice(-HISTORY_LIMIT);
    set({
      graph: edit.graph,
      past: nextPast,
      pastPresets: [...pastPresets, current].slice(-HISTORY_LIMIT),
      future: [],
      futurePresets: [],
      ...(preset !== undefined
        ? { preset, promptDismissed: false, loadEpoch: get().loadEpoch + 1 }
        : {}),
      canUndo: true,
      canRedo: false,
      error: null,
    });
    return edit;
  },
  undo: () => {
    const { graph, past, future, pastPresets, futurePresets, preset } = get();
    const prev = past[past.length - 1];
    if (!prev) return;
    const nextPast = past.slice(0, -1);
    const prevPreset = pastPresets[pastPresets.length - 1] ?? null;
    set({
      graph: prev,
      past: nextPast,
      pastPresets: pastPresets.slice(0, -1),
      future: [...future, graph],
      futurePresets: [...futurePresets, preset],
      preset: prevPreset,
      ...(prevPreset !== preset ? { promptDismissed: false } : {}),
      canUndo: nextPast.length > 0,
      canRedo: true,
      error: null,
    });
  },
  redo: () => {
    const { graph, past, future, pastPresets, futurePresets, preset } = get();
    const next = future[future.length - 1];
    if (!next) return;
    const nextFuture = future.slice(0, -1);
    const nextPreset = futurePresets[futurePresets.length - 1] ?? null;
    set({
      graph: next,
      past: [...past, graph],
      pastPresets: [...pastPresets, preset],
      futurePresets: futurePresets.slice(0, -1),
      preset: nextPreset,
      ...(nextPreset !== preset ? { promptDismissed: false } : {}),
      future: nextFuture,
      canUndo: true,
      canRedo: nextFuture.length > 0,
      error: null,
    });
  },
  loadGraph: (loaded) => {
    // Re-parse so a hand-built graph that left out optional fields gets the schema defaults;
    // a field that is undefined would turn a controlled input uncontrolled. An invalid graph is
    // kept as given, so the page shows its issues.
    const reparsed = parseGraph(loaded);
    const graph = reparsed.ok ? reparsed.graph : loaded;
    set({
      graph,
      loadEpoch: get().loadEpoch + 1,
      past: [],
      future: [],
      pastPresets: [],
      futurePresets: [],
      preset: null,
      promptDismissed: false,
      canUndo: false,
      canRedo: false,
      error: null,
    });
  },
  clearError: () => set({ error: null }),
});
