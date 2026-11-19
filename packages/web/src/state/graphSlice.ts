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

/** The SPEC "Graph format" example. */
export function initialGraph(): Graph {
  const parsed = parseGraph({
    version: 1,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'lin1', type: 'linear', out: 4, bias: true },
      { id: 'act1', type: 'activation', fn: 'relu' },
      { id: 'lin2', type: 'linear', out: 1, bias: true },
      { id: 'yhat', type: 'output', link: 'identity' },
      { id: 'y', type: 'target' },
      { id: 'loss', type: 'loss', fn: 'squared_error', penalty: { kind: 'l2', lambda: 0.0 } },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ],
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
      ...(preset !== undefined ? { preset, promptDismissed: false } : {}),
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
  loadGraph: (graph) =>
    set({
      graph,
      past: [],
      future: [],
      pastPresets: [],
      futurePresets: [],
      preset: null,
      promptDismissed: false,
      canUndo: false,
      canRedo: false,
      error: null,
    }),
  clearError: () => set({ error: null }),
});
