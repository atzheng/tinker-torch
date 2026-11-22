import type { StateCreator } from 'zustand';
import {
  parseFile,
  parseGraph,
  type EditResult,
  type Graph,
  type HostParams,
  type Show,
} from '@tinker/core';
import scalarPreset from '../presets/scalar.json';
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
  /** `show` parallel to `pastPresets` / `futurePresets`: it travels with the preset. */
  pastShows: (Show | null)[];
  futureShows: (Show | null)[];
  /** `startParams` parallel to `pastPresets` / `futurePresets`: it travels with the graph. */
  pastStarts: (HostParams | null)[];
  futureStarts: (HostParams | null)[];
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
  apply: (edit: EditResult, preset?: ActivePreset | null, show?: Show | null) => EditResult;
  undo: () => void;
  redo: () => void;
  /** Replace the graph and clear the history and the preset (a new graph). */
  loadGraph: (graph: Graph) => void;
  clearError: () => void;
};

/** The first-open graph: the scalar linear regression preset (`presets/scalar.json`). */
export function initialGraph(): Graph {
  const parsed = parseFile(scalarPreset);
  if (!parsed.ok) throw new Error(`initial graph is invalid: ${parsed.error}`);
  return parsed.file.graph;
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
  pastShows: [],
  futureShows: [],
  pastStarts: [],
  futureStarts: [],
  canUndo: false,
  canRedo: false,
  error: null,
  apply: (edit, preset, show) => {
    if (!edit.ok) {
      set({ error: edit.error });
      return edit;
    }
    const { graph, past, pastPresets, pastShows, preset: current, show: currentShow } = get();
    const { pastStarts, startParams } = get();
    const nextPast = [...past, graph].slice(-HISTORY_LIMIT);
    set({
      graph: edit.graph,
      past: nextPast,
      pastPresets: [...pastPresets, current].slice(-HISTORY_LIMIT),
      pastShows: [...pastShows, currentShow].slice(-HISTORY_LIMIT),
      pastStarts: [...pastStarts, startParams].slice(-HISTORY_LIMIT),
      future: [],
      futurePresets: [],
      futureShows: [],
      futureStarts: [],
      ...(preset !== undefined
        ? {
            preset,
            show: show ?? null,
            startParams: null,
            promptDismissed: false,
            loadEpoch: get().loadEpoch + 1,
          }
        : {}),
      canUndo: true,
      canRedo: false,
      error: null,
    });
    return edit;
  },
  undo: () => {
    const { graph, past, future, pastPresets, futurePresets, preset } = get();
    const { pastShows, futureShows, show } = get();
    const { pastStarts, futureStarts, startParams } = get();
    const prev = past[past.length - 1];
    if (!prev) return;
    const nextPast = past.slice(0, -1);
    const prevPreset = pastPresets[pastPresets.length - 1] ?? null;
    const prevShow = pastShows[pastShows.length - 1] ?? null;
    set({
      graph: prev,
      past: nextPast,
      pastPresets: pastPresets.slice(0, -1),
      future: [...future, graph],
      futurePresets: [...futurePresets, preset],
      pastShows: pastShows.slice(0, -1),
      futureShows: [...futureShows, show],
      pastStarts: pastStarts.slice(0, -1),
      futureStarts: [...futureStarts, startParams],
      startParams: pastStarts[pastStarts.length - 1] ?? null,
      preset: prevPreset,
      show: prevShow,
      ...(prevPreset !== preset ? { promptDismissed: false } : {}),
      canUndo: nextPast.length > 0,
      canRedo: true,
      error: null,
    });
  },
  redo: () => {
    const { graph, past, future, pastPresets, futurePresets, preset } = get();
    const { pastShows, futureShows, show } = get();
    const { pastStarts, futureStarts, startParams } = get();
    const next = future[future.length - 1];
    if (!next) return;
    const nextFuture = future.slice(0, -1);
    const nextPreset = futurePresets[futurePresets.length - 1] ?? null;
    const nextShow = futureShows[futureShows.length - 1] ?? null;
    set({
      graph: next,
      past: [...past, graph],
      pastPresets: [...pastPresets, preset],
      futurePresets: futurePresets.slice(0, -1),
      pastShows: [...pastShows, show],
      futureShows: futureShows.slice(0, -1),
      pastStarts: [...pastStarts, startParams],
      futureStarts: futureStarts.slice(0, -1),
      startParams: futureStarts[futureStarts.length - 1] ?? null,
      preset: nextPreset,
      show: nextShow,
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
      pastShows: [],
      futureShows: [],
      pastStarts: [],
      futureStarts: [],
      startParams: null,
      preset: null,
      show: null,
      promptDismissed: false,
      canUndo: false,
      canRedo: false,
      error: null,
    });
  },
  clearError: () => set({ error: null }),
});
