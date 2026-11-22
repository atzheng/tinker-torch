import type { StateCreator } from 'zustand';
import type { HostParams, NodeId, Show } from '@tinker/core';

/** What a loaded preset leaves in the store (CONTRACTS.md section 12). */
export type ActivePreset = {
  title: string;
  prompt?: string;
  instructions?: string[];
  /** Label overrides by UI id; read with `useUiLabel`. */
  labels?: Record<string, string>;
  /** Locked piece ids that exist in the loaded graph (unknown ids in the file are ignored). */
  locked: NodeId[];
};

export type PresetSlice = {
  /**
   * The preset of the current graph, or null. Its `locked` list is what every edit refuses to
   * change. It travels with the graph in the undo history (`graphSlice`): a file load sets it
   * together with the graph (`apply(edit, preset)`), undo and redo restore it, a plain file or
   * `loadGraph` (new graph) clears it.
   */
  preset: ActivePreset | null;
  /**
   * What the loaded preset shows (CONTRACTS.md section 12), or null: everything. Preset-level
   * state, not part of the undo history: only a file load (`apply(edit, preset, show)`) or a new
   * graph (`loadGraph`) changes it.
   */
  show: Show | null;
  /**
   * The parameter values a loaded file or preset carried (possibly for some nodes only), or null.
   * Reset returns to them (nodes without an entry are drawn from the seed). Travels with the
   * graph in the undo history (`pastStarts`); cleared by a new graph (`loadGraph`) and by loading
   * a file without params. Set by the file loader after `apply`.
   */
  startParams: HostParams | null;
  /** True once the person dismissed the prompt line of the current preset. */
  promptDismissed: boolean;
  setPreset: (preset: ActivePreset | null) => void;
  setStartParams: (start: HostParams | null) => void;
  dismissPrompt: () => void;
};

export const createPresetSlice: StateCreator<PresetSlice, [], [], PresetSlice> = (set) => ({
  preset: null,
  show: null,
  startParams: null,
  promptDismissed: false,
  setStartParams: (startParams) => set({ startParams }),
  setPreset: (preset) => set({ preset, promptDismissed: false }),
  dismissPrompt: () => set({ promptDismissed: true }),
});
