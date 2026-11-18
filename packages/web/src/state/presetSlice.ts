import type { StateCreator } from 'zustand';
import type { NodeId } from '@tinker/core';

/** What a loaded preset leaves in the store (CONTRACTS.md section 12). */
export type ActivePreset = {
  title: string;
  prompt?: string;
  /** Locked piece ids that exist in the loaded graph (unknown ids in the file are ignored). */
  locked: NodeId[];
};

export type PresetSlice = {
  /**
   * The preset that was last loaded, or null. Its `locked` list is what every edit refuses to change.
   * It belongs to the load, not to the graph history: undo and redo keep it (a locked id that is no
   * longer in the graph simply guards nothing), and it is cleared when a file without preset
   * fields is loaded or `loadGraph` starts a new graph.
   */
  preset: ActivePreset | null;
  /** True once the person dismissed the prompt line of the current preset. */
  promptDismissed: boolean;
  setPreset: (preset: ActivePreset | null) => void;
  dismissPrompt: () => void;
};

export const createPresetSlice: StateCreator<PresetSlice, [], [], PresetSlice> = (set) => ({
  preset: null,
  promptDismissed: false,
  setPreset: (preset) => set({ preset, promptDismissed: false }),
  dismissPrompt: () => set({ promptDismissed: true }),
});
