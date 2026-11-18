import { create } from 'zustand';
import { sample } from '@tinker/core';
import { createDataSlice, type DataSlice } from './dataSlice.js';
import { createGraphSlice, type GraphSlice } from './graphSlice.js';
import { createPresetSlice, type PresetSlice } from './presetSlice.js';
import { attachTrainLoop, createTrainSlice, type TrainSlice } from './trainSlice.js';
import { createViewSlice, type ViewSlice } from './viewSlice.js';

export type Store = GraphSlice & DataSlice & TrainSlice & ViewSlice & PresetSlice;

export const useStore = create<Store>()((set, get, api) => {
  const graph = createGraphSlice(set, get, api);
  return {
    ...graph,
    // Starting a new graph also ends the preset: its locks and prompt belong to the loaded file.
    loadGraph: (g) => {
      graph.loadGraph(g);
      set({ preset: null, promptDismissed: false });
    },
    ...createDataSlice(set, get, api),
    ...createTrainSlice(set, get, api),
    ...createViewSlice(set, get, api),
    ...createPresetSlice(set, get, api),
  };
});

// The dataset follows graph.data whatever changed it (data action, undo, redo, load).
useStore.subscribe((s, prev) => {
  if (s.graph.data !== prev.graph.data) {
    const dataset = sample(s.graph.data);
    useStore.setState({ dataset, dataInfo: dataset.info });
  }
});

// After the data subscription: the train loop re-compiles on graph or dataset changes.
attachTrainLoop(useStore);
