import { create } from 'zustand';
import { sample } from '@tinker/core';
import { createDataSlice, type DataSlice } from './dataSlice.js';
import { createGraphSlice, type GraphSlice } from './graphSlice.js';
import { createTrainSlice, type TrainSlice } from './trainSlice.js';
import { createViewSlice, type ViewSlice } from './viewSlice.js';

export type Store = GraphSlice & DataSlice & TrainSlice & ViewSlice;

export const useStore = create<Store>()((...a) => ({
  ...createGraphSlice(...a),
  ...createDataSlice(...a),
  ...createTrainSlice(...a),
  ...createViewSlice(...a),
}));

// The dataset follows graph.data whatever changed it (data action, undo, redo, load).
useStore.subscribe((s, prev) => {
  if (s.graph.data !== prev.graph.data) {
    const dataset = sample(s.graph.data);
    useStore.setState({ dataset, dataInfo: dataset.info });
  }
});
