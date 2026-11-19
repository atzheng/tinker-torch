import { create } from 'zustand';
import { analyse, sample } from '@tinker/core';
import { neuronSelectionLive } from '../canvas/pure.js';
import { noteHost } from './hostShape.js';
import { createDataSlice, type DataSlice } from './dataSlice.js';
import { createGraphSlice, type GraphSlice } from './graphSlice.js';
import { createPresetSlice, type PresetSlice } from './presetSlice.js';
import { attachTrainLoop, createTrainSlice, type TrainSlice } from './trainSlice.js';
import { createViewSlice, type ViewSlice } from './viewSlice.js';

export type Store = GraphSlice & DataSlice & TrainSlice & ViewSlice & PresetSlice;

export const useStore = create<Store>()((...a) => ({
  ...createGraphSlice(...a),
  ...createDataSlice(...a),
  ...createTrainSlice(...a),
  ...createViewSlice(...a),
  ...createPresetSlice(...a),
}));

// The dataset follows graph.data whatever changed it (data action, undo, redo, load).
useStore.subscribe((s, prev) => {
  if (s.graph.data !== prev.graph.data) {
    const dataset = sample(s.graph.data);
    useStore.setState({ dataset, dataInfo: dataset.info });
  }
});

// After the data subscription: the train loop re-compiles on graph or dataset changes.
attachTrainLoop(useStore);

// Host snapshots remember the graph they were published with (the neuron view checks shapes).
useStore.subscribe((s, prev) => {
  if (s.host !== prev.host) noteHost(s.host, s.graph);
});

// An edge or unit selection that no longer fits the widths is cleared now, not just hidden, so
// undo or growing the layer again cannot bring it back on a different weight.
useStore.subscribe((s, prev) => {
  const sel = s.selection;
  if (!sel || (sel.kind !== 'edge' && sel.kind !== 'unit')) return;
  if (s.graph === prev.graph && s.dataInfo === prev.dataInfo) return;
  if (!neuronSelectionLive(sel, s.graph, analyse(s.graph, s.dataInfo))) s.select(null);
});
