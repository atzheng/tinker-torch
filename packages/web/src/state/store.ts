import { create } from 'zustand';
import {
  analyse,
  embedCsv,
  getTable,
  hasTable,
  isRealSet,
  loadRealSet,
  registerEmbeddedCsv,
  sample,
  type DataConfig,
  type EmbeddedCsv,
} from '@tinker/core';
import { neuronSelectionLive } from '../canvas/pure.js';
import { noteHost } from './hostShape.js';
import { createDataSlice, type DataSlice } from './dataSlice.js';
import { createGraphSlice, type GraphSlice } from './graphSlice.js';
import { createPresetSlice, type PresetSlice } from './presetSlice.js';
import { attachTrainLoop, createTrainSlice, type TrainSlice } from './trainSlice.js';
import { attachWeightHistory } from './weightHistory.js';
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
function refreshDataset(data: DataConfig): void {
  if (isRealSet(data.set) && !hasTable(data.set)) {
    // A graph (file load, undo) names a real set whose rows are not here. The old dataset stays
    // until they are: a bundled set loads from the app's own files; a CSV has to be opened again.
    if (data.set === 'csv') {
      useStore.getState().setData({ set: 'line' });
      useStore.setState({
        error: 'This file used a CSV that is not open. Open it again from the Data menu.',
      });
      return;
    }
    loadRealSet(data.set).then(
      () => {
        if (useStore.getState().graph.data.set === data.set)
          refreshDataset(useStore.getState().graph.data);
      },
      () =>
        useStore.setState({
          error: 'The dataset could not be loaded. Reload the page and try again.',
        }),
    );
    return;
  }
  const dataset = sample(data);
  useStore.setState({ dataset, dataInfo: dataset.info });
}

// The CSV is a session global outside the undo history, so each graph on the csv set remembers
// the CSV it was made with, and undo, redo or any switch to that graph puts the same rows back.
const csvOfGraph = new WeakMap<object, { csv: EmbeddedCsv; name: string | null }>();
function syncCsv(s: Store, prev: Store): void {
  if (s.graph === prev.graph || s.graph.data.set !== 'csv') return;
  const known = csvOfGraph.get(s.graph);
  if (!known) {
    const csv = embedCsv(s.csvName);
    if (csv) csvOfGraph.set(s.graph, { csv, name: s.csvName });
    return;
  }
  if (getTable('csv')?.x === known.csv.x) return;
  registerEmbeddedCsv(known.csv);
  useStore.setState({ csvName: known.name });
}

useStore.subscribe((s, prev) => {
  syncCsv(s, prev);
  if (s.graph.data !== prev.graph.data) refreshDataset(useStore.getState().graph.data);
});

// After the data subscription: the train loop re-compiles on graph or dataset changes.
attachTrainLoop(useStore);
// After the train loop, so a resync has finished before undo or redo writes hand-set weights back.
attachWeightHistory(useStore);

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
