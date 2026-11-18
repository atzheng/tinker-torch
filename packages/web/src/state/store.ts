import { create } from 'zustand';
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
