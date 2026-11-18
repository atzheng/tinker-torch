import type { StateCreator } from 'zustand';

/**
 * Empty slice, owned by tinker-27m.5 (top bar data controls). Shape to fill in:
 *   TODO dataset: Dataset | null        sampled from graph.data via core/data/sample.ts
 *   TODO dataInfo: DataInfo | null      what analyse() needs (columns, classes)
 *   TODO resample(): void               new seed, re-sample, keep the graph
 *   TODO setData(patch: Partial<DataConfig>): void
 */
export type DataSlice = Record<never, never>;

export const createDataSlice: StateCreator<DataSlice, [], [], DataSlice> = () => ({});
