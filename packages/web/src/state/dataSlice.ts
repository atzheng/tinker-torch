import type { StateCreator } from 'zustand';
import {
  makeRng,
  sample,
  setColumns,
  type DataConfig,
  type DataInfo,
  type Dataset,
  type Graph,
} from '@tinker/core';
import { initialGraph } from './graphSlice.js';
import type { Store } from './store.js';

/**
 * Data slice (tinker-27m.5). The dataset is always `sample(graph.data)`: it is recomputed whenever
 * `graph.data` changes identity (a data action, undo, redo, load), by the subscription in `store.ts`.
 * Consumers (fit plot, engine driver) only read `dataset`; no other slice stores sampled rows.
 *
 *   dataset  - xTrain / xVal are stored standardised when graph.data.standardize is on; `domain`
 *              stays in raw units; `dataset.info` is the DataInfo that `analyse()` needs.
 *   dataInfo - same object as `dataset.info`.
 */
export type DataSlice = {
  dataset: Dataset;
  dataInfo: DataInfo;
  /**
   * Patch `graph.data` (clamped to the schema ranges). One undo step per call (through
   * `graphSlice.apply`; no step when nothing changes), because history holds whole graphs
   * including `graph.data`. When the set changes to one with different columns, the Input pieces' columns
   * become all columns of the new set; otherwise they are kept.
   */
  setData: (patch: Partial<DataConfig>) => void;
  /** New seed derived from the current one through core/rng (no Math.random), then re-sample. */
  resample: () => void;
  /** Set which columns an Input piece uses (kept in dataset order, at least one). */
  setInputColumns: (id: string, columns: string[]) => void;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function sanitizeData(d: DataConfig): DataConfig {
  const seed = Number.isFinite(d.seed) ? Math.trunc(d.seed) : 0;
  return {
    ...d,
    n: Math.round(clamp(Number.isFinite(d.n) ? d.n : 40, 10, 500)),
    noise: clamp(Number.isFinite(d.noise) ? d.noise : 0, 0, 1),
    val: clamp(Number.isFinite(d.val) ? d.val : 0, 0, 0.5),
    seed: clamp(seed, -2147483648, 4294967295),
  };
}

export const createDataSlice: StateCreator<Store, [], [], DataSlice> = (set, get) => {
  // get() is not populated yet while slices are being created.
  const initial = sample(initialGraph().data);
  return {
    dataset: initial,
    dataInfo: initial.info,
    setData: (patch) => {
      const { graph, dataInfo } = get();
      const data = sanitizeData({ ...graph.data, ...patch });
      let next: Graph = { ...graph, data };
      if (data.set !== graph.data.set) {
        const info = sample({ ...data, n: 10 }).info;
        if (info.columns.join() !== dataInfo.columns.join()) {
          // Sequential core edits, applied once below, so the data change and the resets are one undo step.
          const locked = get().preset?.locked ?? [];
          for (const n of graph.nodes) {
            if (n.type !== 'input') continue;
            const r = setColumns(next, n.id, info.columns, { locked });
            if (!r.ok) {
              get().apply(r);
              return;
            }
            next = r.graph;
          }
        }
      }
      if (JSON.stringify(next) === JSON.stringify(graph)) return;
      get().apply({ ok: true, graph: next });
    },
    resample: () => {
      const { graph, setData } = get();
      setData({
        seed: makeRng(graph.data.seed)
          .fork('resample')
          .int(2 ** 31),
      });
    },
    setInputColumns: (id, columns) => {
      const { graph, dataInfo } = get();
      const keep = dataInfo.columns.filter((c) => columns.includes(c));
      if (keep.length === 0) return;
      const locked = get().preset?.locked ?? [];
      get().apply(setColumns(graph, id, keep, { locked }));
    },
  };
};
