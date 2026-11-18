import type { StateCreator } from 'zustand';

/**
 * Empty slice, owned by tinker-27m.2 (training loop). Shape to fill in:
 *   TODO model: Model | null            from engine.compile(graph); null when the graph is invalid
 *   TODO state: TrainState | null       engine state; arrays never stored here (host copies only)
 *   TODO running: boolean               the requestAnimationFrame loop is active
 *   TODO diverged: boolean              a non-finite loss stopped the run
 *   TODO history: { step: number; train: number; val: number }[]   loss curves
 *   TODO best: { step: number; val: number; params: HostParams } | null
 *   TODO speed: number                  steps per frame
 *   TODO step(), run(), pause(), reset(), restoreBest()
 */
export type TrainSlice = Record<never, never>;

export const createTrainSlice: StateCreator<TrainSlice, [], [], TrainSlice> = () => ({});
