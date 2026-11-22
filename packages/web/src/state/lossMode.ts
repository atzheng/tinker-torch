import {
  effectiveLossMode,
  isShown,
  lossModesFor,
  type ExampleStats,
  type LossMode,
} from '@tinker/core';
import type { Store } from './store.js';

/** Per-example summaries of the current parameters on each split (tinker-6t7.5); null when not available. */
export type ExampleStatsPair = { train: ExampleStats | null; val: ExampleStats | null };

/** Modes the loaded graph offers and the preset shows: [] with no Mean or no mode shown. */
export function shownLossModes(s: Pick<Store, 'graph' | 'analysis' | 'show'>): LossMode[] {
  const ui = s.show?.ui;
  return lossModesFor(s.graph, s.analysis).filter((m) => isShown(ui, `loss-mode-${m}`));
}

/** The mode the readout uses now (null: no mode is shown, so the readout shows the mean as before). */
export function currentLossMode(
  s: Pick<Store, 'graph' | 'analysis' | 'show' | 'lossMode'>,
): LossMode | null {
  return effectiveLossMode(
    lossModesFor(s.graph, s.analysis),
    (id) => isShown(s.show?.ui, id),
    s.lossMode,
  );
}
