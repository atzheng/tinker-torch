import { predictionChain } from '../plot/histogramPure.js';
import { plotIndex } from './plotPrediction.js';
import { shownSurfaceParams } from '../plot/lossSurface.js';
import { useStore } from './store.js';
import { PANEL_IDS, panelListed, panelOnTab } from '../results/panelLayout.js';
import {
  effectiveTab,
  isShown,
  shownTabs,
  tabsSplit,
  UI_IDS,
  type UiRegion,
  type ViewTab,
} from '@tinker/core';

/** Is the control or panel `id` shown by the loaded preset? Hidden ones are not rendered at all. */
export function useUiShown(id: string): boolean {
  return useStore((s) => isShown(s.show?.ui, id));
}

/** The tab being drawn: the one asked for (`viewTab`), or a shown one when the preset hides it. */
export function useViewTab(): ViewTab {
  return useStore((s) => effectiveTab(s.viewTab, s.show?.ui));
}

/** True when both tabs are shown; with one, that tab draws the whole graph and every panel and piece. */
export function useSplitTabs(): boolean {
  return useStore((s) => tabsSplit(s.show?.ui));
}

/** The tabs the preset leaves visible, in tab order (a new array each call: compare by `join`). */
export function useShownTabs(): ViewTab[] {
  const key = useStore((s) => shownTabs(s.show?.ui).join(','));
  return key.split(',') as ViewTab[];
}

/**
 * The label of control or panel `id`: the loaded preset's override (CONTRACTS.md section 12), else
 * `fallback`. Plain text. The override travels with the preset, so undo of a load restores it.
 */
export function useUiLabel(id: string, fallback: string): string {
  return useStore((s) => s.preset?.labels?.[id] ?? fallback);
}

/**
 * A panel toggle's state (CONTRACTS.md section 12, `options`): the student's change, else the
 * preset's initial value, else false. A toggle hidden by `show.ui` is fixed to the preset's value.
 * Held in the store so a remount keeps it.
 */
export function useUiOption(id: string): [boolean, (v: boolean) => void] {
  const epoch = useStore((s) => s.loadEpoch);
  const value = useStore((s) => {
    const init = s.preset?.options?.[id] === true;
    const e = s.optionEdits;
    if (!isShown(s.show?.ui, id) || e === null || e.preset !== s.preset || e.epoch !== epoch) {
      return init;
    }
    return e.values[id] ?? init;
  });
  const setEdit = useStore((s) => s.setOptionEdit);
  return [value, (v) => setEdit(id, v, epoch)];
}

/** Is the region on screen: at least one of its ids is shown. */
export function useRegionShown(
  region: Exclude<UiRegion, 'panel-controls' | 'canvas' | 'tabs'>,
): boolean {
  return useStore((s) => {
    if (region === 'results') {
      // Only the panels that actually render count: the loss surface needs at least two parameters.
      const surface = s.host
        ? shownSurfaceParams(s.host, s.analysis.order, s.show?.ui) !== null
        : false;
      const params = s.host !== null && s.analysis.paramNodes.length > 0;
      const pred = s.graph.predictions[plotIndex(s.graph, s.plotPrediction)];
      const histogram =
        pred !== undefined &&
        s.dataset.info.kind !== 'multiclass' &&
        !predictionChain(s.graph, pred.split('.')[0] ?? pred).hasInput;
      const tab = effectiveTab(s.viewTab, s.show?.ui);
      return PANEL_IDS.map((p) => [p, `panel-${p}`] as const).some(
        ([p, id]) =>
          panelOnTab(p, tab, tabsSplit(s.show?.ui)) &&
          panelListed(s.show?.ui, p) &&
          (id !== 'panel-surface' || surface) &&
          (id !== 'panel-histogram' || histogram) &&
          ((id !== 'panel-loss-1d' && id !== 'panel-step-log') ||
            (params && s.show?.ui?.includes(id) === true)) &&
          (id !== 'panel-examples' || s.show?.ui?.includes(id) === true) &&
          (id !== 'panel-class-bars' ||
            (s.dataset.info.kind !== 'regression' && s.show?.ui?.includes(id) === true)),
      );
    }
    return UI_IDS[region].some((id) => isShown(s.show?.ui, id));
  });
}

/** Non-hook form for event handlers (keyboard shortcuts). */
export function uiShown(id: string): boolean {
  return isShown(useStore.getState().show?.ui, id);
}
