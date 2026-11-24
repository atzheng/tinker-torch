import { predictionChain } from '../plot/histogramPure.js';
import { plotIndex } from './plotPrediction.js';
import { shownSurfaceParams } from '../plot/lossSurface.js';
import { useStore } from './store.js';
import { PANEL_IDS, panelListed, panelOnTab } from '../results/panelLayout.js';
import {
  effectiveTab,
  isShown,
  pinRolesOf,
  shownTabs,
  tabIsWhole,
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

/** True when more than one tab is shown; with one, that tab draws the whole graph and every panel and piece. */
export function useSplitTabs(): boolean {
  return useStore((s) => tabsSplit(s.show?.ui));
}

/** True when the tab being drawn shows the whole graph: the full tab, or the only tab shown. */
export function useWholeTab(): boolean {
  return useStore((s) => tabIsWhole(effectiveTab(s.viewTab, s.show?.ui), s.show?.ui));
}

/** The pin roles the tab being drawn may set, move or remove (`pinRolesOf`), as a string key for stable selection. */
export function pinRolesNow(): ('pred' | 'loss')[] {
  const s = useStore.getState();
  const tab = effectiveTab(s.viewTab, s.show?.ui);
  return pinRolesOf(tab, tabIsWhole(tab, s.show?.ui));
}

/** Hook form of `pinRolesNow`: a new array each call when the roles change. */
export function usePinRoles(): ('pred' | 'loss')[] {
  const key = useStore((s) => {
    const tab = effectiveTab(s.viewTab, s.show?.ui);
    return pinRolesOf(tab, tabIsWhole(tab, s.show?.ui)).join(',');
  });
  return key.split(',') as ('pred' | 'loss')[];
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
 * preset's initial value, else `dflt` (false unless the caller says otherwise). A toggle hidden by `show.ui` is fixed to the preset's value.
 * Held in the store so a remount keeps it.
 */
export function useUiOption(id: string, dflt = false): [boolean, (v: boolean) => void] {
  const epoch = useStore((s) => s.loadEpoch);
  const value = useStore((s) => {
    const o = s.preset?.options?.[id];
    const init = typeof o === 'boolean' ? o : dflt;
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
          panelOnTab(p, tab, tabsSplit(s.show?.ui), s.show?.panelTabs) &&
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
