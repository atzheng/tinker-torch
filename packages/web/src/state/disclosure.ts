import { predictionChain } from '../plot/histogramPure.js';
import { plotIndex } from './plotPrediction.js';
import { twoParams } from '../plot/lossSurface.js';
import { useStore } from './store.js';
import { panelTab, type PanelId } from '../results/panelLayout.js';
import {
  effectiveTab,
  isShown,
  shownTabs,
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
      // Only the panels that actually render count: the loss surface needs a 2-parameter model.
      const surface = s.host ? twoParams(s.host, s.analysis.order) !== null : false;
      const params = s.host !== null && s.analysis.paramNodes.length > 0;
      const pred = s.graph.predictions[plotIndex(s.graph, s.plotPrediction)];
      const histogram =
        pred !== undefined &&
        s.dataset.info.kind !== 'multiclass' &&
        !predictionChain(s.graph, pred.split('.')[0] ?? pred).hasInput;
      const tab = effectiveTab(s.viewTab, s.show?.ui);
      return (
        [
          'panel-fit',
          'panel-loss',
          'panel-surface',
          'panel-histogram',
          'panel-loss-1d',
          'panel-examples',
          'panel-step-log',
        ] as const
      ).some(
        (id) =>
          panelTab(id.slice('panel-'.length) as PanelId) === tab &&
          isShown(s.show?.ui, id) &&
          (id !== 'panel-surface' || surface) &&
          (id !== 'panel-histogram' || histogram) &&
          ((id !== 'panel-loss-1d' && id !== 'panel-step-log') ||
            (params && s.show?.ui?.includes(id) === true)) &&
          (id !== 'panel-examples' || s.show?.ui?.includes(id) === true),
      );
    }
    return UI_IDS[region].some((id) => isShown(s.show?.ui, id));
  });
}

/** Non-hook form for event handlers (keyboard shortcuts). */
export function uiShown(id: string): boolean {
  return isShown(useStore.getState().show?.ui, id);
}
