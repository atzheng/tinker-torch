import { predictionChain } from '../plot/histogramPure.js';
import { plotIndex } from './plotPrediction.js';
import { twoParams } from '../plot/lossSurface.js';
import { useStore } from './store.js';
import { isShown, UI_IDS, type UiRegion } from '@tinker/core';

/** Is the control or panel `id` shown by the loaded preset? Hidden ones are not rendered at all. */
export function useUiShown(id: string): boolean {
  return useStore((s) => isShown(s.show?.ui, id));
}

/** Is the region on screen: at least one of its ids is shown. */
export function useRegionShown(region: Exclude<UiRegion, 'panel-controls' | 'canvas'>): boolean {
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
      return (
        [
          'panel-fit',
          'panel-loss',
          'panel-surface',
          'panel-histogram',
          'panel-loss-1d',
          'panel-step-log',
        ] as const
      ).some(
        (id) =>
          isShown(s.show?.ui, id) &&
          (id !== 'panel-surface' || surface) &&
          (id !== 'panel-histogram' || histogram) &&
          ((id !== 'panel-loss-1d' && id !== 'panel-step-log') ||
            (params && s.show?.ui?.includes(id) === true)),
      );
    }
    return UI_IDS[region].some((id) => isShown(s.show?.ui, id));
  });
}

/** Non-hook form for event handlers (keyboard shortcuts). */
export function uiShown(id: string): boolean {
  return isShown(useStore.getState().show?.ui, id);
}
