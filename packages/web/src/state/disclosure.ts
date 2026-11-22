import { useStore } from './store.js';
import { isShown, UI_IDS, type UiRegion } from '@tinker/core';

/** Is the control or panel `id` shown by the loaded preset? Hidden ones are not rendered at all. */
export function useUiShown(id: string): boolean {
  return useStore((s) => isShown(s.show?.ui, id));
}

/** Is the region on screen: at least one of its ids is shown. */
export function useRegionShown(region: UiRegion): boolean {
  return useStore((s) => UI_IDS[region].some((id) => isShown(s.show?.ui, id)));
}

/** Non-hook form for event handlers (keyboard shortcuts). */
export function uiShown(id: string): boolean {
  return isShown(useStore.getState().show?.ui, id);
}
