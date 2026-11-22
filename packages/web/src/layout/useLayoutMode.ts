import { useSyncExternalStore } from 'react';
import { layoutFor, type LayoutMode } from './layoutMode.js';

const subscribe = (cb: () => void) => {
  window.addEventListener('resize', cb);
  window.addEventListener('orientationchange', cb);
  return () => {
    window.removeEventListener('resize', cb);
    window.removeEventListener('orientationchange', cb);
  };
};
const get = (): LayoutMode => layoutFor(window.innerWidth, window.innerHeight);

/** The current layout (landscape, portrait or notice), live on resize and rotation. */
export function useLayoutMode(): LayoutMode {
  return useSyncExternalStore(subscribe, get, () => 'landscape');
}
