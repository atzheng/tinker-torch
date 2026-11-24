import { useSyncExternalStore } from 'react';
import { loadBundled } from '../topbar/PresetMenu.js';
import { parseRoute, routePreset, type Route } from './lectureRoute.js';

function subscribe(cb: () => void): () => void {
  window.addEventListener('hashchange', cb);
  return () => window.removeEventListener('hashchange', cb);
}
const snapshot = (): string => window.location.hash;

/** The current hash route, live on `hashchange` (back and forward included). */
export function useRoute(): Route {
  return parseRoute(useSyncExternalStore(subscribe, snapshot, () => ''));
}

/**
 * Loads the stage a route names (one undo step, as the Presets menu does) at start and on every
 * hashchange to a stage route. The landing page and the free canvas load nothing, so work in
 * progress stays. Returns a stop function.
 */
export function startRouting(): () => void {
  const apply = () => {
    const name = routePreset(parseRoute(window.location.hash));
    if (name) loadBundled(name);
  };
  apply();
  window.addEventListener('hashchange', apply);
  return () => window.removeEventListener('hashchange', apply);
}
