import { useSyncExternalStore } from 'react';
import { loadBundled } from '../topbar/PresetMenu.js';
import { presetTitle } from '../presets/index.js';
import { useStore } from '../state/store.js';
import { parseRoute, routePreset, routeStep, type Route } from './lectureRoute.js';

function subscribe(cb: () => void): () => void {
  window.addEventListener('hashchange', cb);
  return () => window.removeEventListener('hashchange', cb);
}
const snapshot = (): string => window.location.hash;

/** The current hash route, live on `hashchange` (back and forward included). */
export function useRoute(): Route {
  return parseRoute(useSyncExternalStore(subscribe, snapshot, () => ''));
}

/** True when the hash is a lecture stage route (start-up then skips `?preset=`). */
export function hashIsStage(hash: string): boolean {
  return parseRoute(hash).kind === 'stage';
}

/**
 * Keeps the URL and the loaded preset in step (CONTRACTS section 12). At start and on every
 * hashchange to a stage route the stage is loaded (one undo step, as the Presets menu does) unless
 * it is already the loaded preset. When something else later replaces the stage (another preset,
 * a file, Undo) the hash is cleared with `replaceState` (no history entry, no hashchange). The
 * landing page and the free canvas load nothing. Returns a stop function.
 */
export function startRouting(): () => void {
  let armed = false;
  const step = (mayLoad: boolean) => {
    const route = parseRoute(window.location.hash);
    const name = routePreset(route);
    const title = useStore.getState().preset?.title ?? null;
    const r = routeStep(route, name !== null && title === presetTitle(name), armed);
    armed = r.armed;
    if (r.clear) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    } else if (r.load && mayLoad && name) {
      loadBundled(name);
      // A preset on a real dataset loads asynchronously: it arms once its title shows up.
      armed = (useStore.getState().preset?.title ?? null) === presetTitle(name);
    }
  };
  step(true);
  const onHash = () => step(true);
  window.addEventListener('hashchange', onHash);
  const unsub = useStore.subscribe(() => step(false));
  return () => {
    window.removeEventListener('hashchange', onHash);
    unsub();
  };
}
