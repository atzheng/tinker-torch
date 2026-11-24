/**
 * Hash routes (CONTRACTS.md section 12). Pure: parse and format only.
 *   `#/lectures`               the landing page
 *   `#/<lecture>/stage<k>`     one lecture stage, e.g. `#/lecture1/stage2`, `#/bias-variance/stage1`
 * Anything else (empty, `#x=...` share links, unknown lecture or stage) is the free canvas.
 */
import { LECTURES, stagePresetName, type LectureInfo } from '../presets/lectures.js';

export type Route =
  { kind: 'free' } | { kind: 'lectures' } | { kind: 'stage'; lecture: string; stage: number };

export const LANDING_HASH = '#/lectures';

export function parseRoute(hash: string, lectures: LectureInfo[] = LECTURES): Route {
  const h = hash.startsWith('#') ? hash.slice(1) : hash;
  if (h === '/lectures' || h === '/lectures/') return { kind: 'lectures' };
  const m = /^\/([a-z0-9-]+)\/stage(\d+)\/?$/.exec(h);
  if (!m) return { kind: 'free' };
  const lecture = m[1] as string;
  const stage = Number(m[2]);
  const info = lectures.find((l) => l.slug === lecture);
  if (!info || !info.stages.includes(stage)) return { kind: 'free' };
  return { kind: 'stage', lecture, stage };
}

export function formatRoute(route: Route): string {
  if (route.kind === 'lectures') return LANDING_HASH;
  if (route.kind === 'stage') return `#/${route.lecture}/stage${route.stage}`;
  return '';
}

export function routePreset(route: Route): string | null {
  return route.kind === 'stage' ? stagePresetName(route.lecture, route.stage) : null;
}

/** The previous and next stage of the same lecture (null at either end). */
export function stageNeighbours(
  route: Route,
  lectures: LectureInfo[] = LECTURES,
): { prev: Route | null; next: Route | null } {
  if (route.kind !== 'stage') return { prev: null, next: null };
  const info = lectures.find((l) => l.slug === route.lecture);
  if (!info) return { prev: null, next: null };
  const i = info.stages.indexOf(route.stage);
  const at = (k: number | undefined): Route | null =>
    k === undefined ? null : { kind: 'stage', lecture: info.slug, stage: k };
  return { prev: i > 0 ? at(info.stages[i - 1]) : null, next: at(info.stages[i + 1]) };
}

/**
 * What the router does when the hash or the loaded preset changes. `matches` says whether the
 * loaded preset is the route's stage; `armed` is true once that stage has been loaded under this
 * route. A stage route whose stage is not loaded (and not armed) loads it; once armed, a mismatch
 * (another preset or file, or Undo past the load) clears the hash instead, so the URL never names
 * a stage that is not open. Other routes reset `armed`.
 */
export function routeStep(
  route: Route,
  matches: boolean,
  armed: boolean,
): { load: boolean; clear: boolean; armed: boolean } {
  if (route.kind !== 'stage') return { load: false, clear: false, armed: false };
  if (matches) return { load: false, clear: false, armed: true };
  if (armed) return { load: false, clear: true, armed: false };
  return { load: true, clear: false, armed: false };
}
