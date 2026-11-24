/** Pure placement of the piece inspector. It never covers the target; see `fitInspector`. */

/** `x0`..`x1` (optional) is the target's horizontal extent, which allows a placement beside it. */
export type InspectorTarget = { cx: number; top: number; bottom: number; x0?: number; x1?: number };
export type InspectorFit = { left: number; top: number; maxHeight: number };

/** Height assumed until the inspector has been measured. */
export const INSPECTOR_GUESS_H = 230;
const EDGE = 4;
const MIN_SCROLL_H = 96;

/**
 * Where to put the inspector (`width` x `height` px, the measured natural size) for a target
 * spanning `top`..`bottom` px at horizontal centre `cx` inside a `view`.
 *
 * Rule: below the target if the whole panel fits there, else above it if it fits there, else on
 * whichever side has more room, with `maxHeight` set to that room so the panel scrolls inside
 * itself. The panel never overlaps the `top`..`bottom` band, so it cannot catch a tap meant for the
 * piece. Nothing goes above `minTop` (the canvas header strip holds buttons). Horizontally it is centred on the target and kept inside the view.
 * When neither side has `MIN_SCROLL_H` px and the target's x extent is known, the panel goes beside
 * it (right, then left) over the full height between `minTop` and the view bottom; failing that it
 * takes the larger side's room even if small (`maxHeight` 0 at worst). Covering the target is never
 * an option.
 */
export function fitInspector(
  target: InspectorTarget,
  view: { w: number; h: number },
  width: number,
  height = INSPECTOR_GUESS_H,
  gap = 16,
  minTop = EDGE,
): InspectorFit {
  const left = Math.min(
    Math.max(EDGE, target.cx - width / 2),
    Math.max(EDGE, view.w - width - EDGE),
  );
  const roomBelow = view.h - (target.bottom + gap) - EDGE;
  const roomAbove = target.top - gap - minTop;
  if (height <= roomBelow) return { left, top: target.bottom + gap, maxHeight: roomBelow };
  if (height <= roomAbove) return { left, top: target.top - gap - height, maxHeight: roomAbove };
  const below = roomBelow >= roomAbove;
  const room = Math.max(below ? roomBelow : roomAbove, 0);
  if (room < MIN_SCROLL_H && target.x0 !== undefined && target.x1 !== undefined) {
    const full = Math.max(view.h - EDGE - minTop, 0);
    const h = Math.min(height, full);
    const top = Math.min(Math.max(target.top, minTop), minTop + full - h);
    if (view.w - (target.x1 + gap) - EDGE >= width) {
      return { left: target.x1 + gap, top, maxHeight: full };
    }
    if (target.x0 - gap - EDGE >= width)
      return { left: target.x0 - gap - width, top, maxHeight: full };
  }
  const h = Math.min(height, room);
  return below
    ? { left, top: target.bottom + gap, maxHeight: room }
    : { left, top: target.top - gap - h, maxHeight: room };
}
