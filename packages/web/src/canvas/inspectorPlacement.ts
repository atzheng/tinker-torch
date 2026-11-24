/** Pure placement of the piece inspector. It never covers the target; see `fitInspector`. */

export type InspectorTarget = { cx: number; top: number; bottom: number };
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
 * (A side with less than `MIN_SCROLL_H` px is still used: covering the piece is the worse failure.)
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
  if (roomBelow >= roomAbove) {
    const h = Math.max(roomBelow, Math.min(MIN_SCROLL_H, height));
    return { left, top: Math.min(target.bottom + gap, view.h - EDGE - h), maxHeight: h };
  }
  const h = Math.max(roomAbove, Math.min(MIN_SCROLL_H, height));
  return { left, top: Math.max(minTop, target.top - gap - h), maxHeight: h };
}
