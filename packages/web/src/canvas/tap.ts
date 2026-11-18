import type { PointerEvent as ReactPointerEvent } from 'react';
import { startDrag, type DragHandlers } from '../gestures/drag.js';

/**
 * A `pointerdown` handler for a canvas element: runs `startDrag` (capture, 4 px slop) and stops the
 * event, so the canvas background does not also see the press. `onTap` fires for a click or a
 * touch tap; later beads add `onStart`/`onMove`/`onEnd` here for drags that begin on a piece
 * (wire from a port, weight drag). Mouse buttons other than the main one are ignored.
 */
export function pointerDownWith(handlers: DragHandlers) {
  return (e: ReactPointerEvent<Element>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.stopPropagation();
    startDrag(e, handlers);
  };
}
