/**
 * Pointer Events drag helper (CONTRACTS.md section 10). Every gesture uses it: palette drag,
 * wire drag, edge weight drag, unit +/-, sliders. No HTML5 drag and drop.
 *
 * Call `startDrag` from a `pointerdown` handler. The helper captures the pointer on the element,
 * waits for 4 px of movement (the slop) before the drag starts, and cancels on `pointercancel`
 * or when capture is lost.
 */

export const DRAG_SLOP_PX = 4;

export type DragPoint = {
  /** Current pointer position (client coordinates). */
  x: number;
  y: number;
  /** Where the pointer went down. */
  startX: number;
  startY: number;
  /** Movement since pointer down. */
  dx: number;
  dy: number;
  event: PointerEvent;
};

export type DragHandlers = {
  /** Once, when movement first reaches the slop. `p` is the position at that moment. */
  onStart?: (p: DragPoint) => void;
  /** On every pointermove after the drag started, including the one that started it. */
  onMove?: (p: DragPoint) => void;
  /** On pointerup after the drag started. */
  onEnd?: (p: DragPoint) => void;
  /** On pointerup when the pointer never left the slop: a tap or click. */
  onTap?: (p: DragPoint) => void;
  /** On pointercancel or lost capture, whether or not the drag had started. */
  onCancel?: () => void;
};

export type DragStartEvent = {
  currentTarget: EventTarget & Element;
  pointerId: number;
  clientX: number;
  clientY: number;
};

/**
 * Begin tracking a pointer. Returns a function that abandons the gesture without calling any
 * handler (use it on unmount). A second pointer going down is ignored; only `pointerId` counts.
 */
export function startDrag(
  e: DragStartEvent,
  handlers: DragHandlers,
  opts: { slop?: number } = {},
): () => void {
  const el = e.currentTarget;
  const pointerId = e.pointerId;
  const slop = opts.slop ?? DRAG_SLOP_PX;
  const startX = e.clientX;
  const startY = e.clientY;
  let started = false;
  let done = false;

  try {
    el.setPointerCapture(pointerId);
  } catch {
    // Not capturable (synthetic event, or the pointer is already gone): carry on uncaptured.
  }

  const point = (ev: PointerEvent): DragPoint => ({
    x: ev.clientX,
    y: ev.clientY,
    startX,
    startY,
    dx: ev.clientX - startX,
    dy: ev.clientY - startY,
    event: ev,
  });

  const cleanup = () => {
    done = true;
    el.removeEventListener('pointermove', onMove as EventListener);
    el.removeEventListener('pointerup', onUp as EventListener);
    el.removeEventListener('pointercancel', onCancel as EventListener);
    el.removeEventListener('lostpointercapture', onCancel as EventListener);
    try {
      if (el.hasPointerCapture(pointerId)) el.releasePointerCapture(pointerId);
    } catch {
      // Already released.
    }
  };

  function onMove(ev: PointerEvent) {
    if (done || ev.pointerId !== pointerId) return;
    const p = point(ev);
    if (!started) {
      if (Math.hypot(p.dx, p.dy) < slop) return;
      started = true;
      handlers.onStart?.(p);
    }
    handlers.onMove?.(p);
  }

  function onUp(ev: PointerEvent) {
    if (done || ev.pointerId !== pointerId) return;
    const p = point(ev);
    cleanup();
    if (started) handlers.onEnd?.(p);
    else handlers.onTap?.(p);
  }

  function onCancel(ev: Event) {
    if (done || (ev as PointerEvent).pointerId !== pointerId) return;
    cleanup();
    handlers.onCancel?.();
  }

  el.addEventListener('pointermove', onMove as EventListener);
  el.addEventListener('pointerup', onUp as EventListener);
  el.addEventListener('pointercancel', onCancel as EventListener);
  el.addEventListener('lostpointercapture', onCancel as EventListener);

  return cleanup;
}
