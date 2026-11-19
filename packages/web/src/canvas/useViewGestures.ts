import { useEffect, useRef, type RefObject } from 'react';
import type { Transform } from './pure.js';
import { pinchView, zoomAbout } from './viewport.js';

export type ViewGestureOptions = {
  /** Only the neuron view pans and zooms. */
  enabled: boolean;
  /** The transform on screen now. */
  current: () => Transform;
  /** Zoom limits (scale) and the clamp that keeps the drawing reachable. */
  min: number;
  max: number;
  clamp: (t: Transform) => Transform;
  set: (t: Transform) => void;
};

/**
 * Pan and zoom by Pointer Events and the wheel, attached to the canvas region in the capture phase.
 * Two touch pointers pinch and pan together wherever they land (the second press never reaches a
 * piece; a one-finger drag that had begun is cancelled by releasing its capture). Ctrl or Cmd with
 * the wheel zooms about the cursor (a trackpad pinch arrives that way), the plain wheel pans.
 * One-finger and mouse drags on empty background pan too, see Canvas `onBackgroundDrag`.
 */
export function useViewGestures(ref: RefObject<HTMLElement | null>, o: ViewGestureOptions) {
  const opts = useRef(o);
  opts.current = o;

  useEffect(() => {
    const el = ref.current;
    if (!el || !o.enabled) return;
    const pts = new Map<number, [number, number]>();
    let pinch: {
      ids: [number, number];
      start: Transform;
      a0: [number, number];
      b0: [number, number];
    } | null = null;
    const local = (ev: { clientX: number; clientY: number }): [number, number] => {
      const r = el.getBoundingClientRect();
      return [ev.clientX - r.left, ev.clientY - r.top];
    };
    const releaseCapture = (target: EventTarget | null, id: number) => {
      for (let n = target as Element | null; n; n = n.parentElement) {
        try {
          if (n.hasPointerCapture(id)) n.releasePointerCapture(id);
        } catch {
          // Pointer already gone.
        }
      }
    };
    const isTouch = (ev: PointerEvent) => ev.pointerType === 'touch' || ev.pointerType === 'pen';

    const down = (ev: PointerEvent) => {
      if (!isTouch(ev)) return;
      pts.set(ev.pointerId, local(ev));
      if (pinch || pts.size < 2) return;
      const [first, second] = [...pts.keys()] as [number, number];
      const a0 = pts.get(first);
      const b0 = pts.get(second);
      if (!a0 || !b0) return;
      pinch = { ids: [first, second], start: opts.current.current(), a0, b0 };
      // The finger that was down first may have begun a drag on a piece: cancel it.
      for (const id of pinch.ids) {
        for (const c of el.querySelectorAll('*')) {
          if (c.hasPointerCapture?.(id)) releaseCapture(c, id);
        }
      }
      ev.stopPropagation();
      ev.preventDefault();
    };
    const move = (ev: PointerEvent) => {
      if (!pts.has(ev.pointerId)) return;
      pts.set(ev.pointerId, local(ev));
      if (!pinch || !pinch.ids.includes(ev.pointerId)) return;
      const a1 = pts.get(pinch.ids[0]);
      const b1 = pts.get(pinch.ids[1]);
      if (!a1 || !b1) return;
      ev.stopPropagation();
      const { min, max, clamp, set } = opts.current;
      set(clamp(pinchView(pinch.start, pinch.a0, pinch.b0, a1, b1, min, max)));
    };
    const up = (ev: PointerEvent) => {
      if (!pts.has(ev.pointerId)) return;
      pts.delete(ev.pointerId);
      if (pinch && pinch.ids.includes(ev.pointerId)) {
        ev.stopPropagation();
        // The gesture ends with its first finger; the other one is not a tap or a drag.
        for (const id of pinch.ids) pts.delete(id);
        pinch = null;
      }
    };
    const wheel = (ev: WheelEvent) => {
      const { min, max, clamp, set, current } = opts.current;
      ev.preventDefault();
      const t = current();
      if (ev.ctrlKey || ev.metaKey) {
        const [x, y] = local(ev);
        set(clamp(zoomAbout(t, Math.exp(-ev.deltaY * 0.01), x, y, min, max)));
      } else {
        const dx = ev.shiftKey && ev.deltaX === 0 ? ev.deltaY : ev.deltaX;
        const dy = ev.shiftKey && ev.deltaX === 0 ? 0 : ev.deltaY;
        set(clamp({ ...t, tx: t.tx - dx, ty: t.ty - dy }));
      }
    };

    el.addEventListener('pointerdown', down, true);
    el.addEventListener('pointermove', move, true);
    el.addEventListener('pointerup', up, true);
    el.addEventListener('pointercancel', up, true);
    el.addEventListener('wheel', wheel, { passive: false });
    return () => {
      el.removeEventListener('pointerdown', down, true);
      el.removeEventListener('pointermove', move, true);
      el.removeEventListener('pointerup', up, true);
      el.removeEventListener('pointercancel', up, true);
      el.removeEventListener('wheel', wheel);
    };
  }, [ref, o.enabled]);
}
