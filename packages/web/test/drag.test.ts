// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from 'vitest';
import { startDrag, type DragHandlers } from '../src/gestures/drag.js';

function ptr(type: string, x: number, y: number, pointerId = 1): PointerEvent {
  const ev = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(ev, { pointerId, clientX: x, clientY: y });
  return ev as PointerEvent;
}

let el: HTMLElement;
let captured: Set<number>;

beforeEach(() => {
  el = document.createElement('div');
  captured = new Set();
  // jsdom has no pointer capture; record it.
  el.setPointerCapture = (id: number) => void captured.add(id);
  el.hasPointerCapture = (id: number) => captured.has(id);
  el.releasePointerCapture = (id: number) => void captured.delete(id);
});

function begin(handlers: DragHandlers, x = 100, y = 100, slop?: number) {
  return startDrag(
    { currentTarget: el, pointerId: 1, clientX: x, clientY: y },
    handlers,
    slop === undefined ? {} : { slop },
  );
}

test('captures the pointer on start and releases it at the end', () => {
  begin({});
  expect(captured.has(1)).toBe(true);
  el.dispatchEvent(ptr('pointerup', 100, 100));
  expect(captured.has(1)).toBe(false);
});

test('movement under the 4 px slop does not start a drag; release is a tap', () => {
  const h = {
    onStart: vi.fn(),
    onMove: vi.fn(),
    onEnd: vi.fn(),
    onTap: vi.fn(),
    onCancel: vi.fn(),
  };
  begin(h);
  el.dispatchEvent(ptr('pointermove', 102, 102)); // distance 2.83
  el.dispatchEvent(ptr('pointermove', 103, 100)); // distance 3
  expect(h.onStart).not.toHaveBeenCalled();
  expect(h.onMove).not.toHaveBeenCalled();
  el.dispatchEvent(ptr('pointerup', 103, 100));
  expect(h.onTap).toHaveBeenCalledOnce();
  expect(h.onEnd).not.toHaveBeenCalled();
});

test('reaching 4 px starts the drag once, then moves and ends', () => {
  const h = { onStart: vi.fn(), onMove: vi.fn(), onEnd: vi.fn(), onTap: vi.fn() };
  begin(h);
  el.dispatchEvent(ptr('pointermove', 104, 100));
  el.dispatchEvent(ptr('pointermove', 120, 110));
  expect(h.onStart).toHaveBeenCalledOnce();
  expect(h.onMove).toHaveBeenCalledTimes(2);
  expect(h.onMove.mock.lastCall?.[0]).toMatchObject({
    x: 120,
    y: 110,
    dx: 20,
    dy: 10,
    startX: 100,
  });
  // Returning inside the slop does not un-start the drag.
  el.dispatchEvent(ptr('pointermove', 100, 100));
  expect(h.onMove).toHaveBeenCalledTimes(3);
  el.dispatchEvent(ptr('pointerup', 100, 100));
  expect(h.onEnd).toHaveBeenCalledOnce();
  expect(h.onTap).not.toHaveBeenCalled();
});

test('pointercancel cancels, releases capture and stops further events', () => {
  const h = { onStart: vi.fn(), onMove: vi.fn(), onEnd: vi.fn(), onCancel: vi.fn() };
  begin(h);
  el.dispatchEvent(ptr('pointermove', 110, 100));
  el.dispatchEvent(ptr('pointercancel', 110, 100));
  expect(h.onCancel).toHaveBeenCalledOnce();
  expect(captured.has(1)).toBe(false);
  el.dispatchEvent(ptr('pointermove', 150, 100));
  el.dispatchEvent(ptr('pointerup', 150, 100));
  expect(h.onMove).toHaveBeenCalledTimes(1);
  expect(h.onEnd).not.toHaveBeenCalled();
});

test('lost capture cancels', () => {
  const h = { onCancel: vi.fn(), onEnd: vi.fn() };
  begin(h);
  el.dispatchEvent(ptr('lostpointercapture', 100, 100));
  expect(h.onCancel).toHaveBeenCalledOnce();
});

test('events from another pointer are ignored', () => {
  const h = { onStart: vi.fn(), onEnd: vi.fn(), onCancel: vi.fn() };
  begin(h);
  el.dispatchEvent(ptr('pointermove', 200, 200, 2));
  el.dispatchEvent(ptr('pointerup', 200, 200, 2));
  el.dispatchEvent(ptr('pointercancel', 200, 200, 2));
  expect(h.onStart).not.toHaveBeenCalled();
  expect(h.onEnd).not.toHaveBeenCalled();
  expect(h.onCancel).not.toHaveBeenCalled();
});

test('the returned function abandons the gesture silently', () => {
  const h = { onStart: vi.fn(), onCancel: vi.fn(), onEnd: vi.fn() };
  const stop = begin(h);
  stop();
  expect(captured.has(1)).toBe(false);
  el.dispatchEvent(ptr('pointermove', 150, 100));
  el.dispatchEvent(ptr('pointerup', 150, 100));
  expect(h.onStart).not.toHaveBeenCalled();
  expect(h.onEnd).not.toHaveBeenCalled();
  expect(h.onCancel).not.toHaveBeenCalled();
});

test('slop is configurable', () => {
  const h = { onStart: vi.fn() };
  begin(h, 0, 0, 10);
  el.dispatchEvent(ptr('pointermove', 8, 0));
  expect(h.onStart).not.toHaveBeenCalled();
  el.dispatchEvent(ptr('pointermove', 10, 0));
  expect(h.onStart).toHaveBeenCalledOnce();
});
