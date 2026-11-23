import { useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { startDrag } from '../gestures/drag.js';
import styles from './ResizeHandle.module.css';

export type HandleProps = {
  /** `y`: a horizontal bar that moves up and down; `x`: a vertical bar that moves left and right. */
  axis: 'x' | 'y';
  label: string;
  testId: string;
  /** `aria-valuenow`, `-min`, `-max`: percent. */
  now: number;
  min: number;
  max: number;
  /**
   * Called at pointer down: capture what the drag needs and return the move function, which gets
   * the pointer's movement along the axis (px) since the press.
   */
  begin: () => (deltaPx: number) => void;
  /** An arrow key along the axis: `dir` is the way the bar moves (-1 up or left), `big` with Shift. */
  onStep: (dir: -1 | 1, big: boolean) => void;
  /**
   * Where the grab strip lies: `before` it (over the plot above or left of a panel boundary, never
   * over the next panel's header) or `centered` on the line.
   */
  strip?: 'before' | 'centered';
  /** Placement class from the owner's stylesheet (grid area etc.). */
  className?: string;
};

/**
 * A resize handle (tinker-9qh): a separator with a grip mark centred on it, a `row-resize` /
 * `col-resize` cursor, accent on hover, press and focus, 8 px to grab (32 px on touch), focusable,
 * and arrow keys move it in steps. It is a control (CONTRACTS.md section 10, controls vs
 * diagnostics). Pointer Events through `startDrag`.
 */
export function ResizeHandle(p: HandleProps) {
  const [dragging, setDragging] = useState(false);
  const down = (e: ReactPointerEvent<HTMLElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const move = p.begin();
    startDrag(e, {
      onStart: () => setDragging(true),
      onMove: (pt) => move(p.axis === 'x' ? pt.dx : pt.dy),
      onEnd: () => setDragging(false),
      onCancel: () => setDragging(false),
    });
  };
  const key = (e: KeyboardEvent<HTMLElement>) => {
    const keys = p.axis === 'x' ? ['ArrowLeft', 'ArrowRight'] : ['ArrowUp', 'ArrowDown'];
    const i = keys.indexOf(e.key);
    if (i < 0) return;
    e.preventDefault();
    p.onStep(i === 0 ? -1 : 1, e.shiftKey);
  };
  return (
    <div
      className={`${styles.handle} ${p.className ?? ''}`}
      data-axis={p.axis}
      data-strip={p.strip ?? 'before'}
      data-dragging={dragging}
      data-testid={p.testId}
      role="separator"
      aria-orientation={p.axis === 'x' ? 'vertical' : 'horizontal'}
      aria-label={p.label}
      aria-valuenow={p.now}
      aria-valuemin={p.min}
      aria-valuemax={p.max}
      tabIndex={0}
      onPointerDown={down}
      onKeyDown={key}
    />
  );
}
