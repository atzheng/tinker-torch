import { useEffect, useRef, useState, type ReactNode } from 'react';
import styles from './Popover.module.css';

/**
 * A top bar button that opens a panel under it. The panel is always mounted (hidden while closed)
 * so its controls keep their test ids; it is fixed-positioned so the bar's clipping never hides it.
 * A press outside, Escape, or a resize closes it. `children` may be a function of `close`.
 */
export function Popover(props: {
  label: ReactNode;
  ariaLabel: string;
  buttonTestId: string;
  panelTestId: string;
  children: ReactNode | ((close: () => void) => ReactNode);
}) {
  const [left, setLeft] = useState<number | null>(null);
  const open = left !== null;
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = () => setLeft(null);
    const onDown = (e: PointerEvent) => {
      if (e.target instanceof Node && root.current?.contains(e.target)) return;
      close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', close);
    };
  }, [open]);
  const close = () => setLeft(null);
  return (
    <div ref={root} className={styles.root}>
      <button
        type="button"
        className={styles.button}
        data-testid={props.buttonTestId}
        aria-label={props.ariaLabel}
        aria-haspopup="true"
        aria-expanded={open}
        onClick={(e) =>
          setLeft(open ? null : Math.max(4, e.currentTarget.getBoundingClientRect().left))
        }
      >
        {props.label} <span aria-hidden="true">▾</span>
      </button>
      <div
        className={styles.panel}
        style={{ left: left ?? 0 }}
        hidden={!open}
        data-testid={props.panelTestId}
      >
        {typeof props.children === 'function' ? props.children(close) : props.children}
      </div>
    </div>
  );
}
