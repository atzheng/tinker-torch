import { useEffect, type ReactNode } from 'react';
import styles from './Modal.module.css';

/** A centred dialog over a dimmed page. A press on the backdrop or Escape closes it. */
export function Modal(props: {
  title: string;
  testId: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const { onClose } = props;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div
      className={styles.backdrop}
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={styles.dialog}
        role="dialog"
        aria-label={props.title}
        data-testid={props.testId}
      >
        <div className={styles.head}>
          <h2>{props.title}</h2>
          <button
            type="button"
            aria-label="Close"
            data-testid={`${props.testId}-close`}
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        <div className={styles.body}>{props.children}</div>
      </div>
    </div>
  );
}
