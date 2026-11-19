import { useStore } from '../state/store.js';
import styles from './DataControls.module.css';

/** Undo and redo buttons for every canvas action (Ctrl/Cmd+Z and Shift+Ctrl/Cmd+Z do the same). */
export function UndoRedo() {
  const canUndo = useStore((s) => s.canUndo);
  const canRedo = useStore((s) => s.canRedo);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  return (
    <div className={styles.row} role="group" aria-label="History">
      <button
        type="button"
        data-testid="btn-undo"
        aria-label="Undo"
        title="Undo"
        disabled={!canUndo}
        onClick={undo}
      >
        ↶
      </button>
      <button
        type="button"
        data-testid="btn-redo"
        aria-label="Redo"
        title="Redo"
        disabled={!canRedo}
        onClick={redo}
      >
        ↷
      </button>
    </div>
  );
}
