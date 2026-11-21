import { useStore } from '../state/store.js';
import { isMiniBatch } from '../state/trainSlice.js';
import { lossReadout } from './readoutModel.js';
import styles from './Results.module.css';

/** Large current train loss with the validation loss beside it, above the loss plot. */
export function LossReadout() {
  // Selectors return strings and booleans, so this re-renders only when a shown value changes.
  const train = useStore(
    (s) =>
      lossReadout(s.history, isMiniBatch(s.graph.train.batch, s.dataset.nTrain), s.diverged).train,
  );
  const val = useStore(
    (s) =>
      lossReadout(s.history, isMiniBatch(s.graph.train.batch, s.dataset.nTrain), s.diverged).val,
  );
  const diverged = useStore((s) => s.diverged);
  const probIssue = useStore((s) => s.probIssue);
  return (
    <div className={styles.readout} data-testid="loss-readout">
      <span className={styles.readoutGroup}>
        <span className={styles.readoutLabel}>train</span>
        {diverged ? (
          <span className={`${styles.readoutDiverged} tk-issue`} data-testid="loss-readout-train">
            diverged
          </span>
        ) : probIssue ? (
          <span
            className={`${styles.readoutProb} tk-issue`}
            role="status"
            data-testid="loss-readout-prob"
          >
            {probIssue}
          </span>
        ) : (
          <span className={styles.readoutTrain} data-testid="loss-readout-train">
            {train}
          </span>
        )}
      </span>
      <span className={styles.readoutGroup}>
        <span className={styles.readoutLabel}>val</span>
        <span className={styles.readoutVal} data-testid="loss-readout-val">
          {val}
        </span>
      </span>
    </div>
  );
}
