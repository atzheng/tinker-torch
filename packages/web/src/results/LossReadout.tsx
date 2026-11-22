import type { LossMode } from '@tinker/core';
import { useUiLabel } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import { isMiniBatch } from '../state/trainSlice.js';
import { useShallow } from 'zustand/react/shallow';
import { currentLossMode, shownLossModes } from '../state/lossMode.js';
import { lossReadout } from './readoutModel.js';
import styles from './Results.module.css';

/** Large current train loss with the validation loss beside it, above the loss plot. */
function ModeButton({ m }: { m: LossMode }) {
  const label = useUiLabel(`loss-mode-${m}`, m);
  const mode = useStore((s) => currentLossMode(s));
  const setLossMode = useStore((s) => s.setLossMode);
  return (
    <button
      type="button"
      className={styles.readoutMode}
      aria-pressed={mode === m}
      aria-label={label}
      title={label}
      data-testid={`loss-mode-${m}`}
      onClick={() => setLossMode(m)}
    >
      {label}
    </button>
  );
}

export function LossReadout() {
  // Selectors return strings, so this re-renders only when a shown value changes.
  const text = useStore(
    useShallow((s) => {
      const r = lossReadout(
        s.history,
        isMiniBatch(s.graph.train.batch, s.dataset.nTrain),
        s.diverged,
        {
          live: s.liveLoss,
          stats: s.exampleStats,
          mode: currentLossMode(s),
          nTrain: s.dataset.nTrain,
          nVal: s.dataset.nVal,
        },
      );
      return [r.train, r.val] as const;
    }),
  );
  const [train, val] = text;
  const modes = useStore(useShallow((s) => shownLossModes(s)));
  const title = useUiLabel('loss-readout', '');
  const diverged = useStore((s) => s.diverged);
  const probIssue = useStore((s) => s.probIssue);
  return (
    <div
      className={styles.readout}
      data-testid="loss-readout"
      role={title ? 'group' : undefined}
      aria-label={title || undefined}
    >
      {title && (
        <span className={styles.readoutLabel} data-testid="loss-readout-title">
          {title}
        </span>
      )}
      {modes.length > 1 && (
        <span className={styles.readoutModes} role="group" aria-label="Score">
          {modes.map((m) => (
            <ModeButton key={m} m={m} />
          ))}
        </span>
      )}
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
