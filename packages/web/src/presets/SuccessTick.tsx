import { useMemo } from 'react';
import { evaluateChecks } from '@tinker/core';
import { useUiShown } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import styles from './SuccessTick.module.css';

/**
 * Results of the loaded preset's success checks right now (no latch): per instructions step, and
 * for the checks without a step, whether all of them hold (null: there are none). Empty while the
 * `success-check` id is hidden.
 */
export function useCheckResults(): { steps: Map<number, boolean>; general: boolean | null } {
  const checks = useStore((s) => s.preset?.checks);
  const liveLoss = useStore((s) => s.liveLoss);
  const host = useStore((s) => s.host);
  const shown = useUiShown('success-check');
  return useMemo(
    () =>
      shown && checks && checks.length > 0
        ? evaluateChecks(checks, { liveLoss, host })
        : { steps: new Map<number, boolean>(), general: null },
    [shown, checks, liveLoss, host],
  );
}

/** The tick itself: present exactly while met; the label is for assistive technology only. */
export function Tick({ testId }: { testId: string }) {
  return (
    <span className={styles.tick} role="img" aria-label="Done" data-testid={testId}>
      ✓
    </span>
  );
}

/**
 * The tick for checks without a step when the preset has no instructions panel to carry it: a thin
 * row at the top of the canvas region. With instructions it shows in the panel header instead.
 */
export function SuccessTick() {
  const hasInstructions = useStore((s) => (s.preset?.instructions?.length ?? 0) > 0);
  const { general } = useCheckResults();
  if (hasInstructions || !general) return null;
  return (
    <div className={styles.row}>
      <Tick testId="success-check" />
    </div>
  );
}
