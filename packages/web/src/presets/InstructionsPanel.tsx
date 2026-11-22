import { useState } from 'react';
import { useUiShown } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import { TexText } from '../tex/Tex.js';
import styles from './InstructionsPanel.module.css';

/**
 * The loaded preset's numbered exercise steps, verbatim, in a collapsible panel at the top of the
 * canvas region. The preset's prompt (if any) is the title line; collapsed, the header shows the
 * step count. Nothing shows when the preset has no instructions. Collapse state is per viewer and
 * resets to expanded when a preset is loaded (the panel is keyed by the preset object).
 */
export function InstructionsPanel() {
  const preset = useStore((s) => s.preset);
  const shown = useUiShown('instructions');
  const steps = preset?.instructions;
  if (!preset || !steps || steps.length === 0 || !shown) return null;
  return <Panel key={preset.title} title={preset.prompt} steps={steps} />;
}

function Panel({ title, steps }: { title: string | undefined; steps: string[] }) {
  const [open, setOpen] = useState(true);
  return (
    <section className={styles.panel} data-testid="instructions" aria-label="Instructions">
      <button
        type="button"
        className={styles.header}
        aria-expanded={open}
        data-testid="instructions-toggle"
        onClick={() => setOpen(!open)}
      >
        <span className={styles.caret} aria-hidden="true">
          {open ? '▾' : '▸'}
        </span>
        <span className={styles.title}>{title ? <TexText>{title}</TexText> : 'Instructions'}</span>
        <span className={styles.count}>
          {steps.length} {steps.length === 1 ? 'step' : 'steps'}
        </span>
      </button>
      {open && (
        <ol className={styles.steps} data-testid="instructions-steps">
          {steps.map((step, i) => (
            // One slot per step: a later success-check bead can add a tick beside the text.
            <li key={i} className={styles.step} data-step={i}>
              <span className={styles.stepText}>
                <TexText>{step}</TexText>
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
