import { useRef, useState } from 'react';
import { useUiShown } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import { TexText } from '../tex/Tex.js';
import { Tick, useCheckResults } from './SuccessTick.js';
import { ResizeHandle } from '../layout/ResizeHandle.js';
import { REGION_LIMITS, clampSize, dragSize, percent, stepSize } from '../layout/regionSizes.js';
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
  const { steps: done, general } = useCheckResults();
  const cap = useStore((s) => s.regionSizes.instructions);
  const setSize = useStore((s) => s.setRegionSize);
  const listRef = useRef<HTMLOListElement>(null);
  const lim = REGION_LIMITS.instructions;
  // The cap on the steps list, a fraction of the window height (dragged by the handle below it).
  const current = (): number => {
    if (cap !== null) return cap;
    const h = listRef.current?.clientHeight ?? 0;
    return h > 0 ? clampSize('instructions', h / window.innerHeight) : lim.def;
  };
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
        {general && <Tick testId="success-check" />}
        <span className={styles.count}>
          {steps.length} {steps.length === 1 ? 'step' : 'steps'}
        </span>
      </button>
      {open && (
        <ol
          className={styles.steps}
          data-testid="instructions-steps"
          ref={listRef}
          style={
            cap === null ? undefined : { height: `${cap * 100}vh`, maxHeight: `${cap * 100}vh` }
          }
        >
          {steps.map((step, i) => (
            // One slot per step; a check naming this step puts its tick here.
            <li key={i} className={styles.step} data-step={i}>
              <span className={styles.stepText}>
                <TexText>{step}</TexText>
              </span>
              {done.get(i) && <Tick testId="success-check-step" />}
            </li>
          ))}
        </ol>
      )}
      {open && (
        <ResizeHandle
          axis="y"
          strip="centered"
          className={styles.resize}
          testId="resize-instructions"
          label="Resize the instructions"
          now={percent(cap ?? lim.def)}
          min={percent(lim.min)}
          max={percent(lim.max)}
          begin={() => {
            const start = current();
            return (d) =>
              setSize('instructions', dragSize('instructions', start, d, window.innerHeight, 1));
          }}
          onStep={(dir, big) =>
            setSize('instructions', stepSize('instructions', current(), dir, big, 1))
          }
        />
      )}
    </section>
  );
}
