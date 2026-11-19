import { useStore } from '../state/store.js';
import styles from './PromptLine.module.css';

/**
 * The loaded preset's prompt, verbatim, as a dismissible line above the canvas. The tool adds no
 * text of its own: nothing shows when the preset has no prompt.
 */
export function PromptLine() {
  const prompt = useStore((s) => s.preset?.prompt);
  const dismissed = useStore((s) => s.promptDismissed);
  const dismiss = useStore((s) => s.dismissPrompt);
  if (!prompt || dismissed) return null;
  return (
    <div className={styles.line} data-testid="preset-prompt" role="status">
      <span className={styles.text}>{prompt}</span>
      <button type="button" aria-label="Dismiss" data-testid="prompt-dismiss" onClick={dismiss}>
        ×
      </button>
    </div>
  );
}
