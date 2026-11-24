import { useEffect, useState } from 'react';
import {
  applyThemePref,
  nextThemePref,
  readThemePref,
  writeThemePref,
  type ThemePref,
} from '../themePref.js';
import styles from './DataControls.module.css';

const LABEL: Record<ThemePref, string> = { system: 'System', light: 'Light', dark: 'Dark' };
const ICON: Record<ThemePref, string> = { system: '◐', light: '☀', dark: '☾' };

/** Cycles the theme System, Light, Dark. System follows the OS preference live. Always shown. */
export function ThemeToggle() {
  const [pref, setPref] = useState<ThemePref>(readThemePref);

  useEffect(() => {
    applyThemePref(pref);
    if (pref !== 'system' || typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyThemePref('system');
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [pref]);

  const label = `Theme: ${LABEL[pref]}`;
  return (
    <div className={styles.row}>
      <button
        type="button"
        data-testid="theme-toggle"
        aria-label={label}
        title={`${label} (click to change)`}
        onClick={() => {
          const n = nextThemePref(pref);
          writeThemePref(n);
          setPref(n);
        }}
      >
        <span aria-hidden="true">{ICON[pref]}</span>
      </button>
    </div>
  );
}
