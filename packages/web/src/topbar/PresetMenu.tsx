import { useEffect, useRef, useState } from 'react';
import { bundledPreset, PRESET_NAMES, presetTitle } from '../presets/index.js';
import { loadParsed } from './FileMenu.js';
import styles from './PresetMenu.module.css';

/** Loads a bundled preset by name (one undo step); false for an unknown name. */
export function loadBundled(name: string): boolean {
  const file = bundledPreset(name);
  if (!file) return false;
  loadParsed(file);
  return true;
}

/** Presets menu in the top bar: one button per bundled preset (SPEC "Save and presets"). */
export function PresetMenu() {
  const [left, setLeft] = useState<number | null>(null);
  const open = left !== null;
  const root = useRef<HTMLDivElement>(null);
  // While open: a press outside the menu, Escape, a scroll (the top bar scrolls on narrow
  // screens) or a resize closes it. The listeners are removed when it closes.
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
    document.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);
  return (
    <div ref={root} className={styles.row} data-testid="preset-menu">
      <button
        type="button"
        className={styles.button}
        data-testid="presets-button"
        aria-haspopup="true"
        aria-expanded={open}
        onClick={(e) => setLeft(open ? null : e.currentTarget.getBoundingClientRect().left)}
      >
        Presets
      </button>
      {open && (
        <ul className={styles.list} style={{ left }} data-testid="presets-list">
          {PRESET_NAMES.map((name) => (
            <li key={name}>
              <button
                type="button"
                className={styles.item}
                data-testid={`preset-${name}`}
                onClick={() => {
                  loadBundled(name);
                  setLeft(null);
                }}
              >
                {presetTitle(name)}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
