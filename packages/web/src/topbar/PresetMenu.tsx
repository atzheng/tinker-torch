import { useState } from 'react';
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
  return (
    <div className={styles.row} data-testid="preset-menu">
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
