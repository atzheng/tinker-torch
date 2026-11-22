import { useRef, useState } from 'react';
import {
  hasTable,
  isRealSet,
  loadRealSet,
  paramsFromJson,
  paramsToJson,
  parseFileText,
  serializeFile,
  type Preset,
  type SaveFile,
} from '@tinker/core';
import type { ActivePreset } from '../state/presetSlice.js';
import { useStore } from '../state/store.js';
import { Popover } from './Popover.js';
import styles from './FileMenu.module.css';

export const SAVE_FILE_NAME = 'tinker-model.tinker.json';

/** Text of the file Save would write; weights only when asked for and available. */
export function saveText(withWeights: boolean): string {
  const { graph, host, preset } = useStore.getState();
  const text = serializeFile(graph, withWeights && host ? paramsToJson(host) : undefined);
  if (!preset) return text;
  // A preset that is being edited stays a preset: title, prompt, locked ids and the disclosure lists are written back.
  const doc = JSON.parse(text) as Record<string, unknown>;
  doc['title'] = preset.title;
  if (preset.prompt !== undefined) doc['prompt'] = preset.prompt;
  if (preset.instructions !== undefined) doc['instructions'] = preset.instructions;
  if (preset.checks !== undefined) doc['checks'] = preset.checks;
  if (preset.locked.length > 0) doc['locked'] = preset.locked;
  const { show } = useStore.getState();
  if (show) doc['show'] = show;
  return JSON.stringify(doc, null, 2) + '\n';
}

/**
 * Loads the text of a file into the store. Returns the notice to show: an error sentence (the
 * current graph is left alone) or, when saved weights did not fit, a plain sentence about it.
 */
export function loadText(text: string): { ok: boolean; message: string | null } {
  const parsed = parseFileText(text);
  if (!parsed.ok) return { ok: false, message: parsed.error };
  return loadParsed(parsed.file);
}

/**
 * Loads a parsed file (also a bundled preset). A preset's title, prompt and locked ids are kept in
 * the store (locked ids the graph does not have are ignored); a plain file clears any preset.
 */
export function loadParsed(file: SaveFile | Preset): { ok: boolean; message: string | null } {
  const set = file.graph.data.set;
  if (isRealSet(set) && !hasTable(set)) {
    if (set === 'csv') {
      return {
        ok: false,
        message: 'This file uses a CSV that is not open. Open the CSV from the Data menu first.',
      };
    }
    // A bundled dataset: load its rows (from the app's own files), then load the file.
    loadRealSet(set).then(
      () => loadParsed(file),
      () =>
        useStore.setState({
          error: 'The dataset could not be loaded. Reload the page and try again.',
        }),
    );
    return { ok: true, message: null };
  }
  const s = useStore.getState();
  let active: ActivePreset | null = null;
  if ('title' in file) {
    const ids = new Set(file.graph.nodes.map((n) => n.id));
    active = {
      title: file.title,
      locked: (file.locked ?? []).filter((id) => ids.has(id)),
    };
    if (file.prompt !== undefined) active.prompt = file.prompt;
    if (file.instructions !== undefined) active.instructions = file.instructions;
    if (file.checks !== undefined) active.checks = file.checks;
  }
  // The file's seed drives init of unsaved layers, batch order and dropout; one undo step.
  s.setRunSeed(file.graph.train.seed);
  s.apply({ ok: true, graph: file.graph }, active, 'show' in file ? (file.show ?? null) : null);
  if (!file.params) return { ok: true, message: null };
  const after = useStore.getState();
  if (!after.analysis.trainable) {
    return { ok: true, message: 'The saved weights were not loaded because the graph has issues.' };
  }
  const { params, dropped } = paramsFromJson(file.params, after.analysis);
  // Reset returns to these values (nodes without an entry are drawn from the seed).
  after.setStartParams(Object.keys(params).length > 0 ? params : null);
  if (!after.loadHost(params)) {
    return {
      ok: true,
      message: 'The saved weights were not loaded because the model could not be built.',
    };
  }
  if (dropped.length === 0) return { ok: true, message: null };
  const shown = dropped.slice(0, 3).map((d) => `"${d}"`);
  const names =
    dropped.length > 3 ? `${shown.join(', ')} and ${dropped.length - 3} more` : shown.join(', ');
  return {
    ok: true,
    message: `Saved weights for ${names} did not fit this graph and were left out; those layers start fresh.`,
  };
}

/** Downloads `text` as a file through a Blob and an object URL, which is revoked afterwards. */
function download(text: string, name: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Save and Load in the top bar (SPEC "Save and presets"). */
export function FileMenu() {
  const hasHost = useStore((s) => s.host !== null);
  const [withWeights, setWithWeights] = useState(true);
  const [note, setNote] = useState<{ ok: boolean; message: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function onFile(file: File | undefined): Promise<void> {
    if (!file) return;
    let text: string;
    try {
      text = await file.text();
    } catch {
      setNote({ ok: false, message: 'This file could not be read.' });
      return;
    }
    const r = loadText(text);
    setNote(r.message ? { ok: r.ok, message: r.message } : null);
  }

  return (
    <div className={styles.row} data-testid="file-menu">
      <Popover label="File" ariaLabel="File" buttonTestId="file-button" panelTestId="file-panel">
        {(close) => (
          <>
            <button
              type="button"
              data-testid="file-save"
              onClick={() => {
                download(saveText(withWeights), SAVE_FILE_NAME);
                setNote(null);
              }}
            >
              Save
            </button>
            <label className={styles.check}>
              <input
                type="checkbox"
                data-testid="file-with-weights"
                checked={withWeights && hasHost}
                disabled={!hasHost}
                onChange={(e) => setWithWeights(e.target.checked)}
              />
              <span>Weights</span>
            </label>
            <button
              type="button"
              data-testid="file-load"
              onClick={() => {
                input.current?.click();
                close();
              }}
            >
              Load
            </button>
          </>
        )}
      </Popover>
      <input
        ref={input}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = ''; // the same file can be chosen again
          void onFile(f);
        }}
      />
      {note && (
        <div
          role={note.ok ? 'status' : 'alert'}
          data-testid="file-note"
          className={note.ok ? styles.note : styles.error}
        >
          <span>{note.message}</span>
          <button type="button" aria-label="Dismiss" onClick={() => setNote(null)}>
            ×
          </button>
        </div>
      )}
    </div>
  );
}
