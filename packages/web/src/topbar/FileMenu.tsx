import { useRef, useState } from 'react';
import {
  paramsFromJson,
  paramsToJson,
  parseFileText,
  serializeFile,
  type Preset,
  type SaveFile,
} from '@tinker/core';
import { useStore } from '../state/store.js';
import styles from './FileMenu.module.css';

export const SAVE_FILE_NAME = 'tinker-model.tinker.json';

/** Text of the file Save would write; weights only when asked for and available. */
export function saveText(withWeights: boolean): string {
  const { graph, host, preset } = useStore.getState();
  const text = serializeFile(graph, withWeights && host ? paramsToJson(host) : undefined);
  if (!preset) return text;
  // A preset that is being edited stays a preset: title, prompt and locked ids are written back.
  const doc = JSON.parse(text) as Record<string, unknown>;
  doc['title'] = preset.title;
  if (preset.prompt !== undefined) doc['prompt'] = preset.prompt;
  if (preset.locked.length > 0) doc['locked'] = preset.locked;
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
  const s = useStore.getState();
  if ('title' in file) {
    const ids = new Set(file.graph.nodes.map((n) => n.id));
    const active: { title: string; prompt?: string; locked: string[] } = {
      title: file.title,
      locked: (file.locked ?? []).filter((id) => ids.has(id)),
    };
    if (file.prompt !== undefined) active.prompt = file.prompt;
    s.setPreset(active);
  } else {
    s.setPreset(null);
  }
  // The file's seed drives init of unsaved layers, batch order and dropout; one undo step.
  s.setRunSeed(file.graph.train.seed);
  s.apply({ ok: true, graph: file.graph });
  if (!file.params) return { ok: true, message: null };
  const after = useStore.getState();
  if (!after.analysis.trainable) {
    return { ok: true, message: 'The saved weights were not loaded because the graph has issues.' };
  }
  const { params, dropped } = paramsFromJson(file.params, file.graph, after.analysis);
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
      <button type="button" data-testid="file-load" onClick={() => input.current?.click()}>
        Load
      </button>
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
