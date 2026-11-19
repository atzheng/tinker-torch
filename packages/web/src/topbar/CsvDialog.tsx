import { useMemo, useState } from 'react';
import { buildCsv, inferKind, parseCsv, type Csv, type SetKind } from '@tinker/core';
import { useStore } from '../state/store.js';
import { Modal } from './Modal.js';
import { fetchCsvText, readCsvFile, type Loaded } from './csvSource.js';
import styles from './CsvDialog.module.css';

const KINDS: { kind: SetKind; label: string }[] = [
  { kind: 'regression', label: 'Regression (a number)' },
  { kind: 'binary', label: 'Binary classes' },
  { kind: 'multiclass', label: 'Multi-class' },
];

/** The default target: the last column the file allows as one. */
function defaultTarget(csv: Csv): number {
  for (let i = csv.columns.length - 1; i >= 0; i--) if (inferKind(csv, i) !== null) return i;
  return csv.columns.length - 1;
}

/**
 * Open a CSV from a URL or a local file (SPEC "Data"). Nothing is fetched until the Fetch button
 * is pressed; a local file is read in the browser. Then: pick the target, the input columns and
 * the kind (inferred, overridable), and open it as the `csv` dataset.
 */
export function CsvDialog({ onClose }: { onClose: () => void }) {
  const openCsv = useStore((s) => s.openCsv);
  const [url, setUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [source, setSource] = useState<{ csv: Csv; name: string } | null>(null);
  const [target, setTarget] = useState(0);
  const [inputs, setInputs] = useState<number[]>([]);
  const [kind, setKind] = useState<SetKind>('regression');

  const accept = (l: Loaded) => {
    if (!l.ok) {
      setLoadError(l.error);
      setSource(null);
      return;
    }
    const p = parseCsv(l.text);
    if (!p.ok) {
      setLoadError(p.error);
      setSource(null);
      return;
    }
    setLoadError(null);
    setSource({ csv: p.csv, name: l.name });
    const t = defaultTarget(p.csv);
    setTarget(t);
    setInputs(p.csv.columns.flatMap((c, i) => (i !== t && c.numeric ? [i] : [])));
    setKind(inferKind(p.csv, t) ?? 'multiclass');
  };

  const built = useMemo(
    () => (source ? buildCsv(source.csv, { target, inputs, kind }) : null),
    [source, target, inputs, kind],
  );

  const pickTarget = (t: number) => {
    if (!source) return;
    setTarget(t);
    setInputs(source.csv.columns.flatMap((c, i) => (i !== t && c.numeric ? [i] : [])));
    setKind(inferKind(source.csv, t) ?? 'multiclass');
  };

  return (
    <Modal title="Open a CSV file" testId="csv-dialog" onClose={onClose}>
      <p className="tk-muted">
        The first row must be the column names. Nothing is fetched until you press Fetch; a file you
        choose here stays in your browser.
      </p>
      <div className={styles.row}>
        <input
          type="url"
          placeholder="https://example.org/data.csv"
          aria-label="Address of a CSV file"
          data-testid="csv-url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.currentTarget.blur();
          }}
        />
        <button
          type="button"
          data-testid="csv-fetch"
          disabled={busy || url.trim() === ''}
          onClick={async () => {
            setBusy(true);
            accept(await fetchCsvText(url));
            setBusy(false);
          }}
        >
          {busy ? 'Fetching…' : 'Fetch'}
        </button>
      </div>
      <label className={styles.row}>
        <span>or a file</span>
        <input
          type="file"
          accept=".csv,.tsv,.txt,text/csv,text/plain"
          data-testid="csv-file"
          onChange={async (e) => {
            const f = e.currentTarget.files?.[0];
            if (f) accept(await readCsvFile(f));
          }}
        />
      </label>
      {loadError && (
        <p className={styles.error} role="alert" data-testid="csv-error">
          {loadError}
        </p>
      )}
      {source && (
        <>
          <p data-testid="csv-summary">
            {source.name}: {source.csv.rows.length} rows, {source.csv.header.length} columns.
          </p>
          <label className={styles.row}>
            <span>Target</span>
            <select
              data-testid="csv-target"
              value={target}
              onChange={(e) => pickTarget(Number(e.target.value))}
            >
              {source.csv.header.map((h, i) => (
                <option key={i} value={i}>
                  {h}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.row}>
            <span>Kind</span>
            <select
              data-testid="csv-kind"
              value={kind}
              onChange={(e) => setKind(e.target.value as SetKind)}
            >
              {KINDS.map((k) => (
                <option key={k.kind} value={k.kind}>
                  {k.label}
                </option>
              ))}
            </select>
          </label>
          <fieldset className={styles.inputs}>
            <legend>Input columns</legend>
            {source.csv.header.map((h, i) =>
              i === target ? null : (
                <label key={i} className={styles.check}>
                  <input
                    type="checkbox"
                    data-testid={`csv-input-${i}`}
                    disabled={!source.csv.columns[i]?.numeric}
                    checked={inputs.includes(i) && !!source.csv.columns[i]?.numeric}
                    onChange={(e) =>
                      setInputs(
                        e.target.checked
                          ? [...inputs, i].sort((a, b) => a - b)
                          : inputs.filter((j) => j !== i),
                      )
                    }
                  />
                  <span>
                    {h}
                    {source.csv.columns[i]?.numeric ? '' : ' (text, left out)'}
                  </span>
                </label>
              ),
            )}
          </fieldset>
          {built && !built.ok && (
            <p className={styles.error} role="alert" data-testid="csv-build-error">
              {built.error}
            </p>
          )}
          {built?.ok && (
            <>
              {built.notes.map((n) => (
                <p key={n} className="tk-muted" data-testid="csv-note">
                  {n}
                </p>
              ))}
              <p className="tk-muted">
                {built.table.y.length} rows, {built.spec.d} inputs
                {built.spec.classNames ? `, classes: ${built.spec.classNames.join(', ')}` : ''}.
              </p>
            </>
          )}
          <div className={styles.row}>
            <button
              type="button"
              data-testid="csv-open"
              disabled={!built?.ok}
              onClick={() => {
                if (!built?.ok || !source) return;
                openCsv(built, source.name);
                onClose();
              }}
            >
              Open
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
