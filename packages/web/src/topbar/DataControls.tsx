import { useEffect, useState } from 'react';
import { BUNDLED, REAL_META, SETS, isRealSet, type DatasetName } from '@tinker/core';
import { useUiLabel, useUiShown } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import { CsvDialog } from './CsvDialog.js';
import { Popover } from './Popover.js';
import { SourcesDialog } from './SourcesDialog.js';
import styles from './DataControls.module.css';

const GROUPS: { label: string; sets: { name: DatasetName; label: string }[] }[] = [
  {
    label: '1-D regression',
    sets: [
      { name: 'line', label: 'Line' },
      { name: 'quadratic', label: 'Quadratic' },
      { name: 'sine', label: 'Sine' },
      { name: 'step', label: 'Step' },
      { name: 'wiggle', label: 'Wiggle' },
    ],
  },
  {
    label: '2-D regression',
    sets: [
      { name: 'plane', label: 'Plane' },
      { name: 'bowl', label: 'Bowl' },
      { name: 'saddle', label: 'Saddle' },
    ],
  },
  {
    label: '2-D classification',
    sets: [
      { name: 'blobs', label: 'Two blobs' },
      { name: 'circle', label: 'Circle' },
      { name: 'xor', label: 'XOR' },
      { name: 'moons', label: 'Two moons' },
    ],
  },
  {
    label: '2-D multi-class',
    sets: [
      { name: 'blobs3', label: 'Three blobs' },
      { name: 'spiral3', label: 'Three spirals' },
    ],
  },
  {
    label: 'Delivery story',
    sets: [
      { name: 'delivery_late', label: 'Delivery: late or not' },
      { name: 'delivery_time', label: 'Delivery: time (min)' },
    ],
  },
  {
    label: 'Classic datasets',
    sets: BUNDLED.map((name) => ({ name, label: REAL_META[name].title })),
  },
];

/**
 * Number input that keeps what the user types as a draft and commits it on every valid change,
 * so clearing the field to type a new value does not snap back. `scale` maps display to stored.
 */
function NumField(props: {
  label: string;
  testid: string;
  value: number;
  min: number;
  max: number;
  step: number;
  integer?: boolean;
  scale?: number;
  onCommit: (v: number) => void;
}) {
  const label = useUiLabel(props.testid, props.label); // the testid is the UI id
  const scale = props.scale ?? 1;
  const shown = String(Math.round(props.value * scale * 1000) / 1000);
  const [draft, setDraft] = useState(shown);
  useEffect(() => setDraft(shown), [shown]);
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <input
        type="number"
        data-testid={props.testid}
        aria-label={label}
        value={draft}
        min={props.min * scale}
        max={props.max * scale}
        step={props.step}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
        }}
        onBlur={() => {
          // One undo step per committed edit: commit on blur or Enter, not on every keystroke.
          const v = Number(draft);
          const ok =
            draft.trim() !== '' &&
            Number.isFinite(v) &&
            (!props.integer || Number.isInteger(v)) &&
            v >= props.min * scale &&
            v <= props.max * scale;
          if (ok) props.onCommit(v / scale);
          else setDraft(shown);
        }}
      />
    </label>
  );
}

/** The dataset half of the top bar (SPEC "Data"). */
export function DataControls() {
  const data = useStore((s) => s.graph.data);
  const setData = useStore((s) => s.setData);
  const resample = useStore((s) => s.resample);
  const chooseSet = useStore((s) => s.chooseSet);
  const csvName = useStore((s) => s.csvName);
  const size = useStore((s) => s.dataset.nTrain + s.dataset.nVal);
  const [dialog, setDialog] = useState<'csv' | 'sources' | null>(null);
  const real = isRealSet(data.set);
  const lbl = {
    set: useUiLabel('data-set', 'Dataset'),
    n: useUiLabel('data-n', 'n'),
    standardize: useUiLabel('data-standardize', 'Standardise'),
    csv: useUiLabel('data-open-csv', 'Open CSV…'),
    sources: useUiLabel('data-sources', 'Sources'),
    resample: useUiLabel('data-resample', 'Resample'),
  };
  const sh = {
    set: useUiShown('data-set'),
    n: useUiShown('data-n'),
    noise: useUiShown('data-noise'),
    val: useUiShown('data-val'),
    seed: useUiShown('data-seed'),
    standardize: useUiShown('data-standardize'),
    csv: useUiShown('data-open-csv'),
    sources: useUiShown('data-sources'),
    resample: useUiShown('data-resample'),
  };
  // The Data popover has a button only while one of its controls is shown.
  const anyMore =
    (sh.noise && !real && !SETS[data.set].noiseUnused) ||
    sh.val ||
    sh.seed ||
    sh.standardize ||
    sh.csv ||
    sh.sources;
  return (
    <div className={styles.row} data-testid="data-controls">
      {sh.set && (
        <label className={styles.field}>
          <span>{lbl.set}</span>
          <select
            data-testid="data-set"
            aria-label={lbl.set}
            value={data.set}
            onChange={(e) => void chooseSet(e.target.value as DatasetName)}
          >
            {GROUPS.map((g) => (
              <optgroup key={g.label} label={g.label}>
                {g.sets.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.label}
                  </option>
                ))}
              </optgroup>
            ))}
            {csvName !== null && (
              <optgroup label="Your file">
                <option value="csv">{csvName}</option>
              </optgroup>
            )}
          </select>
        </label>
      )}
      {sh.n &&
        (real ? (
          // A real dataset has a fixed number of rows: n shows it and cannot be changed.
          <label className={styles.field}>
            <span>{lbl.n}</span>
            <input
              type="number"
              aria-label={lbl.n}
              data-testid="data-n"
              value={size}
              disabled
              readOnly
            />
          </label>
        ) : (
          <NumField
            label="n"
            testid="data-n"
            value={data.n}
            min={10}
            max={500}
            step={10}
            integer
            onCommit={(n) => setData({ n })}
          />
        ))}
      {anyMore && (
        <Popover
          label="Data"
          ariaLabel="More data settings"
          buttonTestId="data-more"
          panelTestId="data-more-panel"
        >
          {(close) => (
            <>
              {!real && !SETS[data.set].noiseUnused && sh.noise && (
                <NumField
                  label="Noise"
                  testid="data-noise"
                  value={data.noise}
                  min={0}
                  max={1}
                  step={0.05}
                  onCommit={(noise) => setData({ noise })}
                />
              )}
              {sh.val && (
                <NumField
                  label="Val %"
                  testid="data-val"
                  value={data.val}
                  min={0}
                  max={0.5}
                  step={5}
                  scale={100}
                  onCommit={(val) => setData({ val })}
                />
              )}
              {sh.seed && (
                <NumField
                  label="Seed"
                  testid="data-seed"
                  value={data.seed}
                  min={-2147483648}
                  max={4294967295}
                  step={1}
                  integer
                  onCommit={(seed) => setData({ seed })}
                />
              )}
              {sh.standardize && (
                <label className={styles.check}>
                  <input
                    type="checkbox"
                    data-testid="data-standardize"
                    aria-label={lbl.standardize}
                    checked={data.standardize}
                    onChange={(e) => setData({ standardize: e.target.checked })}
                  />
                  <span>{lbl.standardize}</span>
                </label>
              )}
              {sh.csv && (
                <button
                  type="button"
                  data-testid="data-open-csv"
                  aria-label={lbl.csv}
                  onClick={() => {
                    close();
                    setDialog('csv');
                  }}
                >
                  {lbl.csv}
                </button>
              )}
              {sh.sources && (
                <button
                  type="button"
                  data-testid="data-sources"
                  aria-label={lbl.sources}
                  onClick={() => {
                    close();
                    setDialog('sources');
                  }}
                >
                  {lbl.sources}
                </button>
              )}
            </>
          )}
        </Popover>
      )}
      {sh.resample && (
        <button
          type="button"
          data-testid="data-resample"
          aria-label={lbl.resample}
          onClick={resample}
        >
          {lbl.resample}
        </button>
      )}
      {dialog === 'csv' && <CsvDialog onClose={() => setDialog(null)} />}
      {dialog === 'sources' && <SourcesDialog onClose={() => setDialog(null)} />}
    </div>
  );
}
