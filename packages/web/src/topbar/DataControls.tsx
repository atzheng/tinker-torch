import { useEffect, useState } from 'react';
import type { DatasetName } from '@tinker/core';
import { useStore } from '../state/store.js';
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
  const scale = props.scale ?? 1;
  const shown = String(Math.round(props.value * scale * 1000) / 1000);
  const [draft, setDraft] = useState(shown);
  useEffect(() => setDraft(shown), [shown]);
  return (
    <label className={styles.field}>
      <span>{props.label}</span>
      <input
        type="number"
        data-testid={props.testid}
        value={draft}
        min={props.min * scale}
        max={props.max * scale}
        step={props.step}
        onChange={(e) => {
          setDraft(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() === '' || !Number.isFinite(v)) return;
          if (props.integer && !Number.isInteger(v)) return;
          if (v < props.min * scale || v > props.max * scale) return;
          props.onCommit(v / scale);
        }}
        onBlur={() => setDraft(shown)}
      />
    </label>
  );
}

/** The dataset half of the top bar (SPEC "Data"). */
export function DataControls() {
  const data = useStore((s) => s.graph.data);
  const setData = useStore((s) => s.setData);
  const resample = useStore((s) => s.resample);
  return (
    <div className={styles.row} data-testid="data-controls">
      <label className={styles.field}>
        <span>Dataset</span>
        <select
          data-testid="data-set"
          value={data.set}
          onChange={(e) => setData({ set: e.target.value as DatasetName })}
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
        </select>
      </label>
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
      <NumField
        label="Noise"
        testid="data-noise"
        value={data.noise}
        min={0}
        max={1}
        step={0.05}
        onCommit={(noise) => setData({ noise })}
      />
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
      <button type="button" data-testid="data-resample" onClick={resample}>
        Resample
      </button>
      <label className={styles.check}>
        <input
          type="checkbox"
          data-testid="data-standardize"
          checked={data.standardize}
          onChange={(e) => setData({ standardize: e.target.checked })}
        />
        <span>Standardise</span>
      </label>
      {/* Enabled by the neuron view bead (switches layer and neuron grain). */}
      <button type="button" data-testid="grain-toggle" disabled title="Grain">
        Grain
      </button>
    </div>
  );
}
