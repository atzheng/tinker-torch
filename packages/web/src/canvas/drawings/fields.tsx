/** Small inspector form controls shared by the piece `Fields`. Every change is one core edit. */
import { useEffect, useState, type ReactNode } from 'react';
import {
  setLinearOut,
  setPenalty,
  setSetting,
  type EditResult,
  type NodeId,
  type Penalty,
} from '@tinker/core';
import { useStore } from '../../state/store.js';

function run(make: (g: ReturnType<typeof useStore.getState>['graph']) => EditResult): void {
  const { graph, apply } = useStore.getState();
  apply(make(graph));
}

export const editSetting = (id: NodeId, key: string, value: unknown) =>
  run((g) => setSetting(g, id, key, value));
export const editOut = (id: NodeId, out: number) => run((g) => setLinearOut(g, id, out));
export const editPenalty = (id: NodeId, p: Penalty | null) => run((g) => setPenalty(g, id, p));

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="tk-field">
      <span className="tk-field-label">{label}</span>
      {children}
    </label>
  );
}

export function SelectField<V extends string>(props: {
  label: string;
  testid: string;
  value: V;
  options: readonly (readonly [V, string])[];
  onChange: (v: V) => void;
}) {
  return (
    <Field label={props.label}>
      <select
        data-testid={props.testid}
        value={props.value}
        onChange={(e) => {
          const hit = props.options.find(([v]) => v === e.target.value);
          if (hit) props.onChange(hit[0]);
        }}
      >
        {props.options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function CheckField(props: {
  label: string;
  testid: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="tk-field tk-check">
      <input
        type="checkbox"
        data-testid={props.testid}
        style={{ accentColor: 'var(--accent)' }}
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      <span>{props.label}</span>
    </label>
  );
}

/**
 * A number box that keeps what is typed in a draft and commits on blur or Enter (one undo step per
 * committed edit, like the data controls). Escape reverts the draft; an empty or unparsable draft
 * reverts; a value equal to the current one makes no edit. The draft resyncs whenever the store
 * value changes (undo, redo, another edit), even while the box has focus. Refused values (out of
 * range) show the store's error toast and the box shows the stored value again.
 */
export function NumberField(props: {
  label: string;
  testid: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onCommit: (v: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  useEffect(() => setDraft(null), [props.value]);
  const commit = () => {
    if (draft === null) return;
    setDraft(null);
    const v = Number(draft);
    if (draft.trim() !== '' && Number.isFinite(v) && v !== props.value) props.onCommit(v);
  };
  return (
    <Field label={props.label}>
      <input
        type="number"
        data-testid={props.testid}
        value={draft ?? String(props.value)}
        min={props.min}
        max={props.max}
        step={props.step}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit();
          } else if (e.key === 'Escape') {
            e.stopPropagation();
            setDraft(null);
          }
        }}
      />
    </Field>
  );
}
