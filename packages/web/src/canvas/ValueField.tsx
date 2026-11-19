import { useRef, useState } from 'react';
import { useStore } from '../state/store.js';
import { beginWeightEdit, type WeightTarget } from './weightEdit.js';

/**
 * Type an exact value for a weight or bias; committed on Enter or blur as one undo step, Escape
 * drops the draft. Disabled while training runs (a running loop would overwrite it).
 */
export function ValueField(p: {
  testid: string;
  label: string;
  value: number;
  target: WeightTarget;
}) {
  const running = useStore((s) => s.running);
  const [draft, setDraft] = useState<string | null>(null);
  const dropped = useRef(false); // Escape: the blur that follows must not commit
  const commit = () => {
    const text = dropped.current ? null : draft;
    dropped.current = false;
    setDraft(null);
    if (text === null) return;
    const v = Number(text);
    if (text.trim() === '' || !Number.isFinite(v)) return;
    const edit = beginWeightEdit(p.target);
    if (!edit) return;
    edit.set(v);
    edit.commit();
  };
  return (
    <label className="tk-valuefield">
      <span className="tk-muted">{p.label}</span>
      <input
        type="text"
        inputMode="decimal"
        data-testid={p.testid}
        disabled={running}
        value={draft ?? p.value.toFixed(4)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          else if (e.key === 'Escape') {
            dropped.current = true;
            setDraft(null);
            e.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}
