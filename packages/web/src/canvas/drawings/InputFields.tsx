import { useStore } from '../../state/store.js';
import { featureName } from '../../state/names.js';
import type { NodeOf } from './types.js';

/** Inspector settings of an Input piece: one checkbox per dataset column (at least one stays on). */
export function InputFields({ node }: { node: NodeOf<'input'> }) {
  const info = useStore((s) => s.dataInfo);
  const columns = info.columns;
  const setInputColumns = useStore((s) => s.setInputColumns);
  const toggle = (c: string, on: boolean) =>
    setInputColumns(node.id, on ? [...node.columns, c] : node.columns.filter((x) => x !== c));
  return (
    <fieldset data-testid="input-columns" style={{ border: 0, margin: 0, padding: 0 }}>
      <legend>Columns</legend>
      {columns.map((c) => {
        const on = node.columns.includes(c);
        return (
          <label
            key={c}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginRight: 12 }}
          >
            <input
              type="checkbox"
              style={{ accentColor: 'var(--accent)' }}
              data-testid={`input-col-${c}`}
              checked={on}
              disabled={on && node.columns.length === 1}
              onChange={(e) => toggle(c, e.target.checked)}
            />
            <span>{featureName(info, c)}</span>
          </label>
        );
      })}
    </fieldset>
  );
}
