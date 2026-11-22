import type { CategoryDial } from '@tinker/core';
import { UNIT_R } from './neuronPure.js';
import { useStore } from '../state/store.js';
import { fitDialName } from './dialLabelFit.js';
import './categoryDial.css';

/** `-0.4`, `1.2`, `12`: short enough to sit inside a unit circle. */
export function dialText(v: number): string {
  if (!Number.isFinite(v)) return '?';
  return Math.abs(v) < 10 ? v.toFixed(1) : String(Math.round(v));
}

/**
 * The inside of a categorical unit (One-hot or Lookup, neuron view): the category's value in the
 * circle when a dial sets a scalar parameter for it, and the category name under the circle. The
 * value follows `host` (`count` guards a stale shape: the table must have one row per category).
 */
export function CategoryDialMark(p: { dial: CategoryDial | undefined; count: number }) {
  const { dial } = p;
  const value = useStore((s) => {
    const t = dial?.target;
    const w = t ? s.host?.[t.node]?.['W'] : undefined;
    return t && w && w.length === p.count ? w[t.i] : undefined;
  });
  return (
    <>
      {value !== undefined ? (
        <text
          className="tk-dial-value"
          data-testid="dial-value"
          textAnchor="middle"
          dominantBaseline="central"
        >
          {dialText(value)}
        </text>
      ) : (
        <circle className="tk-dot" r={UNIT_R * 0.35} />
      )}
      {dial && (
        <text
          className="tk-dial-label"
          data-testid={`dial-label-${dial.node}-${dial.index}`}
          textAnchor="middle"
          y={UNIT_R + 9}
        >
          {fitDialName(dial.name)}
          <title>{dial.name}</title>
        </text>
      )}
    </>
  );
}
