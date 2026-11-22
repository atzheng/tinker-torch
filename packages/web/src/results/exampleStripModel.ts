// Per-example strip (tinker-6t7.6): pure cell model. Log loss and cross-entropy show p of what
// happened, exp(-loss), for every example; any other loss shows the per-example loss itself.

export type StripCell = {
  /** The value shown: p (0 to 1) in 'prob' mode, the per-example loss in 'loss' mode. */
  value: number;
  /** 0 to 1: fill height or shade. */
  frac: number;
  /** Observed class (classification), or -1 for regression. */
  cls: number;
};

export type StripMode = 'prob' | 'loss';

/** Above this many examples the cells shrink and the numbers are hidden (value on hover or tap). */
export const NUMBERS_MAX = 60;

export function stripCells(
  losses: Float32Array,
  y: ArrayLike<number>,
  mode: StripMode,
  classification: boolean,
): StripCell[] {
  let max = 0;
  if (mode === 'loss') for (const l of losses) if (Number.isFinite(l) && l > max) max = l;
  const out: StripCell[] = [];
  for (let i = 0; i < losses.length; i++) {
    const l = losses[i] ?? Number.NaN;
    const cls = classification ? Math.round(y[i] ?? 0) : -1;
    if (mode === 'prob') {
      const p = Math.exp(-l);
      out.push({ value: p, frac: Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0, cls });
    } else {
      out.push({ value: l, frac: Number.isFinite(l) && max > 0 ? Math.min(1, l / max) : 0, cls });
    }
  }
  return out;
}

export function formatCell(v: number): string {
  if (!Number.isFinite(v)) return '-';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1000)) return v.toExponential(1);
  return a >= 100 ? v.toFixed(0) : v.toFixed(a >= 10 ? 1 : 2);
}
