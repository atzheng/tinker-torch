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

/**
 * `scale` is the loss that gets the darkest shade (loss mode); the caller keeps it growing so an
 * improving fit visibly lightens. A non-finite loss always gets the worst shade.
 */
export function stripCells(
  losses: Float32Array,
  y: ArrayLike<number>,
  mode: StripMode,
  classification: boolean,
  scale = 0,
): StripCell[] {
  const out: StripCell[] = [];
  // No scale given: shade against the largest finite loss of this set.
  const s = scale > 0 ? scale : maxFinite(losses);
  for (let i = 0; i < losses.length; i++) {
    const l = losses[i] ?? Number.NaN;
    const cls = classification ? Math.round(y[i] ?? 0) : -1;
    if (mode === 'prob') {
      const p = Math.exp(-l);
      out.push({ value: p, frac: Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0, cls });
    } else {
      const frac = !Number.isFinite(l) ? 1 : s > 0 ? Math.min(1, l / s) : 0;
      out.push({ value: l, frac, cls });
    }
  }
  return out;
}

/** The largest finite loss. */
export function maxFinite(losses: Float32Array): number {
  let m = 0;
  for (const l of losses) if (Number.isFinite(l) && l > m) m = l;
  return m;
}

/** At most 5 characters, so it fits a 34 px cell: '0', '<.01', '0.49', '12.3', '1234', '1e5'. */
export function formatCell(v: number): string {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 0.01) return v < 0 ? '-<.01' : '<.01';
  if (a < 10) return v.toFixed(2);
  if (a < 100) return v.toFixed(1);
  if (a < 10000) return v.toFixed(0);
  return v.toExponential(0).replace('e+', 'e');
}
