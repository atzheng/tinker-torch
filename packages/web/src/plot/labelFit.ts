/** Ellipsis used for a shortened label. */
const ELLIPSIS = '…';
/** Characters of a label that stay readable when thinning is chosen over further truncation. */
const MIN_CHARS = 6;

/** `s` cut to at most `max` px wide (by `measure`), ending in an ellipsis when it was cut. */
export function truncateLabel(s: string, max: number, measure: (s: string) => number): string {
  if (measure(s) <= max) return s;
  if (measure(ELLIPSIS) > max) return '';
  let lo = 0;
  let hi = s.length;
  // longest prefix whose text plus the ellipsis fits
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(s.slice(0, mid).trimEnd() + ELLIPSIS) <= max) lo = mid;
    else hi = mid - 1;
  }
  return lo === 0 ? ELLIPSIS : s.slice(0, lo).trimEnd() + ELLIPSIS;
}

/**
 * Labels for a categorical axis whose categories sit `slot` px apart. Long names are truncated
 * with an ellipsis to the width available; when even a short readable prefix (about MIN_CHARS
 * characters) would not fit, every k-th label is shown (the rest are null) and the shown ones may
 * use k slots. `measure` gives the text width in px, so the rule is testable without a canvas.
 */
export function fitCategoryLabels(
  names: readonly string[],
  slot: number,
  measure: (s: string) => number,
  gap = 6,
): (string | null)[] {
  const widest = names.reduce((m, s) => Math.max(m, measure(s)), 0);
  const readable = Math.min(widest, measure('M'.repeat(MIN_CHARS)));
  let k = 1;
  while (k < names.length && k * slot - gap < readable) k++;
  const room = Math.max(0, k * slot - gap);
  return names.map((s, i) => (i % k === 0 ? truncateLabel(s, room, measure) : null));
}
