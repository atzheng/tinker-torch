/** Pure helpers for plot axes: ticks, linear and log scales, number formatting. No DOM. */

export type Scale = {
  /** Domain value to pixel. */
  px: (v: number) => number;
  lo: number;
  hi: number;
  log: boolean;
};

/** Linear scale from [lo, hi] to [p0, p1] (p1 may be smaller than p0 for a y axis). */
export function linearScale(lo: number, hi: number, p0: number, p1: number): Scale {
  const span = hi - lo || 1;
  return { px: (v) => p0 + ((v - lo) / span) * (p1 - p0), lo, hi, log: false };
}

/** Log10 scale; values at or below zero map to the p0 end (callers clamp or skip them first). */
export function logScale(lo: number, hi: number, p0: number, p1: number): Scale {
  const a = Math.log10(lo);
  const span = Math.log10(hi) - a || 1;
  return {
    px: (v) => (v > 0 ? p0 + ((Math.log10(v) - a) / span) * (p1 - p0) : p0),
    lo,
    hi,
    log: true,
  };
}

/** "Nice" tick values covering [lo, hi] with about `count` ticks (1, 2, 5 times a power of ten). */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) return [];
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi === lo) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
  const out: number[] = [];
  const first = Math.ceil(lo / step - 1e-9);
  const last = Math.floor(hi / step + 1e-9);
  for (let i = first; i <= last; i++) out.push(Number((i * step).toPrecision(12)));
  return out;
}

/**
 * Log ticks inside [lo, hi] (both positive): every power of ten, plus 2 and 5 times it when the
 * range spans three decades or fewer. Falls back to the two ends when nothing fits.
 */
export function logTicks(lo: number, hi: number): number[] {
  if (!(lo > 0 && hi > lo)) return [];
  const decades = Math.log10(hi / lo);
  const mants = decades > 3 ? [1] : [1, 2, 5];
  const every = Math.max(1, Math.ceil(decades / 6)); // thin to about 6 labels
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    if (e % every !== 0) continue;
    for (const m of mants) {
      const v = Number((m * 10 ** e).toPrecision(12));
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
    }
  }
  return out.length >= 2 ? out : [lo, hi];
}

/** Short label for a tick value. */
export function formatTick(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-3) return v.toExponential(0).replace('e+', 'e');
  return String(Number(v.toPrecision(3)));
}

/** Range [lo, hi] of finite values from `a` and `b`, padded by a fraction on each side. */
export function paddedRange(lo: number, hi: number, frac = 0.08): [number, number] {
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) return [0, 1];
  if (hi === lo) return [lo - 1, hi + 1];
  const p = (hi - lo) * frac;
  return [lo - p, hi + p];
}

/**
 * Draws a polyline of (x, y) pairs at pixel resolution: points are bucketed by pixel column of x
 * and each bucket emits its first, lowest, highest and last y in order, so peaks survive. Calls
 * `emit(px, py)` without allocating. Pairs with a non-finite or (when `skip` says so) skipped y
 * are left out. `n` pairs are read through `getX(i)`, `getY(i)` (NaN means no point).
 */
export function decimatePath(
  n: number,
  getX: (i: number) => number,
  getY: (i: number) => number,
  xPx: (v: number) => number,
  yPx: (v: number) => number,
  emit: (px: number, py: number) => void,
): void {
  let col = Number.NaN;
  let first = 0;
  let lo = 0;
  let hi = 0;
  let last = 0;
  let firstX = 0;
  let loX = 0;
  let hiX = 0;
  let lastX = 0;
  let have = false;
  const flush = () => {
    if (!have) return;
    emit(firstX, first);
    // pixel y grows downwards; emit in the order that keeps the line connected to the neighbours
    if (lo !== first && lo !== last) emit(loX, lo);
    if (hi !== first && hi !== last && hi !== lo) emit(hiX, hi);
    if (last !== first || lastX !== firstX) emit(lastX, last);
  };
  for (let i = 0; i < n; i++) {
    const y = getY(i);
    if (!Number.isFinite(y)) continue;
    const x = xPx(getX(i));
    const py = yPx(y);
    const c = Math.round(x);
    if (!have || c !== col) {
      flush();
      col = c;
      first = lo = hi = last = py;
      firstX = loX = hiX = lastX = x;
      have = true;
    } else {
      if (py < lo) {
        lo = py;
        loX = x;
      }
      if (py > hi) {
        hi = py;
        hiX = x;
      }
      last = py;
      lastX = x;
    }
  }
  flush();
}

/**
 * Padded y range for the fit plot: the data range [lo, hi] extended by the curve, where the curve
 * counts only up to half the data span beyond the data (so a diverging curve cannot squash the points).
 */
export function fitYRange(
  lo: number,
  hi: number,
  curve: ArrayLike<number> | null,
): [number, number] {
  const span = hi - lo || 1;
  const cLo = lo - span * 0.5;
  const cHi = hi + span * 0.5;
  let a = lo;
  let b = hi;
  if (curve) {
    for (let i = 0; i < curve.length; i++) {
      const v = curve[i];
      if (v === undefined || !Number.isFinite(v)) continue;
      const c = Math.min(cHi, Math.max(cLo, v));
      if (c < a) a = c;
      if (c > b) b = c;
    }
  }
  return paddedRange(a, b);
}
