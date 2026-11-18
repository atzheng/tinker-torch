/** Pure helpers for 2-D fit plots: colour ramps, grid-to-image mapping, contour lines. No DOM. */

export type Rgb = [number, number, number];

/** Parses `#rgb`, `#rrggbb` or `rgb(r, g, b)`; anything else gives mid grey. */
export function parseColor(s: string): Rgb {
  const t = s.trim();
  if (t.startsWith('#')) {
    const h = t.length === 4 ? [...t.slice(1)].map((ch) => ch + ch).join('') : t.slice(1, 7);
    const n = Number.parseInt(h, 16);
    if (h.length === 6 && Number.isFinite(n)) return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/.exec(t);
  if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
  return [128, 128, 128];
}

/** Number of entries in a ramp lookup table. */
export const RAMP_SIZE = 256;

/** Lookup table (RAMP_SIZE x 3 bytes) interpolating evenly spaced colour stops. */
export function makeRamp(stops: readonly string[]): Uint8ClampedArray {
  const cols = stops.map(parseColor);
  const lut = new Uint8ClampedArray(RAMP_SIZE * 3);
  const segs = Math.max(1, cols.length - 1);
  for (let i = 0; i < RAMP_SIZE; i++) {
    const u = (i / (RAMP_SIZE - 1)) * segs;
    const k = Math.min(segs - 1, Math.floor(u));
    const f = u - k;
    const a = cols[k] ?? [128, 128, 128];
    const b = cols[k + 1] ?? a;
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = (a[j] ?? 0) + ((b[j] ?? 0) - (a[j] ?? 0)) * f;
  }
  return lut;
}

/** Position of v in [lo, hi] as 0..1, clamped; NaN gives NaN. */
export function unit(v: number, lo: number, hi: number): number {
  if (Number.isNaN(v)) return Number.NaN;
  const t = (v - lo) / (hi - lo || 1);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** CSS `rgb(...)` of the ramp at t in 0..1 (clamped; NaN gives the first entry). */
export function rampCss(lut: Uint8ClampedArray, t: number): string {
  const i = Number.isFinite(t) ? Math.round(Math.min(1, Math.max(0, t)) * (RAMP_SIZE - 1)) : 0;
  return `rgb(${lut[i * 3]},${lut[i * 3 + 1]},${lut[i * 3 + 2]})`;
}

/**
 * Writes an m x m grid of values (row index iy * m + ix, x1 fastest) into RGBA pixels of an
 * m x m image. Image row 0 is the top, so grid row iy lands on image row m - 1 - iy: x1 grows to
 * the right and x2 grows upward. Non-finite values are left transparent.
 */
export function fillGridImage(
  rgba: Uint8ClampedArray,
  values: ArrayLike<number>,
  m: number,
  lut: Uint8ClampedArray,
  lo: number,
  hi: number,
): void {
  for (let iy = 0; iy < m; iy++) {
    const row = (m - 1 - iy) * m;
    for (let ix = 0; ix < m; ix++) {
      const v = values[iy * m + ix] ?? Number.NaN;
      const o = (row + ix) * 4;
      const t = unit(v, lo, hi);
      if (Number.isNaN(t)) {
        rgba[o + 3] = 0;
        continue;
      }
      const k = Math.round(t * (RAMP_SIZE - 1)) * 3;
      rgba[o] = lut[k] ?? 0;
      rgba[o + 1] = lut[k + 1] ?? 0;
      rgba[o + 2] = lut[k + 2] ?? 0;
      rgba[o + 3] = 255;
    }
  }
}

/**
 * Marching squares for one level on an m x m grid (index coordinates: x = ix, y = iy, each
 * 0..m-1). Returns flat segments [x0, y0, x1, y1, ...]; cells with a non-finite corner are skipped.
 */
export function contourSegments(values: ArrayLike<number>, m: number, level: number): number[] {
  const out: number[] = [];
  const at = (ix: number, iy: number) => values[iy * m + ix] ?? Number.NaN;
  const lerp = (a: number, b: number) => (level - a) / (b - a || 1);
  for (let iy = 0; iy < m - 1; iy++) {
    for (let ix = 0; ix < m - 1; ix++) {
      const v00 = at(ix, iy);
      const v10 = at(ix + 1, iy);
      const v11 = at(ix + 1, iy + 1);
      const v01 = at(ix, iy + 1);
      if (!(
        Number.isFinite(v00) &&
        Number.isFinite(v10) &&
        Number.isFinite(v11) &&
        Number.isFinite(v01)
      ))
        continue;
      const code =
        (v00 >= level ? 1 : 0) |
        (v10 >= level ? 2 : 0) |
        (v11 >= level ? 4 : 0) |
        (v01 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      // edge crossings: bottom (y = iy), right (x = ix + 1), top (y = iy + 1), left (x = ix)
      const bottom = (): [number, number] => [ix + lerp(v00, v10), iy];
      const right = (): [number, number] => [ix + 1, iy + lerp(v10, v11)];
      const top = (): [number, number] => [ix + lerp(v01, v11), iy + 1];
      const left = (): [number, number] => [ix, iy + lerp(v00, v01)];
      const seg = (a: [number, number], b: [number, number]) => out.push(a[0], a[1], b[0], b[1]);
      switch (code) {
        case 1:
        case 14:
          seg(left(), bottom());
          break;
        case 2:
        case 13:
          seg(bottom(), right());
          break;
        case 3:
        case 12:
          seg(left(), right());
          break;
        case 4:
        case 11:
          seg(right(), top());
          break;
        case 6:
        case 9:
          seg(bottom(), top());
          break;
        case 7:
        case 8:
          seg(left(), top());
          break;
        case 5:
        case 10: {
          // saddle: join by the cell centre's side
          const centre = (v00 + v10 + v11 + v01) / 4 >= level;
          if ((code === 5) === centre) {
            seg(left(), top());
            seg(bottom(), right());
          } else {
            seg(left(), bottom());
            seg(right(), top());
          }
          break;
        }
      }
    }
  }
  return out;
}

/** Index coordinate (0..m-1) to a domain value on [lo, hi] (the grid spans the ends inclusive). */
export function indexToDomain(i: number, m: number, lo: number, hi: number): number {
  return lo + (i / (m - 1)) * (hi - lo);
}

/** Column `col` of a row-major [n, width] array as a new or reused Float32Array of length n. */
export function column(out: Float32Array, src: Float32Array, n: number, col: number): Float32Array {
  const width = Math.max(1, Math.round(src.length / n));
  const c = Math.min(col, width - 1);
  for (let i = 0; i < n; i++) out[i] = src[i * width + c] ?? Number.NaN;
  return out;
}

/** Min and max of the finite values of the given arrays; null when there are none. */
export function finiteRange(...arrs: ArrayLike<number>[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrs) {
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (v === undefined || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return hi >= lo ? [lo, hi] : null;
}
