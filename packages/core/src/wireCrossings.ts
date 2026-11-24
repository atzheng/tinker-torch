/**
 * Over/under at wire crossings (tinker-di7, CONTRACTS.md section 10). Pure geometry: wires are the
 * polylines of `layout.wires`; a wire later in the stacking order passes over earlier ones, so the
 * earlier (under) wire gets a gap where it is crossed. The renderer cuts those gaps with a dash
 * pattern, so nothing is painted in the background colour and region shading shows through.
 */
export type Pt = readonly [number, number];

/** A stretch of a wire, as arc-length distances from its start. */
export type Gap = readonly [number, number];

export type GapOptions = {
  /** Gap half-length along the under wire for a perpendicular crossing. */
  half?: number;
  /** No gap closer than this (arc length) to either end of a wire: ports and arrowheads stay whole. */
  endMargin?: number;
};

const EPS = 1e-9;

/**
 * `gaps[i]` lists the sorted, non-overlapping gaps of `wires[i]`. `order` is the stacking order
 * (indices into `wires`, bottom first; default 0..n-1). Only proper crossings of two segments count:
 * collinear overlaps and touches (a shared trunk, a shared start or end point, an elbow) merely
 * meet and get no gap; two wires that share an endpoint but also cross further on do get one. Deterministic: depends only on the geometry and `order`.
 */
export function wireGaps(
  wires: readonly (readonly Pt[])[],
  order?: readonly number[],
  opts: GapOptions = {},
): Gap[][] {
  const half = opts.half ?? 4.5;
  const margin = opts.endMargin ?? 14;
  const rank = new Array<number>(wires.length).fill(0);
  (order ?? wires.map((_, i) => i)).forEach((w, r) => (rank[w] = r));
  const lens = wires.map(cumulative);
  const raw: Gap[][] = wires.map(() => []);
  for (let i = 0; i < wires.length; i++) {
    for (let j = i + 1; j < wires.length; j++) {
      const a = wires[i] as readonly Pt[];
      const b = wires[j] as readonly Pt[];
      if (a.length < 2 || b.length < 2) continue;
      const under = (rank[i] ?? 0) < (rank[j] ?? 0) ? i : j;
      const over = under === i ? j : i;
      const u = wires[under] as readonly Pt[];
      const o = wires[over] as readonly Pt[];
      const ul = lens[under] as number[];
      const total = ul[ul.length - 1] ?? 0;
      for (let s = 0; s + 1 < u.length; s++) {
        for (let t = 0; t + 1 < o.length; t++) {
          const hit = cross(u[s] as Pt, u[s + 1] as Pt, o[t] as Pt, o[t + 1] as Pt);
          if (!hit) continue;
          const at = (ul[s] as number) + hit.along;
          if (at < margin || total - at < margin) continue;
          // A shallow crossing needs a longer gap to clear the other stroke.
          const h = Math.min(half / Math.max(hit.sin, 0.35), half * 3);
          (raw[under] as Gap[]).push([at - h, at + h]);
        }
      }
    }
  }
  return raw.map((g, i) => merge(g, lens[i]?.[(lens[i]?.length ?? 1) - 1] ?? 0));
}

/** Dash array that draws a polyline of `total` length except across `gaps`; undefined when none. */
export function gapDashArray(gaps: readonly Gap[], total: number): string | undefined {
  if (gaps.length === 0) return undefined;
  const parts: number[] = [];
  let at = 0;
  for (const [a, b] of gaps) {
    parts.push(round(a - at), round(b - a));
    at = b;
  }
  // Pattern repeats: the final dash covers the rest, and the trailing gap is zero-length padding.
  parts.push(round(Math.max(total - at, 0)), 0);
  return parts.join(' ');
}

/** Total arc length of a polyline. */
export function polylineLength(pts: readonly Pt[]): number {
  const c = cumulative(pts);
  return c[c.length - 1] ?? 0;
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}

function cumulative(pts: readonly Pt[]): number[] {
  const out = [0];
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i - 1] as Pt;
    const q = pts[i] as Pt;
    out.push((out[i - 1] as number) + Math.hypot(q[0] - p[0], q[1] - p[1]));
  }
  return out;
}

/** Proper interior crossing of segments p-q and r-s: arc length along p-q and the sine of the angle. */
function cross(p: Pt, q: Pt, r: Pt, s: Pt): { along: number; sin: number } | null {
  const d1x = q[0] - p[0];
  const d1y = q[1] - p[1];
  const d2x = s[0] - r[0];
  const d2y = s[1] - r[1];
  const den = d1x * d2y - d1y * d2x;
  const l1 = Math.hypot(d1x, d1y);
  const l2 = Math.hypot(d2x, d2y);
  if (l1 < EPS || l2 < EPS) return null;
  const sin = Math.abs(den) / (l1 * l2);
  if (sin < 0.05) return null; // parallel or collinear: no crossing
  const t = ((r[0] - p[0]) * d2y - (r[1] - p[1]) * d2x) / den;
  const u = ((r[0] - p[0]) * d1y - (r[1] - p[1]) * d1x) / den;
  // Strictly inside both segments: a touch at a vertex (an elbow, a shared run) is not a crossing.
  if (t <= 1e-6 || t >= 1 - 1e-6 || u <= 1e-6 || u >= 1 - 1e-6) return null;
  return { along: t * l1, sin };
}

function merge(gaps: Gap[], total: number): Gap[] {
  const sorted = gaps
    .map(([a, b]) => [Math.max(a, 0), Math.min(b, total)] as [number, number])
    .sort((x, y) => x[0] - y[0]);
  const out: [number, number][] = [];
  for (const g of sorted) {
    const last = out[out.length - 1];
    if (last && g[0] <= last[1] + 1) last[1] = Math.max(last[1], g[1]);
    else out.push([g[0], g[1]]);
  }
  return out;
}
