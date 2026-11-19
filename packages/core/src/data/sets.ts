/** The synthetic generators and their true functions (CONTRACTS.md section 6). */
import type { DatasetName } from '../common.js';
import type { Rng } from '../rng.js';

export type SetKind = 'regression' | 'binary' | 'multiclass';

export type SetSpec = {
  name: DatasetName;
  kind: SetKind;
  d: number;
  /** Per column, in raw units. */
  domain: [lo: number, hi: number][];
  /** Multiclass only: the number of classes k >= 3 (binary has 2, regression 1). */
  numClasses?: number;
  /** Feature names; default x1, x2, ... A real dataset fills these. */
  featureNames?: string[];
  /** Target name; default 'y'. */
  targetName?: string;
  /** Classification: one name per class in label order; default 'class 0', 'class 1', ... */
  classNames?: string[];
  /**
   * Regression: f*(x). Binary: P(y=1|x) of the clean generator. Multiclass: the class index (an
   * integer 0..k-1) of the clean generator. `x` is one row of raw inputs.
   */
  f: (x: readonly number[]) => number;
  /**
   * Binary only: a smooth signed field that is positive where the truth is class 1 and
   * zero on the decision boundary (for contouring; `f` itself jumps for most sets).
   */
  margin?: (x: readonly number[]) => number;
};

const D3: [number, number][] = [[-3, 3]];
const D2: [number, number][] = [
  [-2, 2],
  [-2, 2],
];

const at = (x: readonly number[], i: number): number => x[i] ?? 0;

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

const BLOB_STD = 0.6;
const CIRCLE_R_IN = 1;
const CIRCLE_R_OUT_LO = 1.3;
const CIRCLE_R_OUT_HI = 2;
/** The truth for the circle switches at the middle of the empty gap between the classes. */
const CIRCLE_CUT = (CIRCLE_R_IN + CIRCLE_R_OUT_LO) / 2;

/** Distance from (px, py) to the half circle {c + (r cos t, r sin t * sign) : t in [0, pi]}. */
function distToHalfCircle(px: number, py: number, cx: number, cy: number, upper: boolean): number {
  const dx = px - cx;
  const dy = (upper ? 1 : -1) * (py - cy);
  if (dy >= 0) return Math.abs(Math.hypot(dx, dy) - 1);
  return Math.min(Math.hypot(dx - 1, dy), Math.hypot(dx + 1, dy));
}

/** Three Gaussian blobs on a circle of radius 1.5, at 90, 210 and 330 degrees. */
export const BLOBS3_CENTERS: [number, number][] = [0, 1, 2].map((c) => {
  const a = Math.PI / 2 + (2 * Math.PI * c) / 3;
  return [1.5 * Math.cos(a), 1.5 * Math.sin(a)];
});

function nearest(x: readonly number[], centers: readonly [number, number][]): number {
  let best = 0;
  let bd = Infinity;
  centers.forEach(([cx, cy], i) => {
    const d = (at(x, 0) - cx) ** 2 + (at(x, 1) - cy) ** 2;
    if (d < bd) {
      bd = d;
      best = i;
    }
  });
  return best;
}

/** Three spiral arms: arm c is r = SPIRAL_R t at angle 2 pi c / 3 + SPIRAL_TWIST t, t in [0.1, 1]. */
export const SPIRAL_R = 1.8;
export const SPIRAL_TWIST = 4;
export const SPIRAL_T0 = 0.1;
export function spiralPoint(arm: number, t: number): [number, number] {
  const a = (2 * Math.PI * arm) / 3 + SPIRAL_TWIST * t;
  return [SPIRAL_R * t * Math.cos(a), SPIRAL_R * t * Math.sin(a)];
}
const SPIRAL_STEPS = 120;
const SPIRAL_PTS: [number, number][][] = [0, 1, 2].map((arm) =>
  Array.from({ length: SPIRAL_STEPS + 1 }, (_, i) =>
    spiralPoint(arm, SPIRAL_T0 + ((1 - SPIRAL_T0) * i) / SPIRAL_STEPS),
  ),
);
/** The arm whose polyline is nearest the point (the clean generator's most probable class). */
function nearestArm(px: number, py: number): number {
  let best = 0;
  let bd = Infinity;
  SPIRAL_PTS.forEach((pts, arm) => {
    for (const [qx, qy] of pts) {
      const d = (px - qx) ** 2 + (py - qy) ** 2;
      if (d < bd) {
        bd = d;
        best = arm;
      }
    }
  });
  return best;
}

export const MOONS_OUTER = { cx: 0, cy: 0 };
export const MOONS_INNER = { cx: 1, cy: 0.5 };

export const SETS: Record<DatasetName, SetSpec> = {
  line: { name: 'line', kind: 'regression', d: 1, domain: D3, f: (x) => 0.8 * at(x, 0) + 0.5 },
  quadratic: {
    name: 'quadratic',
    kind: 'regression',
    d: 1,
    domain: D3,
    f: (x) => 0.5 * at(x, 0) ** 2 - 1,
  },
  sine: { name: 'sine', kind: 'regression', d: 1, domain: D3, f: (x) => Math.sin(1.5 * at(x, 0)) },
  step: { name: 'step', kind: 'regression', d: 1, domain: D3, f: (x) => (at(x, 0) < 0 ? -1 : 1) },
  wiggle: {
    name: 'wiggle',
    kind: 'regression',
    d: 1,
    domain: D3,
    f: (x) => 0.5 * Math.sin(3 * at(x, 0)) + 0.3 * at(x, 0),
  },
  plane: {
    name: 'plane',
    kind: 'regression',
    d: 2,
    domain: D2,
    f: (x) => 0.6 * at(x, 0) - 0.4 * at(x, 1),
  },
  bowl: {
    name: 'bowl',
    kind: 'regression',
    d: 2,
    domain: D2,
    f: (x) => 0.5 * (at(x, 0) ** 2 + at(x, 1) ** 2) - 1,
  },
  saddle: {
    name: 'saddle',
    kind: 'regression',
    d: 2,
    domain: D2,
    f: (x) => 0.5 * (at(x, 0) ** 2 - at(x, 1) ** 2),
  },
  blobs: {
    name: 'blobs',
    kind: 'binary',
    d: 2,
    domain: [
      [-3, 3],
      [-3, 3],
    ],
    // Equal-variance Gaussians at (-1,-1) and (1,1): the log odds are linear in x1 + x2.
    f: (x) => sigmoid((4 * (at(x, 0) + at(x, 1))) / (2 * BLOB_STD * BLOB_STD)),
    margin: (x) => at(x, 0) + at(x, 1),
  },
  circle: {
    name: 'circle',
    kind: 'binary',
    d: 2,
    domain: D2,
    f: (x) => (Math.hypot(at(x, 0), at(x, 1)) < CIRCLE_CUT ? 1 : 0),
    margin: (x) => CIRCLE_CUT - Math.hypot(at(x, 0), at(x, 1)),
  },
  xor: {
    name: 'xor',
    kind: 'binary',
    d: 2,
    domain: D2,
    f: (x) => (at(x, 0) * at(x, 1) > 0 ? 1 : 0),
    margin: (x) => at(x, 0) * at(x, 1),
  },
  moons: {
    name: 'moons',
    kind: 'binary',
    d: 2,
    domain: [
      [-1.5, 2.5],
      [-1, 1.5],
    ],
    // Class 1 (inner moon) where the point is nearer to the inner half circle than to the outer one.
    f: (x) => {
      const px = at(x, 0);
      const py = at(x, 1);
      const dOut = distToHalfCircle(px, py, MOONS_OUTER.cx, MOONS_OUTER.cy, true);
      const dIn = distToHalfCircle(px, py, MOONS_INNER.cx, MOONS_INNER.cy, false);
      return dIn < dOut ? 1 : 0;
    },
    margin: (x) => {
      const px = at(x, 0);
      const py = at(x, 1);
      return (
        distToHalfCircle(px, py, MOONS_OUTER.cx, MOONS_OUTER.cy, true) -
        distToHalfCircle(px, py, MOONS_INNER.cx, MOONS_INNER.cy, false)
      );
    },
  },
  blobs3: {
    name: 'blobs3',
    kind: 'multiclass',
    numClasses: 3,
    d: 2,
    domain: [
      [-3, 3],
      [-3, 3],
    ],
    // Equal-variance Gaussians: the most probable class is the nearest centre.
    f: (x) => nearest(x, BLOBS3_CENTERS),
  },
  spiral3: {
    name: 'spiral3',
    kind: 'multiclass',
    numClasses: 3,
    d: 2,
    domain: D2,
    f: (x) => nearestArm(at(x, 0), at(x, 1)),
  },
};

// ---------------------------------------------------------------- spread constant s

const spreadCache = new Map<DatasetName, number>();

/**
 * s: standard deviation of f* over a uniform sample of the domain. Computed once per set on a
 * dense regular grid (deterministic, no randomness needed).
 */
export function spread(set: DatasetName): number {
  const cached = spreadCache.get(set);
  if (cached !== undefined) return cached;
  const spec = SETS[set];
  const m = spec.d === 1 ? 4001 : 401;
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  const visit = (x: number[]): void => {
    const v = spec.f(x);
    n++;
    sum += v;
    sumSq += v * v;
  };
  const coord = (col: number, i: number): number => {
    const [lo, hi] = spec.domain[col] ?? [0, 1];
    return lo + ((i + 0.5) / m) * (hi - lo); // cell centres
  };
  if (spec.d === 1) {
    for (let i = 0; i < m; i++) visit([coord(0, i)]);
  } else {
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) visit([coord(0, i), coord(1, j)]);
  }
  const mean = sum / n;
  const s = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  spreadCache.set(set, s);
  return s;
}

// ---------------------------------------------------------------- generators

/** One generated row: raw inputs and the target. */
export type Row = { x: number[]; y: number };

function uniform(rng: Rng, lo: number, hi: number): number {
  return lo + rng.next() * (hi - lo);
}

/**
 * Draws n rows. Classification rows cycle through the labels 0, 1, (2, ...) so every contiguous
 * slice (the train and validation parts) is balanced to within one point.
 * Regression x is uniform on the domain; y = f*(x) + noise * s * eps.
 */
export function generate(set: DatasetName, n: number, noise: number, rng: Rng): Row[] {
  const spec = SETS[set];
  const rx = rng.fork('x');
  const rn = rng.fork('noise');
  const rows: Row[] = [];

  if (spec.kind === 'regression') {
    const s = spread(set);
    for (let i = 0; i < n; i++) {
      // fround so that the stored Float32 x reproduces f* exactly at noise 0.
      const x = spec.domain.map(([lo, hi]) => Math.fround(uniform(rx, lo, hi)));
      const eps = rn.normal();
      rows.push({ x, y: spec.f(x) + noise * s * eps });
    }
    return rows;
  }

  const jitter = noise * 0.5;
  for (let i = 0; i < n; i++) {
    const label = i % (spec.numClasses ?? 2);
    if (set === 'blobs3') {
      const [cx, cy] = BLOBS3_CENTERS[label] ?? [0, 0];
      const sd = BLOB_STD * (0.5 + noise);
      rows.push({ x: [cx + sd * rx.normal(), cy + sd * rx.normal()], y: label });
      continue;
    }
    if (set === 'spiral3') {
      const t = uniform(rx, SPIRAL_T0, 1);
      const [qx, qy] = spiralPoint(label, t);
      rows.push({
        x: [Math.fround(qx + jitter * rn.normal()), Math.fround(qy + jitter * rn.normal())],
        y: label,
      });
      continue;
    }
    let px: number;
    let py: number;
    let sd = jitter;
    if (set === 'blobs') {
      const c = label === 1 ? 1 : -1;
      sd = BLOB_STD * (0.5 + noise);
      px = c + sd * rx.normal();
      py = c + sd * rx.normal();
      rows.push({ x: [px, py], y: label });
      continue;
    }
    if (set === 'circle') {
      const lo = label === 1 ? 0 : CIRCLE_R_OUT_LO ** 2;
      const hi = label === 1 ? CIRCLE_R_IN ** 2 : CIRCLE_R_OUT_HI ** 2;
      const r = Math.sqrt(uniform(rx, lo, hi)); // uniform in area
      const t = uniform(rx, 0, 2 * Math.PI);
      px = r * Math.cos(t);
      py = r * Math.sin(t);
    } else if (set === 'xor') {
      // Uniform magnitudes and choose signs so that x1*x2 > 0 iff label 1.
      const a = uniform(rx, 0, 2);
      const b = uniform(rx, 0, 2);
      const sa = rx.next() < 0.5 ? -1 : 1;
      const sb = label === 1 ? sa : -sa;
      px = sa * a;
      py = sb * b;
    } else {
      // moons: outer half circle centred at 0 (class 0), inner centred at (1, 0.5) (class 1).
      const t = uniform(rx, 0, Math.PI);
      if (label === 0) {
        px = Math.cos(t);
        py = Math.sin(t);
      } else {
        px = 1 - Math.cos(t);
        py = 0.5 - Math.sin(t);
      }
    }
    px += sd * rn.normal();
    py += sd * rn.normal();
    rows.push({ x: [Math.fround(px), Math.fround(py)], y: label });
  }
  return rows;
}
