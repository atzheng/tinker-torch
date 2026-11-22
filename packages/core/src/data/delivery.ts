/**
 * The delivery story (CONTRACTS.md section 6): a generated dataset for the running example of the
 * first lectures. One generator, two target sets that share the same feature rows for a given seed
 * and n: `delivery_late` (binary) and `delivery_time` (continuous, minutes).
 *
 * Columns, in the order the model sees them: distance (km), rain (mm/h), queue (orders ahead at
 * the kitchen) and zone, a categorical column (4 categories stored as the index 0..3; tinker-6t7.7).
 * Zone's draws come from their own random stream, so adding it changed no other column.
 */
import type { Rng } from '../rng.js';
import type { Row, SetSpec } from './sets.js';

export const DELIVERY_SETS = ['delivery_late', 'delivery_time'] as const;
export type DeliveryName = (typeof DELIVERY_SETS)[number];

export const DELIVERY_ZONES = ['centre', 'north', 'east', 'south'] as const;
const ZONES = DELIVERY_ZONES.length;

/** Per column, raw units: distance km, rain mm/h, queue orders, zone index. */
export const DELIVERY_DOMAIN: [number, number][] = [
  [0.5, 10],
  [0, 10],
  [0, 12],
  [0, ZONES - 1],
];

/** The zone column (x4): its index in a row. */
export const ZONE_COLUMN = 3;

/**
 * True late probability: P(late | distance, rain, queue, zone) = sigmoid(b0 + b1 distance +
 * b2 rain + b3 queue + zone effect). The zone effects average to 0. Averaged over the generator's
 * inputs the rate of late deliveries is `DELIVERY_LATE_RATE`.
 */
export const LATE_TRUE = {
  intercept: -5.0,
  distance: 0.4,
  rain: 0.25,
  queue: 0.12,
  zone: [-0.6, -0.2, 0.2, 0.6],
};

/**
 * The population late rate: the mean of P(late | x, zone) over distance and rain uniform, queue
 * uniform on 0..12 and the four zones equally likely. Computed by midpoint-rule integration on a
 * 400 x 400 grid of (distance, rain) with every queue value and zone (0.34082...); the test
 * recomputes it to 3 decimals.
 */
export const DELIVERY_LATE_RATE = 0.3408;

/** True mean delivery time in minutes: linear in the three inputs plus a zone effect (mean 0). */
export const TIME_TRUE = {
  intercept: 12,
  distance: 3,
  rain: 0.8,
  queue: 1,
  zone: [-2, -1, 1, 2],
};
/** Delivery-time noise standard deviation, in minutes, is NOISE_SD_PER_UNIT * noise. */
export const NOISE_SD_PER_UNIT = 10;

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

const at = (x: readonly number[], i: number): number => x[i] ?? 0;

/** The row's zone as a category index (clamped to the four zones). */
const zoneOf = (x: readonly number[]): number =>
  Math.min(ZONES - 1, Math.max(0, Math.round(at(x, ZONE_COLUMN))));

function lateLogit(x: readonly number[], zone: number): number {
  const t = LATE_TRUE;
  return (
    t.intercept +
    t.distance * at(x, 0) +
    t.rain * at(x, 1) +
    t.queue * at(x, 2) +
    (t.zone[zone] ?? 0)
  );
}

/** P(late | x): the clean generator at the row's own zone (x[3]). */
function lateProb(x: readonly number[]): number {
  return sigmoid(lateLogit(x, zoneOf(x)));
}

function timeMean(x: readonly number[], zone: number): number {
  const t = TIME_TRUE;
  return (
    t.intercept +
    t.distance * at(x, 0) +
    t.rain * at(x, 1) +
    t.queue * at(x, 2) +
    (t.zone[zone] ?? 0)
  );
}

/** E[time | x]: the linear part plus the row's own zone effect. */
function timeTruth(x: readonly number[]): number {
  return timeMean(x, zoneOf(x));
}

const COMMON = {
  d: 4,
  domain: DELIVERY_DOMAIN,
  featureNames: ['distance (km)', 'rain (mm/h)', 'queue (orders)', 'zone'],
  /** Column index -> category names (the zone column). */
  categories: { [ZONE_COLUMN]: [...DELIVERY_ZONES] },
};

export const DELIVERY_SPECS: Record<DeliveryName, SetSpec> = {
  delivery_late: {
    ...COMMON,
    name: 'delivery_late',
    kind: 'binary',
    targetName: 'late',
    classNames: ['on time', 'late'],
    noiseUnused: true,
    f: lateProb,
  },
  delivery_time: {
    ...COMMON,
    name: 'delivery_time',
    kind: 'regression',
    targetName: 'delivery time (min)',
    f: timeTruth,
  },
};

/**
 * Draws n rows. Every column has its own stream, so the feature rows depend on the seed and n only,
 * not on the target set, and the outcome streams do not disturb the features.
 */
export function generateDelivery(set: DeliveryName, n: number, noise: number, rng: Rng): Row[] {
  const rDist = rng.fork('distance');
  const rRain = rng.fork('rain');
  const rQueue = rng.fork('queue');
  const rZone = rng.fork('zone');
  const rLate = rng.fork('late');
  const rNoise = rng.fork('time-noise');
  const [dLo, dHi] = DELIVERY_DOMAIN[0] ?? [0, 1];
  const [wLo, wHi] = DELIVERY_DOMAIN[1] ?? [0, 1];
  const [, qHi] = DELIVERY_DOMAIN[2] ?? [0, 1];
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    // fround: the stored Float32 x then reproduces the truth exactly.
    const zone = rZone.int(ZONES);
    const x = [
      Math.fround(dLo + rDist.next() * (dHi - dLo)),
      Math.fround(wLo + rRain.next() * (wHi - wLo)),
      rQueue.int(qHi + 1),
      zone,
    ];
    // Both outcome streams are always consumed, so a row's draw does not depend on the target set.
    const u = rLate.next();
    const eps = rNoise.normal();
    const y =
      set === 'delivery_late'
        ? u < sigmoid(lateLogit(x, zone))
          ? 1
          : 0
        : timeMean(x, zone) + NOISE_SD_PER_UNIT * noise * eps;
    rows.push({ x, y });
  }
  return rows;
}
