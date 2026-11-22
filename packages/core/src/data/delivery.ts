/**
 * The delivery story (CONTRACTS.md section 6): a generated dataset for the running example of the
 * first lectures. One generator, two target sets that share the same feature rows for a given seed
 * and n: `delivery_late` (binary) and `delivery_time` (continuous, minutes).
 *
 * Columns, in the order the model sees them: distance (km), rain (mm/h), queue (orders ahead at
 * the kitchen). A fourth column, zone (0..3), is drawn and does influence the truth, but it is not
 * an input column yet (the categorical-input bead adds it); its draws come from their own random
 * stream, so adding it later changes no other column.
 */
import type { Rng } from '../rng.js';
import type { SetSpec } from './sets.js';

export const DELIVERY_SETS = ['delivery_late', 'delivery_time'] as const;
export type DeliveryName = (typeof DELIVERY_SETS)[number];

export const DELIVERY_ZONES = ['centre', 'north', 'east', 'south'] as const;
const ZONES = DELIVERY_ZONES.length;

/** Per column, raw units: distance km, rain mm/h, queue orders. */
export const DELIVERY_DOMAIN: [number, number][] = [
  [0.5, 10],
  [0, 10],
  [0, 12],
];

/**
 * True late probability: P(late | distance, rain, queue, zone) = sigmoid(b0 + b1 distance +
 * b2 rain + b3 queue + zone effect). The zone effects average to 0. Averaged over the generator's
 * inputs the rate of late deliveries is about 0.35 (`DELIVERY_LATE_RATE`, checked in the tests).
 */
export const LATE_TRUE = {
  intercept: -4.7,
  distance: 0.4,
  rain: 0.25,
  queue: 0.12,
  zone: [-0.6, -0.2, 0.2, 0.6],
};

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

/** P(late | x), the clean generator averaged over the (equally likely) zones. */
function lateProb(x: readonly number[]): number {
  let s = 0;
  for (let z = 0; z < ZONES; z++) s += sigmoid(lateLogit(x, z));
  return s / ZONES;
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

/** E[time | x]: the zone effects average to 0, so this is the linear part. */
function timeTruth(x: readonly number[]): number {
  return timeMean(x, 0) - (TIME_TRUE.zone[0] ?? 0);
}

const COMMON = {
  d: 3,
  domain: DELIVERY_DOMAIN,
  featureNames: ['distance (km)', 'rain (mm/h)', 'queue (orders)'],
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

export type DeliveryRow = { x: number[]; zone: number; y: number };

/**
 * Draws n rows. Every column has its own stream, so the feature rows depend on the seed and n only,
 * not on the target set, and the outcome streams do not disturb the features.
 */
export function generateDelivery(
  set: DeliveryName,
  n: number,
  noise: number,
  rng: Rng,
): DeliveryRow[] {
  const rDist = rng.fork('distance');
  const rRain = rng.fork('rain');
  const rQueue = rng.fork('queue');
  const rZone = rng.fork('zone');
  const rLate = rng.fork('late');
  const rNoise = rng.fork('time-noise');
  const [dLo, dHi] = DELIVERY_DOMAIN[0] ?? [0, 1];
  const [wLo, wHi] = DELIVERY_DOMAIN[1] ?? [0, 1];
  const [, qHi] = DELIVERY_DOMAIN[2] ?? [0, 1];
  const rows: DeliveryRow[] = [];
  for (let i = 0; i < n; i++) {
    // fround: the stored Float32 x then reproduces the truth exactly.
    const x = [
      Math.fround(dLo + rDist.next() * (dHi - dLo)),
      Math.fround(wLo + rRain.next() * (wHi - wLo)),
      rQueue.int(qHi + 1),
    ];
    const zone = rZone.int(ZONES);
    // Both outcome streams are always consumed, so a row's draw does not depend on the target set.
    const u = rLate.next();
    const eps = rNoise.normal();
    const y =
      set === 'delivery_late'
        ? u < sigmoid(lateLogit(x, zone))
          ? 1
          : 0
        : timeMean(x, zone) + NOISE_SD_PER_UNIT * noise * eps;
    rows.push({ x, zone, y });
  }
  return rows;
}
