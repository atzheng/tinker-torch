/** Seeded PRNG (CONTRACTS.md section 5). sfc32 seeded through splitmix32. */

export type Rng = {
  /** Uniform on [0, 1). */
  next(): number;
  /** Standard normal (Box-Muller). */
  normal(): number;
  /** Uniform integer on [0, n). */
  int(n: number): number;
  /** An independent stream derived from this stream's seed and a label. It does not depend on how many values were drawn. */
  fork(label: string): Rng;
};

function splitmix32(state: number): () => number {
  let s = state >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
}

/** FNV-1a over UTF-16 code units. */
function hashString(label: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < label.length; i++) {
    h = Math.imul(h ^ label.charCodeAt(i), 0x01000193) >>> 0;
  }
  return h;
}

function fromSeed32(seed: number): Rng {
  const sm = splitmix32(seed);
  let a = sm();
  let b = sm();
  let c = sm();
  let d = sm();
  let spare: number | null = null;

  const nextU32 = (): number => {
    const t = (((a + b) >>> 0) + d) >>> 0;
    d = (d + 1) >>> 0;
    a = (b ^ (b >>> 9)) >>> 0;
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return t;
  };
  // Warm up, as recommended for sfc32.
  for (let i = 0; i < 12; i++) nextU32();

  const next = (): number => nextU32() / 4294967296;

  return {
    next,
    normal() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      const u1 = 1 - next(); // in (0, 1], so log is finite
      const u2 = next();
      const r = Math.sqrt(-2 * Math.log(u1));
      spare = r * Math.sin(2 * Math.PI * u2);
      return r * Math.cos(2 * Math.PI * u2);
    },
    int(n) {
      return Math.floor(next() * n);
    },
    fork(label) {
      const h = hashString(label);
      const mix = splitmix32((seed ^ Math.imul(h, 0x9e3779b1)) >>> 0);
      return fromSeed32((mix() ^ h) >>> 0);
    },
  };
}

export function makeRng(seed: number): Rng {
  return fromSeed32(Math.floor(seed) >>> 0);
}
