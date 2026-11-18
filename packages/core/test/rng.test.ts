import { describe, expect, test } from 'vitest';
import { makeRng } from '../src/index.js';

const draw = (r: ReturnType<typeof makeRng>, n: number) =>
  Array.from({ length: n }, () => r.next());

describe('makeRng', () => {
  test('is deterministic per seed', () => {
    expect(draw(makeRng(5), 20)).toEqual(draw(makeRng(5), 20));
    expect(draw(makeRng(5), 20)).not.toEqual(draw(makeRng(6), 20));
  });

  test('next is in [0, 1) and int is in range', () => {
    const r = makeRng(1);
    for (let i = 0; i < 1000; i++) {
      const u = r.next();
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
      const k = r.int(7);
      expect(Number.isInteger(k) && k >= 0 && k < 7).toBe(true);
    }
  });

  test('fork streams differ from each other and from the parent, and are deterministic', () => {
    const a = makeRng(9).fork('a');
    const b = makeRng(9).fork('b');
    expect(draw(a, 10)).not.toEqual(draw(b, 10));
    expect(draw(makeRng(9).fork('a'), 10)).toEqual(draw(makeRng(9).fork('a'), 10));
    expect(draw(makeRng(9).fork('a'), 10)).not.toEqual(draw(makeRng(9), 10));
    expect(draw(makeRng(9).fork('a'), 10)).not.toEqual(draw(makeRng(10).fork('a'), 10));
  });

  test('fork does not depend on draws already made', () => {
    const p = makeRng(3);
    draw(p, 50);
    expect(draw(p.fork('epoch0'), 5)).toEqual(draw(makeRng(3).fork('epoch0'), 5));
  });

  test('normal has mean ~0 and std ~1 over 1e5 draws', () => {
    const r = makeRng(42);
    const n = 100_000;
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < n; i++) {
      const z = r.normal();
      s += z;
      s2 += z * z;
    }
    const mean = s / n;
    const std = Math.sqrt(s2 / n - mean * mean);
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(std - 1)).toBeLessThan(0.02);
  });
});
