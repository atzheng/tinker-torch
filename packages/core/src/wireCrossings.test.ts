import { describe, expect, it } from 'vitest';
import { gapDashArray, polylineLength, wireGaps, type Pt } from './wireCrossings.js';

const h: Pt[] = [
  [0, 50],
  [100, 50],
];
const v: Pt[] = [
  [50, 0],
  [50, 100],
];

describe('wireGaps', () => {
  it('cuts the earlier wire where a later one crosses it', () => {
    const g = wireGaps([h, v]);
    expect(g[1]).toEqual([]);
    expect(g[0]).toHaveLength(1);
    const [a, b] = g[0]?.[0] as [number, number];
    expect((a + b) / 2).toBeCloseTo(50);
  });
  it('follows the stacking order and is deterministic', () => {
    const g = wireGaps([h, v], [1, 0]);
    expect(g[0]).toEqual([]);
    expect(g[1]).toHaveLength(1);
    expect(wireGaps([h, v], [1, 0])).toEqual(g);
  });
  it('cuts a gap where wires sharing a start point cross away from it', () => {
    const a: Pt[] = [
      [0, 0],
      [50, 0],
      [50, 100],
      [120, 100],
    ];
    const b: Pt[] = [
      [0, 0],
      [30, 0],
      [30, 50],
      [120, 50],
    ];
    const [ga, gb] = wireGaps([a, b]);
    expect(ga?.length).toBe(1);
    expect(gb).toEqual([]);
  });

  it('ignores collinear overlaps, shared ends and crossings near an end', () => {
    const inner: Pt[] = [
      [20, 50],
      [80, 50],
    ];
    expect(wireGaps([h, inner])).toEqual([[], []]);
    const fan: Pt[] = [
      [0, 50],
      [50, 100],
    ];
    expect(wireGaps([h, fan])).toEqual([[], []]);
    const nearEnd: Pt[] = [
      [95, 0],
      [95, 100],
    ];
    expect(wireGaps([h, nearEnd])[0]).toEqual([]);
  });
  it('builds a dash array covering the whole length', () => {
    const g = wireGaps([h, v])[0] ?? [];
    expect(gapDashArray(g, polylineLength(h))).toBe('45.5 9 45.5 0');
    expect(gapDashArray([], 100)).toBeUndefined();
  });
});
