import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';

/** CIE L* of an sRGB hex colour (D65). */
function lstar(hex: string): number {
  const n = parseInt(hex.slice(1), 16);
  const lin = (v: number) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const y = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return y > 216 / 24389 ? 116 * Math.cbrt(y) - 16 : (24389 / 27) * y;
}

const css = readFileSync(new URL('../src/theme.css', import.meta.url), 'utf8');
const darkAt = css.indexOf('prefers-color-scheme: dark');
const token = (block: string, name: string): string => {
  const m = new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`).exec(block);
  if (!m?.[1]) throw new Error(`missing ${name}`);
  return m[1];
};

describe('diverging ramp lightness', () => {
  test.each([
    ['light', css.slice(0, darkAt)],
    ['dark', css.slice(darkAt)],
  ])('%s: --div-neg and --div-pos differ by at least 25 in L*', (_name, block) => {
    const d = Math.abs(lstar(token(block, '--div-neg')) - lstar(token(block, '--div-pos')));
    expect(d).toBeGreaterThanOrEqual(25);
  });
});
