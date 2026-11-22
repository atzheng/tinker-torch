import { LAYOUT } from '@tinker/core';
import { truncateLabel } from '../plot/labelFit.js';

const FONT_PX = 10.5;

/** Rough advance width of `s` in the 10.5px label font (no DOM needed, so it is unit-testable). */
export function estimateLabelWidth(s: string): number {
  let em = 0;
  for (const ch of s) {
    if ("iljtfr1.,:;'|! ".includes(ch)) em += 0.32;
    else if ('mwMW@'.includes(ch)) em += 0.85;
    else if (ch >= 'A' && ch <= 'Z') em += 0.72;
    else em += 0.6;
  }
  return em * FONT_PX;
}

/** A category name cut with an ellipsis to fit the unit column (`LAYOUT.unitW`). */
export function fitDialName(name: string, width: number = LAYOUT.unitW): string {
  return truncateLabel(name, width, estimateLabelWidth);
}
