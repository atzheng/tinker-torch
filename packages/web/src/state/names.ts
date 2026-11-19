import type { DataInfo } from '@tinker/core';

/** Display name of a dataset column id (`x1`, ...): the feature name when the dataset has one. */
export function featureName(info: DataInfo, col: string): string {
  const i = info.columns.indexOf(col);
  return (i >= 0 ? info.featureNames?.[i] : undefined) ?? col;
}

/** Axis title for AxisLabels: `x1` becomes the maths $x_1$, a feature name is shown as text ($ removed, it would start maths). */
export function axisLabel(name: string): string {
  const m = /^x(\d+)$/.exec(name);
  return m ? `$x_${m[1]}$` : name.replace(/\$/g, '');
}

/** Cuts `text` with an ellipsis so it fits `w` abstract units at `px` font size (about 0.6 em per character). */
export function fitText(text: string, w: number, px = 13): string {
  const max = Math.max(3, Math.floor(w / (px * 0.6)));
  return text.length <= max ? text : `${text.slice(0, max - 1)}\u2026`;
}

/** Names of the Input's columns as one line: as many whole names as fit, then " +N" for the rest. */
export function inputLine(info: DataInfo, columns: readonly string[], w: number, px = 13): string {
  const names = columns.map((c) => featureName(info, c));
  const room = Math.max(3, Math.floor(w / (px * 0.6)));
  let line = '';
  let shown = 0;
  for (const n of names) {
    const next = shown === 0 ? n : `${line}, ${n}`;
    const more = names.length - shown - 1;
    if (shown > 0 && next.length + (more > 0 ? ` +${more}`.length : 0) > room) break;
    line = next;
    shown++;
  }
  const rest = names.length - shown;
  if (rest === 0) return fitText(line, w, px);
  const suffix = ` +${rest}`;
  const head =
    line.length + suffix.length > room ? fitText(line, w - suffix.length * px * 0.6, px) : line;
  return head + suffix;
}
