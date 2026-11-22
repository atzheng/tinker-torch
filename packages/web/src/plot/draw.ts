import { formatTick, type Scale } from './axes.js';
import type { PlotColors } from './color.js';
import { fitCategoryLabels } from './labelFit.js';

export type Margins = { l: number; r: number; t: number; b: number };
export const MARGINS: Margins = { l: 42, r: 10, t: 8, b: 32 };

export const fontOf = (c: PlotColors, px = 11) => `${px}px ${c.font}`;

/**
 * Grid lines, tick labels and the frame for a plot area. Tick arrays are in domain units. The axis
 * titles are maths, so they are HTML (`AxisLabels`), not canvas text.
 */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  c: PlotColors,
  w: number,
  h: number,
  m: Margins,
  xs: Scale,
  ys: Scale,
  xTicks: number[],
  yTicks: number[],
  /** Text for an x tick (default: the number); a categorical axis passes the category name. */
  xLabel: (t: number) => string = formatTick,
  /** Categorical axis: truncate long labels and thin them (see labelFit.ts). */
  fitLabels = false,
): void {
  const x0 = m.l;
  const x1 = w - m.r;
  const y0 = h - m.b;
  const y1 = m.t;
  ctx.save();
  ctx.font = fontOf(c);
  ctx.lineWidth = 1;
  ctx.strokeStyle = c.border;
  ctx.fillStyle = c.muted;
  ctx.beginPath();
  for (const t of xTicks) {
    const x = Math.round(xs.px(t)) + 0.5;
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y0);
  }
  for (const t of yTicks) {
    const y = Math.round(ys.px(t)) + 0.5;
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
  }
  ctx.globalAlpha = 0.5;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.strokeRect(x0 + 0.5, y1 + 0.5, x1 - x0, y0 - y1);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  let texts = xTicks.map((t) => xLabel(t));
  if (fitLabels && xTicks.length > 1) {
    const slot = Math.abs(xs.px(xTicks[1] ?? 1) - xs.px(xTicks[0] ?? 0));
    texts = fitCategoryLabels(texts, slot, (s) => ctx.measureText(s).width).map((s) => s ?? '');
  }
  xTicks.forEach((t, i) => ctx.fillText(texts[i] ?? '', xs.px(t), y0 + 4));
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yTicks) ctx.fillText(formatTick(t), x0 - 4, ys.px(t));
  ctx.restore();
}

export type LegendItem = {
  label: string;
  color: string;
  kind: 'line' | 'dot' | 'ring' | 'dash';
  /** Swatch opacity, for a faint series. Default 1. */
  alpha?: number;
};

/** Legend row inside the top-left of the plot area. */
export function drawLegend(
  ctx: CanvasRenderingContext2D,
  c: PlotColors,
  m: Margins,
  items: LegendItem[],
): void {
  ctx.save();
  ctx.font = fontOf(c);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  let x = m.l + 8;
  const y = m.t + 10;
  for (const it of items) {
    ctx.strokeStyle = it.color;
    ctx.fillStyle = it.color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = it.alpha ?? 1;
    ctx.setLineDash(it.kind === 'dash' ? [4, 3] : []);
    if (it.kind === 'line' || it.kind === 'dash') {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 14, y);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x + 7, y, 4, 0, Math.PI * 2);
      if (it.kind === 'dot') ctx.fill();
      else {
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.text;
    ctx.fillText(it.label, x + 19, y);
    x += 19 + ctx.measureText(it.label).width + 12;
  }
  ctx.restore();
}
