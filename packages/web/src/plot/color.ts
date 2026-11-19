/** Canvas colours come from the CSS variables in theme.css; re-read when the theme changes. */

export type PlotColors = {
  bg: string;
  text: string;
  muted: string;
  border: string;
  train: string;
  val: string;
  truth: string;
  accent: string;
  /** Colours of the pinned runs, by slot. */
  pins: [string, string, string];
  /** Diverging ramp (class 0 to class 1) and sequential ramp stops, as CSS colour strings. */
  divNeg: string;
  divMid: string;
  divPos: string;
  seq: [string, string, string];
  /** Outline behind 2-D fit points so they stand out on any region of the surface. */
  halo: string;
  /** Point colours for class 0 and class 1. */
  class0: string;
  class1: string;
  /** Categorical class colours (multi-class sets): class c uses `classes[c % classes.length]`. */
  classes: string[];
  font: string;
};

export const CLASS_COUNT = 6;

/** Colour of class `c`, cycling when a set has more classes than the palette. */
export function classColor(colors: PlotColors, c: number): string {
  return (
    colors.classes[((c % colors.classes.length) + colors.classes.length) % colors.classes.length] ??
    colors.text
  );
}

const FALLBACK: PlotColors = {
  bg: '#ffffff',
  text: '#1b1f27',
  muted: '#5b6472',
  border: '#cfd4dc',
  train: '#1f5fd1',
  val: '#d98200',
  truth: '#8a93a1',
  accent: '#2557d6',
  pins: ['#0b8a6a', '#b5499b', '#8a6d00'],
  divNeg: '#f0b273',
  divMid: '#f3f1ee',
  divPos: '#2a62c4',
  seq: ['#440154', '#21918c', '#fde725'],
  halo: '#ffffff',
  class0: '#d35400',
  class1: '#1f5fd1',
  classes: ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#56b4e9', '#d55e00'],
  font: 'system-ui, sans-serif',
};

/** Reads the plot colours from the computed style of `el` (so a `data-theme` or media query applies). */
export function readColors(el: Element): PlotColors {
  const cs = getComputedStyle(el);
  const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
  return {
    bg: v('--surface', FALLBACK.bg),
    text: v('--text', FALLBACK.text),
    muted: v('--text-muted', FALLBACK.muted),
    border: v('--border', FALLBACK.border),
    train: v('--series-train', FALLBACK.train),
    val: v('--series-val', FALLBACK.val),
    truth: v('--series-true', FALLBACK.truth),
    accent: v('--accent', FALLBACK.accent),
    pins: [
      v('--pin-1', FALLBACK.pins[0]),
      v('--pin-2', FALLBACK.pins[1]),
      v('--pin-3', FALLBACK.pins[2]),
    ],
    divNeg: v('--div-neg', FALLBACK.divNeg),
    divMid: v('--div-mid', FALLBACK.divMid),
    divPos: v('--div-pos', FALLBACK.divPos),
    seq: [
      v('--seq-0', FALLBACK.seq[0]),
      v('--seq-1', FALLBACK.seq[1]),
      v('--seq-2', FALLBACK.seq[2]),
    ],
    halo: v('--point-halo', FALLBACK.halo),
    class0: v('--weight-neg', FALLBACK.class0),
    class1: v('--weight-pos', FALLBACK.class1),
    classes: Array.from({ length: CLASS_COUNT }, (_, i) =>
      v(`--class-${i}`, FALLBACK.classes[i] ?? FALLBACK.text),
    ),
    font: v('--font', FALLBACK.font),
  };
}
