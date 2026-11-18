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
  font: string;
};

const FALLBACK: PlotColors = {
  bg: '#ffffff',
  text: '#1b1f27',
  muted: '#5b6472',
  border: '#cfd4dc',
  train: '#1f5fd1',
  val: '#d98200',
  truth: '#8a93a1',
  accent: '#2557d6',
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
    font: v('--font', FALLBACK.font),
  };
}
