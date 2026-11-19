import { memo, type CSSProperties } from 'react';
import katex from 'katex';

/**
 * Math labels (tinker-mit). KaTeX renders TeX to an HTML string once per distinct source (cached),
 * so a label never re-runs KaTeX while a run redraws, and the `memo` below keeps React from even
 * re-diffing it. `trust: false` and `throwOnError: false`: a bad source shows as red source text,
 * and nothing in a label can inject markup or links. Colour is `currentColor`, so light and dark
 * follow the surrounding text.
 */

/** Counters for the "no per-frame re-render" check; read through `window.__tinker.tex` in dev. */
export const texStats = { renders: 0, katexCalls: 0 };
if (import.meta.env.DEV && typeof window !== 'undefined') {
  const w = window as unknown as { __tinker?: Record<string, unknown> };
  w.__tinker = { ...w.__tinker, tex: texStats };
}

const cache = new Map<string, string>();

/** KaTeX HTML for a TeX source, cached. Inline (text-style) maths. */
export function texHtml(src: string): string {
  let html = cache.get(src);
  if (html === undefined) {
    texStats.katexCalls++;
    html = katex.renderToString(src, {
      throwOnError: false,
      trust: false,
      output: 'html',
      displayMode: false,
      strict: 'ignore',
    });
    cache.set(src, html);
  }
  return html;
}

/** Inline maths in a span. `src` is TeX without the dollar signs. */
export const Tex = memo(function Tex({ src }: { src: string }) {
  texStats.renders++;
  return (
    <span className="tk-tex" aria-label={src} dangerouslySetInnerHTML={{ __html: texHtml(src) }} />
  );
});

/** Splits `text $x^2$ more` into text and maths parts. Everything outside `$...$` stays text. */
export function splitTex(s: string): { tex: boolean; s: string }[] {
  const out: { tex: boolean; s: string }[] = [];
  s.split(/\$([^$]*)\$/).forEach((part, i) => {
    if (part !== '') out.push({ tex: i % 2 === 1, s: part });
  });
  return out;
}

/** The string without its maths markers, for aria-labels and titles: `$\lambda$` becomes `\lambda`. */
export const plainOf = (s: string): string => s.replace(/\$([^$]*)\$/g, '$1');

/** Text with `$...$` rendered as maths; the rest is plain React text (escaped). */
export const TexText = memo(function TexText({ children }: { children: string }) {
  const parts = splitTex(children);
  return (
    <>{parts.map((p, i) => (p.tex ? <Tex key={i} src={p.s} /> : <span key={i}>{p.s}</span>))}</>
  );
});

/**
 * Maths inside a canvas drawing: a foreignObject over `box` (node-local units, so it scales with
 * the layer and neuron view transforms). It never takes pointer events: hit-testing stays with the
 * body rect underneath. `src` is TeX; `$` markers are not needed (the whole string is maths).
 */
export function SvgTex(p: {
  src: string;
  box: { x: number; y: number; width: number; height: number };
  fontSize: number;
  align?: 'start' | 'center' | 'end';
  className?: string;
  style?: CSSProperties;
  testid?: string;
  /** `src` is text with `$...$` maths, not pure TeX. */
  mixed?: boolean;
}) {
  const justify = { start: 'flex-start', center: 'center', end: 'flex-end' }[p.align ?? 'center'];
  return (
    <foreignObject
      x={p.box.x}
      y={p.box.y}
      width={p.box.width}
      height={p.box.height}
      className="tk-fo"
      data-testid={p.testid}
    >
      <div
        className={`tk-fo-box ${p.className ?? ''}`}
        style={{ justifyContent: justify, fontSize: p.fontSize, ...p.style }}
      >
        {p.mixed ? <TexText>{p.src}</TexText> : <Tex src={p.src} />}
      </div>
    </foreignObject>
  );
}
