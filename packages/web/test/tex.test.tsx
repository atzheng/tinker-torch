// @vitest-environment jsdom
import { expect, test } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { TexText, plainOf, splitTex, texHtml, texStats } from '../src/tex/Tex.js';

test('splitTex keeps text outside $...$ as text', () => {
  expect(splitTex('a $x^2$ b')).toEqual([
    { tex: false, s: 'a ' },
    { tex: true, s: 'x^2' },
    { tex: false, s: ' b' },
  ]);
  expect(splitTex('no maths')).toEqual([{ tex: false, s: 'no maths' }]);
  expect(plainOf('adam $\\eta=0.1$')).toBe('adam \\eta=0.1');
});

test('texHtml renders with KaTeX and caches per source', () => {
  const before = texStats.katexCalls;
  const a = texHtml('\\hat{y}');
  expect(a).toContain('katex');
  expect(texHtml('\\hat{y}')).toBe(a);
  expect(texStats.katexCalls).toBe(before + 1);
});

test('no markup or link injection: html outside maths is escaped, trust is off', () => {
  const out = renderToStaticMarkup(<TexText>{'<img src=x onerror=alert(1)> $x$'}</TexText>);
  expect(out).not.toContain('<img');
  expect(out).toContain('&lt;img');
  // \href and \url need trust; with trust off they do not become links.
  expect(texHtml('\\href{javascript:alert(1)}{x}')).not.toContain('<a');
  // A bad source does not throw.
  expect(() => texHtml('\\frac{')).not.toThrow();
});
