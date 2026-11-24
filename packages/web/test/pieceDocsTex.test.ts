// Every TeX string a piece card can show parses in the KaTeX the web renders with (core has no KaTeX).
import katex from 'katex';
import { expect, test } from 'vitest';
import {
  analyse,
  docView,
  docViewFor,
  fitTrees,
  PALETTE,
  PIECE_DOCS,
  sample,
  treeLeafCounts,
  type DocView,
} from '@tinker/core';
import { PRESET_NAMES, bundledPreset } from '../src/presets/index.js';

/** Same options as `Tex`, except that a parse error throws. */
const parses = (src: string): void => {
  katex.renderToString(src, {
    throwOnError: true,
    trust: false,
    output: 'html',
    displayMode: false,
    strict: 'ignore',
  });
};

const texOf = (v: DocView): string[] => [
  v.equation,
  ...(v.shapes ? [v.shapes.out, ...Object.values(v.shapes.in)] : []),
  ...(v.params !== undefined ? [v.params] : []),
  ...v.settings.map(() => ''),
];

test('every PIECE_DOCS equation and variant equation parses', () => {
  for (const [key, d] of Object.entries(PIECE_DOCS)) {
    expect(() => parses(d.equation), key).not.toThrow();
    for (const [fn, v] of Object.entries(d.variants ?? {})) {
      expect(() => parses(v.equation), `${key}.${fn}`).not.toThrow();
    }
  }
});

test('every symbolic card parses', () => {
  for (const e of PALETTE) {
    for (const s of texOf(docView(e.key))) expect(() => parses(s), `${e.key}: ${s}`).not.toThrow();
  }
});

test('live cards of the bundled presets parse', () => {
  let nodes = 0;
  for (const name of PRESET_NAMES) {
    const f = bundledPreset(name);
    if (f === null) continue;
    let ds;
    try {
      ds = sample(f.graph.data);
    } catch {
      continue; // a preset whose data needs an embedded CSV
    }
    const a = analyse(f.graph, ds.info, treeLeafCounts(fitTrees(f.graph, ds)));
    for (const n of f.graph.nodes) {
      const v = docViewFor(n, a, ds.info);
      nodes += 1;
      for (const s of texOf(v)) expect(() => parses(s), `${name}/${n.id}: ${s}`).not.toThrow();
    }
  }
  expect(nodes).toBeGreaterThan(20);
});
