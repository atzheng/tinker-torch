import { numpy as np } from '@jax-js/jax';
import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * Takes x and keeps the node's columns, in order. `inputs` is empty (an Input has no input port).
 * Consumes the batch x (one reference from the context).
 */
export const input: EnginePiece<NodeOf<'input'>> = {
  forward(node, _inputs, _layer, ctx) {
    const x = ctx.takeX();
    const idx = node.columns.map((c) => ctx.dataColumns.indexOf(c));
    if (idx.some((v) => v < 0)) {
      x.dispose();
      throw new Error('An Input column is not in the dataset.');
    }
    const identity = idx.length === ctx.dataColumns.length && idx.every((v, i) => v === i);
    if (identity) return x;
    // Static slices (one per contiguous ascending run), never an index array: constants captured
    // by a jitted function are never freed.
    const runs: [number, number][] = [];
    for (const v of idx) {
      const last = runs[runs.length - 1];
      if (last && last[1] === v) last[1] = v + 1;
      else runs.push([v, v + 1]);
    }
    // slice consumes x: take a reference for every run but the last
    const parts = runs.map((r, i) => (i < runs.length - 1 ? x.ref : x).slice([], r));
    const [only] = parts;
    return parts.length === 1 && only ? only : np.concatenate(parts, 1);
  },
};
