import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/**
 * The one-hot of the leaf each row falls in. A Tree reads an Input, so its output is a fixed
 * function of the batch rows: the host computes it (core `treeOneHot`/`treeLeafIndex`) and it arrives
 * as part of the jitted function's argument x, `ctx.takeTree`. Nothing is fitted or stored here, no
 * gradient flows through it. Consumes `inputs.in` (the Input's columns, unused) and returns `[n, leaves]`.
 */
export const tree: EnginePiece<NodeOf<'tree'>> = {
  forward(node, inputs, _layer, ctx) {
    inputs['in']?.dispose();
    return ctx.takeTree(node.id);
  },
};
