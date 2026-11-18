import type { NodeOf, PieceType } from '@tinker/core';
import { activation } from './activation.js';
import { add } from './add.js';
import { dropout } from './dropout.js';
import { feature } from './feature.js';
import { input } from './input.js';
import { linear } from './linear.js';
import { loss } from './loss.js';
import { output } from './output.js';
import { target } from './target.js';
import type { EnginePiece } from './types.js';

export type { EnginePiece, PieceCtx } from './types.js';

/** One entry per PieceType: adding a type fails to typecheck until it has an engine piece. */
export const ENGINE_PIECES: { [T in PieceType]: EnginePiece<NodeOf<T>> } = {
  input,
  feature,
  linear,
  activation,
  dropout,
  add,
  output,
  target,
  loss,
};
