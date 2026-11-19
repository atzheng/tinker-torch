import type { NodeOf, PieceType } from '@tinker/core';
import { abs } from './abs.js';
import { activation } from './activation.js';
import { add } from './add.js';
import { cross_entropy } from './cross_entropy.js';
import { dropout } from './dropout.js';
import { feature } from './feature.js';
import { huber } from './huber.js';
import { input } from './input.js';
import { linear } from './linear.js';
import { link } from './link.js';
import { log_loss } from './log_loss.js';
import { mean } from './mean.js';
import { norm } from './norm.js';
import { scale } from './scale.js';
import { square } from './square.js';
import { subtract } from './subtract.js';
import { target } from './target.js';
import type { EnginePiece } from './types.js';
import { weights } from './weights.js';

export type { EnginePiece, PieceCtx } from './types.js';

/** One entry per PieceType: adding a type fails to typecheck until it has an engine piece. */
export const ENGINE_PIECES: { [T in PieceType]: EnginePiece<NodeOf<T>> } = {
  input,
  target,
  feature,
  linear,
  activation,
  dropout,
  add,
  link,
  subtract,
  square,
  abs,
  huber,
  log_loss,
  cross_entropy,
  mean,
  scale,
  weights,
  norm,
};
