import type { PieceType } from '@tinker/core';
import type { WebPiece } from './types.js';
import { activationPiece } from './activation.js';
import { addPiece } from './add.js';
import { dropoutPiece } from './dropout.js';
import { featurePiece } from './feature.js';
import { inputPiece } from './input.js';
import { linearPiece } from './linear.js';
import {
  absPiece,
  crossEntropyPiece,
  huberPiece,
  linkPiece,
  logLossPiece,
  meanPiece,
  normPiece,
  scalePiece,
  squarePiece,
  subtractPiece,
  weightsPiece,
} from './math.js';
import { targetPiece } from './target.js';
import { lookupPiece, onehotPiece } from './category.js';
import { treePiece } from './tree.js';
import { biasPiece, columnPiece, scalewPiece, sumPiece } from './scalar.js';

export type { WebPiece } from './types.js';

export const WEB_PIECES: { [T in PieceType]: WebPiece<T> } = {
  input: inputPiece,
  feature: featurePiece,
  linear: linearPiece,
  activation: activationPiece,
  dropout: dropoutPiece,
  add: addPiece,
  target: targetPiece,
  link: linkPiece,
  subtract: subtractPiece,
  square: squarePiece,
  abs: absPiece,
  huber: huberPiece,
  log_loss: logLossPiece,
  cross_entropy: crossEntropyPiece,
  mean: meanPiece,
  scale: scalePiece,
  weights: weightsPiece,
  norm: normPiece,
  column: columnPiece,
  scalew: scalewPiece,
  bias: biasPiece,
  sum: sumPiece,
  onehot: onehotPiece,
  lookup: lookupPiece,
  tree: treePiece,
};
