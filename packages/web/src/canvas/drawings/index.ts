import type { PieceType } from '@tinker/core';
import type { WebPiece } from './types.js';
import { activationPiece } from './activation.js';
import { addPiece } from './add.js';
import { dropoutPiece } from './dropout.js';
import { featurePiece } from './feature.js';
import { inputPiece } from './input.js';
import { linearPiece } from './linear.js';
import { lossPiece } from './loss.js';
import { outputPiece } from './output.js';
import { targetPiece } from './target.js';

export type { WebPiece } from './types.js';

export const WEB_PIECES: { [T in PieceType]: WebPiece<T> } = {
  input: inputPiece,
  feature: featurePiece,
  linear: linearPiece,
  activation: activationPiece,
  dropout: dropoutPiece,
  add: addPiece,
  output: outputPiece,
  target: targetPiece,
  loss: lossPiece,
};
