import type { FC } from 'react';
import type { Node, PieceType } from '@tinker/core';

export type NodeOf<T extends PieceType> = Extract<Node, { type: T }>;

/** Web half of a piece (CONTRACTS.md section 3): layer drawing, neuron drawing, inspector fields. */
export type WebPiece<T extends PieceType> = {
  /** SVG group drawn in the layer view. */
  Layer: FC<{ node: NodeOf<T> }>;
  /** SVG group drawn in the neuron view. */
  Neuron: FC<{ node: NodeOf<T> }>;
  /** Inspector settings for this piece. */
  Fields: FC<{ node: NodeOf<T> }>;
};
