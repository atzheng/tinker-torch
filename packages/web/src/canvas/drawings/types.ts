import type { FC } from 'react';
import type { Node, PieceType } from '@tinker/core';

export type NodeOf<T extends PieceType> = Extract<Node, { type: T }>;

/**
 * Props of a layer drawing. The drawing is placed with its box's top-left corner at the origin
 * and fills `w` x `h` abstract units. Shapes use the classes in canvas.css (`tk-shape`,
 * `tk-text`, ...) so that error and selected states recolour them.
 */
export type LayerProps<T extends PieceType> = {
  node: NodeOf<T>;
  w: number;
  h: number;
  /** Width entering the piece, when analysis knows it. */
  inWidth?: number;
  /** Width leaving the piece, when analysis knows it. */
  outWidth?: number;
  /** Parameter count of this piece (0 for most). */
  paramCount: number;
  /** Weights to draw as a heat map (Linear only); read only, never write into the arrays. */
  weights?: { W: Float32Array; rows: number; cols: number };
};

/** Web half of a piece (CONTRACTS.md section 3): layer drawing, neuron drawing, inspector fields. */
export type WebPiece<T extends PieceType> = {
  /** SVG group drawn in the layer view. */
  Layer: FC<LayerProps<T>>;
  /** SVG group drawn in the neuron view. */
  Neuron: FC<{ node: NodeOf<T> }>;
  /** Inspector settings for this piece. Edits go through core edits and `graphSlice.apply`. */
  Fields: FC<{ node: NodeOf<T> }>;
};
