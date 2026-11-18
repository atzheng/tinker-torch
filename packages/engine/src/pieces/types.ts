import type { Array as JArray } from '@jax-js/jax';
import type { Node, OutputLink } from '@tinker/core';
import type { Layer } from '../convert.js';

/** What a piece sees besides its inputs. */
export type PieceCtx = {
  /** True while training (dropout on); false in predict and evaluate. */
  train: boolean;
  /** True when the Output piece must return the pre-link value (the loss applies or fuses the link). */
  rawOutput: boolean;
  /** The link of the Output feeding the loss (the loss piece uses it). */
  link: OutputLink;
  /** Dataset column names, in the order of the batch's columns. */
  dataColumns: readonly string[];
  /**
   * Hand out the batch x ([n, dataColumns]) and y ([n]). Each call returns one reference that the
   * caller consumes (the walker adds `.ref` as needed).
   */
  takeX(): JArray;
  takeY(): JArray;
  /**
   * The 0/1 keep mask [n, width] of a Dropout node for this pass (sampled by the step, or injected by
   * the test-only `debugTrainPass`, CONTRACTS.md section 11). The caller consumes it.
   */
  dropoutMask(nodeId: string): JArray;
};

/**
 * One engine piece. `implemented: false` marks a placeholder that `compile` rejects with "not
 * implemented yet" (a later bead fills the file). `forward` CONSUMES every array in `inputs` and
 * `layer`; the caller passes `.ref` where it reuses one. It returns a new array the caller owns.
 * A loss piece returns the mean loss over the batch (without the penalty).
 */
export type EnginePiece<N extends Node> = {
  implemented: boolean;
  forward(node: N, inputs: Record<string, JArray>, layer: Layer | undefined, ctx: PieceCtx): JArray;
};

export function notYet(what: string): never {
  throw new Error(`${what} is not implemented yet.`);
}
