import type { Array as JArray } from '@jax-js/jax';
import type { v2 } from '@tinker/core';
import type { Layer } from '../convert.js';

/** What a piece sees besides its inputs. */
export type PieceCtx = {
  /** True while training (dropout on); false in predict and evaluate. */
  train: boolean;
  /** Dataset column names, in the order of the batch's columns. */
  dataColumns: readonly string[];
  /**
   * True for a log_loss / cross_entropy node whose `pred` input is the LOGITS, because the link in
   * front of it was fused into it (stable forms, see those pieces). The compiler rewired the edge.
   */
  fused(nodeId: string): boolean;
  /**
   * Hand out the batch x ([n, dataColumns]) and y ([n]). Each call returns one reference that the
   * caller consumes (the walker adds `.ref` as needed). With no target in the batch (predict) y is zeros.
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
 * One engine piece. `forward` CONSUMES every array in `inputs` and `layer`; the caller passes `.ref`
 * where it reuses one. It returns a new array the caller owns. `layer` is the parameters of the
 * node's own Linear, or, for a Weights tap, of the Linear it reads (one reference for this use).
 * Per-example values are `[n, width]`; a Mean or a norm gives a scalar.
 */
export type EnginePiece<N extends v2.Node> = {
  forward(node: N, inputs: Record<string, JArray>, layer: Layer | undefined, ctx: PieceCtx): JArray;
};
