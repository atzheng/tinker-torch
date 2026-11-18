// Public engine types (CONTRACTS.md section 7). No jax array type is exported from the package.
import type { Analysis, Batch, DataInfo, Graph, HostParams, NodeId } from '@tinker/core';

export type { Batch };

/** Call `Engine.disposeModel` when done with it: the jitted functions hold memory. */
export type Model = { readonly graph: Graph; readonly analysis: Analysis };
/** Opaque; owns jax arrays. Becomes unusable once consumed or disposed (later use throws). */
export type Params = { readonly __brand: 'Params' };
/**
 * `opt` and `key` are opaque (they hold jax arrays). `params` is a Params owned by the state.
 * `step` counts successful updates.
 */
export type TrainState = { params: Params; opt: unknown; step: number; key: unknown };
export type StepResult = { state: TrainState; loss: number; diverged: boolean };
/** One trained copy of `trainMany`: its parameters, and whether it diverged (then `params` are its last finite ones). */
export type ManyResult = { params: HostParams; diverged: boolean };
export type ParamAxis = { node: NodeId; tensor: 'W' | 'b'; index: number };

export interface Engine {
  compile(graph: Graph, data: DataInfo): Model;
  init(model: Model, seed: number): Params;
  fromHost(model: Model, host: HostParams): Params;
  toHost(params: Params): HostParams;
  write(params: Params, nodeId: NodeId, values: { W?: Float32Array; b?: Float32Array }): Params;
  startTraining(model: Model, params: Params, seed: number): TrainState;
  step(model: Model, state: TrainState, batch: Batch): StepResult;
  predict(model: Model, params: Params, x: Float32Array, n: number): Float32Array;
  evaluate(model: Model, params: Params, batch: Batch): number;
  lossGrid(
    model: Model,
    params: Params,
    batch: Batch,
    axes: [ParamAxis, ParamAxis],
    a: Float32Array,
    b: Float32Array,
  ): Float32Array;
  trainMany(model: Model, seeds: number[], data: Batch[], steps: number): ManyResult[];
  disposeModel(model: Model): void;
  clone(params: Params): Params;
  dispose(x: Params | TrainState): void;
  liveArrays(): number;
}
