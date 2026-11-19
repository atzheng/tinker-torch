/**
 * TEMPORARY adapter (tinker-8xw.3, removed by tinker-8xw.4): the engine now takes graph version 2,
 * the web app still holds version 1 graphs. This wraps an `Engine` so the rest of the web app keeps
 * calling `compile(v1Graph, data)` and `predict(...)` as before:
 *  - `compile` migrates the v1 graph with `v2.migrateV1` (same node ids for everything except
 *    spliced identity Outputs and Loss groups, so parameter keys, `lossGrid` axes and `write` node
 *    ids are unchanged: they all name Linears);
 *  - `predict` returns the first ŷ pin, which migration puts where the Output the loss read was
 *    (the Output's own node for sigmoid/softmax, its source for identity).
 * Nothing else is translated: every other Engine method is passed through.
 */
import { v2 } from '@tinker/core';
import type { DataInfo, Graph } from '@tinker/core';
import type { Engine as EngineV2, Model } from '@tinker/engine';

export type Engine = Omit<EngineV2, 'compile' | 'predict'> & {
  compile(graph: Graph, data: DataInfo): Model;
  /** The first ŷ pin, `[n, width]` row-major (what the version 1 engine returned). */
  predict(...args: Parameters<EngineV2['predict']>): Float32Array;
};

export function adaptEngine(e: EngineV2): Engine {
  return {
    ...e,
    compile: (graph, data) => e.compile(v2.migrateV1(graph), data),
    predict: (model, params, x, n) => e.predict(model, params, x, n)[0] ?? new Float32Array(0),
  };
}
