/// <reference types="node" />
import { v2 } from '@tinker/core';
import type { DataInfo, HostParams } from '@tinker/core';

type Graph = v2.Graph;
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export type JsonParams = Record<string, { W: number[]; b?: number[] }>;
export type Fixture = {
  name: string;
  graph: unknown;
  params: JsonParams;
  batch: { x: number[]; y: number[]; n: number };
  expect: {
    pred: number[];
    /** v2 fixtures: every ŷ pin in order (`pred` is the first); absent in v1 fixtures. */
    predictions?: number[][];
    loss: number;
    grads: JsonParams;
    after?: { steps: number; params: JsonParams; losses: number[] };
    dropoutMask?: number[][];
    predEval?: number[];
    lossEval?: number;
  };
};

const dir = fileURLToPath(new URL('./fixtures/', import.meta.url));

export function fixtureNames(): string[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .map((f) => f.replace(/\.json$/, ''))
    .sort();
}

export function loadFixture(name: string): { fx: Fixture; graph: Graph; data: DataInfo } {
  const fx = JSON.parse(readFileSync(`${dir}${name}.json`, 'utf8')) as Fixture;
  // v1 fixture graphs are migrated by parseGraph; v2 ones are read as written
  const parsed = v2.parseGraph(fx.graph);
  if (!parsed.ok) throw new Error(parsed.error);
  const input = parsed.graph.nodes.find((n) => n.type === 'input');
  const columns = input?.type === 'input' ? input.columns : [];
  const classification = ['blobs', 'circle', 'xor', 'moons'].includes(parsed.graph.data.set);
  const data: DataInfo = {
    columns,
    kind: classification ? 'binary' : 'regression',
    numClasses: classification ? 2 : 1,
  };
  return { fx, graph: parsed.graph, data };
}

export function toHostParams(p: JsonParams): HostParams {
  const h = Object.create(null) as HostParams;
  for (const [id, v] of Object.entries(p)) {
    h[id] = v.b
      ? { W: Float32Array.from(v.W), b: Float32Array.from(v.b) }
      : { W: Float32Array.from(v.W) };
  }
  return h;
}

export function toBatch(fx: Fixture): { x: Float32Array; y: Float32Array; n: number } {
  return { x: Float32Array.from(fx.batch.x), y: Float32Array.from(fx.batch.y), n: fx.batch.n };
}

/** Largest |a - b| and whether every pair satisfies |a - b| <= tol * (1 + |b|) (absolute and relative). */
export function compare(
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  tol: number,
): { maxAbs: number; ok: boolean } {
  if (got.length !== want.length) return { maxAbs: Infinity, ok: false };
  let maxAbs = 0;
  let ok = true;
  for (let i = 0; i < got.length; i++) {
    const g = got[i] ?? NaN;
    const w = want[i] ?? NaN;
    const e = Math.abs(g - w);
    if (!(e <= tol * (1 + Math.abs(w)))) ok = false;
    if (!(e <= maxAbs)) maxAbs = e;
  }
  return { maxAbs, ok };
}

/** Same graph with another learning rate. */
export function withLr(graph: Graph, lr: number): Graph {
  return { ...graph, train: { ...graph.train, lr } };
}

/** The first yhat pin's predictions (the old single-output `predict`). */
export function predict0(
  engine: { predict(m: never, p: never, x: Float32Array, n: number): Float32Array[] },
  model: unknown,
  params: unknown,
  x: Float32Array,
  n: number,
): Float32Array {
  const out = engine.predict(model as never, params as never, x, n)[0];
  if (!out) throw new Error('This model has no prediction pin.');
  return out;
}
