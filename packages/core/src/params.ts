/** Host-side parameters (CONTRACTS.md section 5): plain Float32Arrays, no jax-js. */
import type { Analysis } from './analyse.js';
import type { NodeId } from './common.js';
import type { ParamInit, ParamShapes } from './registry.js';
import { makeRng } from './rng.js';

/**
 * Named tensors per parameter-owning node, row-major. Linear has W [in, out] and b [out]; other owners
 * use their own names. Only live param nodes of `analysis.paramNodes` have an entry.
 */
export type HostLayer = Record<string, Float32Array>;
export type HostParams = Record<NodeId, HostLayer>;
export type JsonParams = Record<NodeId, Record<string, number[]>>;

const size = (shape: readonly number[]): number => shape.reduce((a, d) => a * d, 1);
const bare = <T>() => Object.create(null) as Record<string, T>; // no prototype: tensor names are data

type Owner = { shapes: ParamShapes; init: ParamInit; type: string | undefined };

/**
 * Shapes and init of every live param node (`analysis.paramNodes`) whose input width is known. A Linear
 * read only through a Weights tap is live too, though it is not in `analysis.order`.
 */
function owners(analysis: Analysis, all = false): Map<NodeId, Owner> {
  const out = new Map<NodeId, Owner>();
  for (const id of all ? Object.keys(analysis.paramShapes) : analysis.paramNodes) {
    const shapes = analysis.paramShapes[id];
    const init = analysis.paramInit[id];
    if (shapes && init) out.set(id, { shapes, init, type: analysis.paramType[id] });
  }
  return out;
}

/** One node's initial parameters: tensors drawn in the key order of `shapes`, from one stream per node. */
export function initNode(
  shapes: ParamShapes,
  init: ParamInit,
  seed: number,
  id: NodeId,
): HostLayer {
  // One stream per node: adding a node does not change the others.
  const rng = makeRng(seed).fork(id);
  const out = bare<Float32Array>();
  for (const k of Object.keys(shapes)) {
    const a = new Float32Array(size(shapes[k] ?? []));
    if (init.kind === 'uniform') {
      for (let i = 0; i < a.length; i++) a[i] = (rng.next() * 2 - 1) * init.bound;
    } else {
      a.fill(init.value);
    }
    out[k] = a;
  }
  return out;
}

function matches(p: Record<string, ArrayLike<number>>, s: ParamShapes): boolean {
  const have = Object.keys(p);
  const want = Object.keys(s);
  if (have.length !== want.length) return false;
  return want.every((k) => Object.hasOwn(p, k) && p[k]?.length === size(s[k] ?? []));
}

/** Initial parameters of every live param node (Linear: W and b uniform on +/- 1/sqrt(in), as nn.Linear). */
export function initParams(analysis: Analysis, seed: number): HostParams {
  const host: HostParams = Object.create(null) as HostParams;
  for (const [id, o] of owners(analysis)) host[id] = initNode(o.shapes, o.init, seed, id);
  return host;
}

const sameShape = (a: readonly number[] | undefined, b: readonly number[] | undefined): boolean =>
  a !== undefined && b !== undefined && a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * Keep each node's arrays (same objects, bit-identical) when their shapes are unchanged between the old and
 * new graph. If every tensor present in both keeps its shape, those are kept, tensors only in the new shapes
 * are drawn fresh and tensors only in the old are dropped (Linear: W is kept when [in, out] is unchanged
 * even if bias was toggled; b is drawn fresh when bias is turned on). If any shared tensor changed shape,
 * or the old entry does not fit, or the node's piece type changed (an id reused for another piece), the
 * whole node is re-inited. A node that exists but is inert (no role needs it, so it is never compiled or
 * trained) keeps its arrays frozen while its param shapes are unchanged, so wiring it in again restores
 * them bit for bit; it is dropped when deleted, when its shapes change or its input width is unknown.
 * Does not mutate `old`. The returned arrays are shared with `old` and must never be written in place.
 */
export function reconcileParams(
  old: HostParams,
  oldAnalysis: Analysis,
  analysis: Analysis,
  seed: number,
): HostParams {
  const oldOwners = owners(oldAnalysis, true);
  const live = new Set(analysis.paramNodes);
  const host: HostParams = Object.create(null) as HostParams;
  for (const [id, o] of owners(analysis, true)) {
    const prev = Object.hasOwn(old, id) ? old[id] : undefined;
    const os = oldOwners.get(id);
    const keys = Object.keys(o.shapes);
    if (!live.has(id)) {
      const same =
        prev !== undefined &&
        os !== undefined &&
        os.type === o.type &&
        keys.length === Object.keys(prev).length &&
        keys.every(
          (k) =>
            prev[k]?.length === size(o.shapes[k] ?? []) && sameShape(os.shapes[k], o.shapes[k]),
        );
      if (same) host[id] = prev;
      continue;
    }
    const shared = os ? keys.filter((k) => Object.hasOwn(os.shapes, k)) : [];
    const keep =
      prev !== undefined &&
      os !== undefined &&
      os.type === o.type &&
      shared.length > 0 &&
      shared.every(
        (k) => sameShape(os.shapes[k], o.shapes[k]) && prev[k]?.length === size(o.shapes[k] ?? []),
      );
    if (!prev || !keep) {
      host[id] = initNode(o.shapes, o.init, seed, id);
      continue;
    }
    let fresh: HostLayer | undefined;
    const layer = bare<Float32Array>();
    for (const k of keys) {
      const kept = shared.includes(k) ? prev[k] : undefined;
      if (kept) {
        layer[k] = kept;
      } else {
        fresh ??= initNode(o.shapes, o.init, seed, id);
        const f = fresh[k];
        if (f) layer[k] = f;
      }
    }
    host[id] = layer;
  }
  return host;
}

/**
 * Overlay start values (what a loaded file or preset carried) on `base` (a fresh init): each node of
 * `base` that has an entry in `start` whose tensors fit its current shapes exactly takes the start
 * values; every other node keeps `base`. Entries that no longer fit (shapes changed by an edit) or
 * name nodes that are gone are ignored, never an error. Does not mutate its arguments; the start
 * arrays are shared with the result (HostParams arrays are immutable).
 */
export function withStartParams(
  base: HostParams,
  start: HostParams | null,
  analysis: Analysis,
): HostParams {
  if (!start) return base;
  const shapes = owners(analysis, true);
  const out: HostParams = Object.create(null) as HostParams;
  for (const id of Object.keys(base)) {
    const s = Object.hasOwn(start, id) ? start[id] : undefined;
    const o = shapes.get(id);
    out[id] = s && o && matches(s, o.shapes) ? s : (base[id] as HostLayer);
  }
  return out;
}

export function countParams(host: HostParams): number {
  let n = 0;
  for (const id of Object.keys(host)) {
    const p = host[id];
    if (p) for (const k of Object.keys(p)) n += p[k]?.length ?? 0;
  }
  return n;
}

export function paramsToJson(host: HostParams): JsonParams {
  const out: JsonParams = Object.create(null) as JsonParams;
  for (const id of Object.keys(host)) {
    const p = host[id];
    if (!p) continue;
    const t: Record<string, number[]> = {};
    for (const k of Object.keys(p)) t[k] = Array.from(p[k] ?? []);
    out[id] = t;
  }
  return out;
}

/**
 * Load saved params for `graph`. Entries whose tensors do not match (or that are malformed, or for nodes
 * that own no live parameters) are left out and their ids listed in `dropped`.
 * Param nodes with no saved entry are simply absent (the caller reconciles to fill them).
 */
export function paramsFromJson(
  json: unknown,
  analysis: Analysis,
): { params: HostParams; dropped: NodeId[] } {
  const params: HostParams = Object.create(null) as HostParams;
  const dropped: NodeId[] = [];
  if (typeof json !== 'object' || json === null || Array.isArray(json)) return { params, dropped };
  const shapes = owners(analysis);
  const isNums = (v: unknown): v is number[] => {
    if (!Array.isArray(v)) return false;
    for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) return false; // also catches holes
    return true;
  };
  for (const id of Object.keys(json)) {
    const raw: unknown = (json as Record<string, unknown>)[id];
    const o = shapes.get(id);
    const r = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
    const cand = bare<number[]>();
    let ok = o !== undefined;
    for (const k of Object.keys(r)) {
      const v = r[k];
      if (isNums(v)) cand[k] = v;
      else ok = false;
    }
    if (!o || !ok || !matches(cand, o.shapes)) {
      dropped.push(id);
      continue;
    }
    const layer = bare<Float32Array>();
    for (const k of Object.keys(cand)) layer[k] = Float32Array.from(cand[k] ?? []);
    params[id] = layer;
  }
  return { params, dropped };
}
