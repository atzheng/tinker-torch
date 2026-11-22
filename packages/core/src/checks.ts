import { analyse, type Analysis } from './analyse.js';
import { sample } from './data/index.js';
import type { Graph } from './graph.js';
import type { HostParams } from './params.js';

/**
 * A preset's success check (CONTRACTS.md section 12): one fixed kind of condition, evaluated
 * against the live loss and the current parameters. `step` is a 0-based index into the preset's
 * `instructions`; a check without it counts toward the preset-level tick.
 */
export type Check =
  | { kind: 'lossBelow'; value: number; split?: 'train' | 'val'; step?: number }
  | {
      kind: 'paramNear';
      node: string;
      tensor?: 'W' | 'b';
      index?: number;
      value: number;
      tol: number;
      step?: number;
    };

export type CheckInput = {
  liveLoss: { train: number | null; val: number | null } | null;
  host: HostParams | null;
};

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Parameter shapes of the graph, or null when they cannot be known at parse time (e.g. a CSV). */
function shapesOf(graph: Graph): Analysis['paramShapes'] | null {
  try {
    return analyse(graph, sample(graph.data).info).paramShapes;
  } catch {
    return null;
  }
}

/**
 * Validates a preset's `checks`; returns the checks or one plain sentence. `idMap` (a migrated
 * version 1 file) maps old node ids to the new ones, as for `locked`.
 */
export function parseChecks(
  raw: unknown,
  graph: Graph,
  stepCount: number,
  idMap?: Record<string, string[]> | null,
): Check[] | string {
  if (!Array.isArray(raw)) return 'The preset checks must be a list.';
  let shapes: Analysis['paramShapes'] | null | undefined;
  const out: Check[] = [];
  for (let i = 0; i < raw.length; i++) {
    const c: unknown = raw[i];
    const at = `Check ${i + 1} of the preset`;
    if (!isRecord(c)) return `${at} must be an object.`;
    const kind = c['kind'];
    if (kind !== 'lossBelow' && kind !== 'paramNear') {
      return `${at} has an unknown kind; use "lossBelow" or "paramNear".`;
    }
    let step: number | undefined;
    if (c['step'] !== undefined) {
      const s = c['step'];
      if (!Number.isInteger(s) || (s as number) < 0 || (s as number) >= stepCount) {
        return `${at} names a step that does not exist.`;
      }
      step = s as number;
    }
    const value = c['value'];
    if (!finite(value)) return `${at} needs a number "value".`;
    if (kind === 'lossBelow') {
      const split = c['split'];
      if (split !== undefined && split !== 'train' && split !== 'val') {
        return `${at} has a split that is not "train" or "val".`;
      }
      const chk: Check = { kind, value };
      if (split !== undefined) chk.split = split;
      if (step !== undefined) chk.step = step;
      out.push(chk);
      continue;
    }
    const rawNode = c['node'];
    if (typeof rawNode !== 'string') return `${at} needs the id of a piece in "node".`;
    const node: string = idMap ? (idMap[rawNode]?.[0] ?? rawNode) : rawNode;
    if (!graph.nodes.some((n) => n.id === node)) return `${at} names a piece that does not exist.`;
    const tol = c['tol'];
    if (!finite(tol) || tol < 0) return `${at} needs a "tol" that is a number, 0 or more.`;
    const tensor = c['tensor'];
    if (tensor !== undefined && tensor !== 'W' && tensor !== 'b') {
      return `${at} has a tensor that is not "W" or "b".`;
    }
    const index = c['index'];
    if (index !== undefined && !(Number.isInteger(index) && (index as number) >= 0)) {
      return `${at} needs an "index" that is a whole number, 0 or more.`;
    }
    shapes ??= shapesOf(graph);
    const own = shapes && Object.hasOwn(shapes, node) ? shapes[node] : undefined;
    if (shapes) {
      if (!own) return `${at} names a piece that has no parameters.`;
      const names = Object.keys(own);
      if (tensor !== undefined && !names.includes(tensor)) {
        return `${at} names a tensor that this piece does not have.`;
      }
      if (tensor === undefined && names.length > 1) {
        return `${at} must say which tensor, "W" or "b", because the piece has both.`;
      }
      const shape = own[tensor ?? (names[0] as string)];
      const size = shape ? shape.reduce((a, d) => a * d, 1) : 0;
      if (index !== undefined && size > 0 && (index as number) >= size) {
        return `${at} has an index beyond the size of that tensor.`;
      }
    }
    const chk: Check = { kind, node, value, tol };
    if (tensor !== undefined) chk.tensor = tensor;
    if (index !== undefined) chk.index = index as number;
    if (step !== undefined) chk.step = step;
    out.push(chk);
  }
  return out;
}

/** True exactly while the condition holds now (no latch). Missing or non-finite values: false (so `lossBelow` on `val` never holds when the validation fraction is 0). */
export function evaluateCheck(check: Check, input: CheckInput): boolean {
  if (check.kind === 'lossBelow') {
    const v = input.liveLoss?.[check.split ?? 'train'];
    return typeof v === 'number' && Number.isFinite(v) && v < check.value;
  }
  const layer =
    input.host && Object.hasOwn(input.host, check.node) ? input.host[check.node] : undefined;
  if (!layer) return false;
  let name: string | undefined = check.tensor;
  if (name === undefined) {
    const names = Object.keys(layer);
    if (names.length !== 1) return false;
    name = names[0];
  }
  const t = name !== undefined && Object.hasOwn(layer, name) ? layer[name] : undefined;
  const v = t?.[check.index ?? 0];
  return v !== undefined && Number.isFinite(v) && Math.abs(v - check.value) <= check.tol;
}

/**
 * Per-step results (a step with several checks holds only when all do) and the preset-level result
 * (null when there is no check without a step; otherwise all of those hold).
 */
export function evaluateChecks(
  checks: readonly Check[],
  input: CheckInput,
): { steps: Map<number, boolean>; general: boolean | null } {
  const steps = new Map<number, boolean>();
  let general: boolean | null = null;
  for (const c of checks) {
    const ok = evaluateCheck(c, input);
    if (c.step === undefined) general = (general ?? true) && ok;
    else steps.set(c.step, (steps.get(c.step) ?? true) && ok);
  }
  return { steps, general };
}
