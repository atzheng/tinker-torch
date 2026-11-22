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

/** Validates a preset's `checks`; returns the checks or one plain sentence. */
export function parseChecks(raw: unknown, graph: Graph, stepCount: number): Check[] | string {
  if (!Array.isArray(raw)) return 'The preset checks must be a list.';
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
    const node = c['node'];
    if (typeof node !== 'string') return `${at} needs the id of a piece in "node".`;
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
    const chk: Check = { kind, node, value, tol };
    if (tensor !== undefined) chk.tensor = tensor;
    if (index !== undefined) chk.index = index as number;
    if (step !== undefined) chk.step = step;
    out.push(chk);
  }
  return out;
}

/** True exactly while the condition holds now (no latch). Missing or non-finite values: false. */
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
