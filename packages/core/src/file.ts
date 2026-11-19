import { parseGraph, serializeGraph, type Graph, type NodeId } from './graph.js';

export type SaveFile = {
  kind: 'tinker';
  version: 2;
  graph: Graph;
  params?: Record<NodeId, { W: number[]; b?: number[] }>;
};
export type Preset = SaveFile & { title: string; prompt?: string; locked?: NodeId[] };

export type ParseFileResult = { ok: true; file: SaveFile | Preset } | { ok: false; error: string };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

const fail = (error: string): ParseFileResult => ({ ok: false, error });

/**
 * Validates a parsed JSON document as a Tinker file (CONTRACTS.md section 12; opens version 1 files by migrating their graph). Gives one plain
 * sentence on failure. `params` is only checked to be an object here: entries that do not fit the
 * graph are dropped later by `paramsFromJson`, with a notice, never an error.
 */
export function parseFile(json: unknown): ParseFileResult {
  if (!isRecord(json) || json['kind'] !== 'tinker') {
    return fail('This is not a Tinker file.');
  }
  if (json['version'] !== 1 && json['version'] !== 2) {
    return fail(
      `This file has version ${JSON.stringify(json['version']) ?? 'unknown'}, but this Tinker opens version 1 and 2 files.`,
    );
  }
  if (!('graph' in json)) return fail('This Tinker file has no graph.');
  const g = parseGraph(json['graph']);
  if (!g.ok) return fail(g.error);

  const params = json['params'];
  if (params !== undefined && !isRecord(params)) {
    return fail('The saved weights in this file must be an object.');
  }

  // A version 1 file is migrated by parseGraph; what we hold is always a version 2 file.
  const file: SaveFile = { kind: 'tinker', version: 2, graph: g.graph };
  if (params !== undefined) file.params = params as SaveFile['params'];

  if ('title' in json || 'prompt' in json || 'locked' in json) {
    const { title, prompt, locked } = json;
    if (typeof title !== 'string') return fail('The preset title must be text.');
    if (prompt !== undefined && typeof prompt !== 'string') {
      return fail('The preset prompt must be text.');
    }
    if (
      locked !== undefined &&
      !(Array.isArray(locked) && locked.every((x) => typeof x === 'string'))
    ) {
      return fail('The preset list of locked pieces must be a list of piece ids.');
    }
    const preset: Preset = { ...file, title };
    if (prompt !== undefined) preset.prompt = prompt;
    if (locked !== undefined) preset.locked = locked as string[];
    return { ok: true, file: preset };
  }
  return { ok: true, file };
}

/** Parses the text of a file: bad JSON gives a sentence like any other failure. */
export function parseFileText(text: string): ParseFileResult {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return fail('This file is not valid JSON.');
  }
  return parseFile(json);
}

/** Two-space JSON with the graph in its stable key order; the same inputs always give the same text. */
export function serializeFile(
  graph: Graph,
  params?: Record<NodeId, { W: number[]; b?: number[] }>,
): string {
  const doc: Record<string, unknown> = {
    kind: 'tinker',
    version: 2,
    graph: JSON.parse(serializeGraph(graph)) as unknown,
  };
  if (params) doc['params'] = params;
  return JSON.stringify(doc, null, 2) + '\n';
}
