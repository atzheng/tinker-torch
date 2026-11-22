import { parseGraphWithMap, serializeGraph, type Graph, type NodeId } from './graph.js';
import type { JsonParams } from './params.js';
import { PALETTE } from './registry.js';

export type SaveFile = {
  kind: 'tinker';
  version: 2;
  graph: Graph;
  params?: JsonParams;
};
/**
 * Progressive disclosure (CONTRACTS.md section 12): what a preset shows. Allow-lists; an absent list
 * shows everything in its category. Hiding changes visibility only, never behaviour.
 */
export type Show = {
  /** Palette entry keys (`PALETTE`) and palette group ids (`PALETTE_GROUP_IDS`). Empty: no palette. */
  palette?: string[];
  /** Control and panel ids (`UI_IDS`). A region with none of its ids listed is removed. */
  ui?: string[];
};
export type Preset = SaveFile & {
  title: string;
  prompt?: string;
  locked?: NodeId[];
  show?: Show;
};

/** The palette's group ids, in palette order (the web palette's `GROUPS` uses exactly these). */
export const PALETTE_GROUP_IDS = [
  'inputs',
  'features',
  'linear',
  'activations',
  'structure',
  'links',
  'math',
  'errors',
  'reductions',
  'losses',
  'scalars',
  'parameters',
  'ready',
] as const;

/**
 * Every UI control and panel a preset can show or hide, by region. A region is on screen only when
 * at least one of its ids is shown. Ids are the controls' `data-testid` where one exists. Later
 * panels and controls add their ids here.
 */
export const UI_IDS = {
  'top-bar': [
    'data-set',
    'data-n',
    'data-noise',
    'data-val',
    'data-seed',
    'data-standardize',
    'data-open-csv',
    'data-sources',
    'data-resample',
    'file-menu',
    'preset-menu',
    'btn-undo',
    'btn-redo',
  ],
  canvas: [
    'grain-toggle',
    'fit-view',
    'tidy-layout',
    'reset-view',
    'equation-strip',
    'group-linear',
    'expand-scalars',
    'equivalence',
    'preset-prompt',
  ],
  results: ['panel-fit', 'panel-loss', 'panel-surface'],
  // Controls inside the results panels; they never make the results region itself visible.
  'panel-controls': [
    'truth-toggle',
    'resample-run',
    'prediction-picker',
    'loss-readout',
    'loss-log-toggle',
    'loss-restore-best',
    'pins',
  ],
  'training-bar': [
    'btn-step',
    'btn-run',
    'btn-reset',
    'btn-restore-best',
    'lr-slider',
    'speed',
    'optimizer',
    'batch',
    'step-count',
    'param-count',
  ],
} as const;

export type UiRegion = keyof typeof UI_IDS;
export const ALL_UI_IDS: readonly string[] = Object.values(UI_IDS).flat();

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
  const g = parseGraphWithMap(json['graph']);
  if (!g.ok) return fail(g.error);

  const params = json['params'];
  if (params !== undefined && !isRecord(params)) {
    return fail('The saved weights in this file must be an object.');
  }

  // A version 1 file is migrated by parseGraph; what we hold is always a version 2 file.
  const file: SaveFile = { kind: 'tinker', version: 2, graph: g.graph };
  if (params !== undefined) file.params = params as SaveFile['params'];

  if ('title' in json || 'prompt' in json || 'locked' in json || 'show' in json) {
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
    const shown = parseShow(json['show']);
    if (typeof shown === 'string') return fail(shown);
    const preset: Preset = { ...file, title };
    if (shown) preset.show = shown;
    if (prompt !== undefined) preset.prompt = prompt;
    if (locked !== undefined) {
      const ids = locked as string[];
      const map = g.idMap;
      // A migrated graph: a locked v1 id covers every node it became (none if it was spliced out).
      preset.locked = map ? [...new Set(ids.flatMap((id) => map[id] ?? []))] : ids;
    }
    return { ok: true, file: preset };
  }
  return { ok: true, file };
}

const PALETTE_IDS = new Set<string>([...PALETTE.map((e) => e.key), ...PALETTE_GROUP_IDS]);
const UI_ID_SET = new Set<string>(ALL_UI_IDS);

/** The preset's `show` field, or one plain sentence when it is malformed or names an unknown id. */
function parseShow(v: unknown): Show | string | undefined {
  if (v === undefined) return undefined;
  if (!isRecord(v)) return 'The preset "show" must be an object with "palette" and "ui" lists.';
  const out: Show = {};
  for (const [key, known, what] of [
    ['palette', PALETTE_IDS, 'a palette piece or group'],
    ['ui', UI_ID_SET, 'a control or panel'],
  ] as const) {
    const list = v[key];
    if (list === undefined) continue;
    if (!(Array.isArray(list) && list.every((x) => typeof x === 'string'))) {
      return `The preset "show.${key}" must be a list of ids.`;
    }
    for (const id of list as string[]) {
      if (!known.has(id)) return `The preset "show.${key}" names "${id}", which is not ${what}.`;
    }
    out[key] = [...new Set(list as string[])];
  }
  for (const k of Object.keys(v)) {
    if (k !== 'palette' && k !== 'ui') {
      return `The preset "show" has "${k}", but only "palette" and "ui" are known.`;
    }
  }
  return out;
}

/** True when `id` passes an allow-list; an absent list shows everything. */
export function isShown(list: readonly string[] | undefined, id: string): boolean {
  return list === undefined || list.includes(id);
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
export function serializeFile(graph: Graph, params?: JsonParams): string {
  const doc: Record<string, unknown> = {
    kind: 'tinker',
    version: 2,
    graph: JSON.parse(serializeGraph(graph)) as unknown,
  };
  if (params) doc['params'] = params;
  return JSON.stringify(doc, null, 2) + '\n';
}
