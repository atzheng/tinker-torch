import { parseGraphWithMap, serializeGraph, type Graph, type NodeId } from './graph.js';
import { parseChecks, type Check } from './checks.js';
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
  /** Numbered exercise steps for the persistent instructions panel (TeX as in `prompt`). */
  instructions?: string[];
  /** Success checks (CONTRACTS.md section 12); each may name an instructions step. */
  checks?: Check[];
  locked?: NodeId[];
  show?: Show;
  /** Label overrides by `UI_IDS` id (CONTRACTS.md section 12). Plain text, 1 to `LABEL_MAX_LENGTH` characters. */
  labels?: Record<string, string>;
  /** Initial state of panel option controls (CONTRACTS.md section 12): `OPTION_BOOL_IDS` take true/false, `loss-mode` a mode name. */
  options?: Record<string, boolean | string>;
};

/** Panel toggles a preset's `options` can set (they are `UI_IDS` ids; the value is the initial checked state). */
export const OPTION_BOOL_IDS = [
  'histogram-squares',
  'loss-1d-tangent',
  'loss-1d-explored',
  'loss-log-toggle',
  'truth-toggle',
] as const;
/** The loss readout mode option key and its values. */
export const OPTION_LOSS_MODE = 'loss-mode';
export const OPTION_LOSS_MODES = ['likelihood', 'total', 'mean'] as const;

/** Longest label override, in characters: the training bar still has to fit 1024 px. */
export const LABEL_MAX_LENGTH = 40;

/** The palette's group ids, in palette order (the web palette's `GROUPS` uses exactly these). */
export const PALETTE_GROUP_IDS = [
  'inputs',
  'categories',
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
    'instructions',
    'success-check',
  ],
  results: [
    'panel-fit',
    'panel-loss',
    'panel-surface',
    'panel-histogram',
    'panel-loss-1d',
    'panel-examples',
    'panel-step-log',
  ],
  // Controls inside the results panels; they never make the results region itself visible.
  'panel-controls': [
    'truth-toggle',
    'resample-run',
    'prediction-picker',
    'loss-readout',
    'loss-mode-likelihood',
    'loss-mode-total',
    'loss-mode-mean',
    'loss-log-toggle',
    'loss-restore-best',
    'loss-1d-axis',
    'loss-1d-explored',
    'loss-1d-tangent',
    'pins',
    'histogram-squares',
  ],
  'training-bar': [
    'btn-step',
    'btn-run',
    'btn-reset',
    'btn-random-start',
    'btn-restore-best',
    'lr-slider',
    'speed',
    'optimizer',
    'batch',
    'step-count',
    'param-count',
  ],
} as const;

/**
 * The ids whose text a preset can override with `labels` (CONTRACTS.md section 12): every control
 * with a visible or accessible name that the web app reads through `useUiLabel`, and the panel
 * titles. Add an id here in the same change that wires it up.
 */
export const LABEL_IDS: readonly string[] = [
  ...UI_IDS['training-bar'],
  'data-set',
  'data-n',
  'data-noise',
  'data-val',
  'data-seed',
  'data-standardize',
  'data-open-csv',
  'data-sources',
  'data-resample',
  'grain-toggle',
  'fit-view',
  'tidy-layout',
  'reset-view',
  ...UI_IDS.results,
  'truth-toggle',
  'loss-readout',
  'loss-mode-likelihood',
  'loss-mode-total',
  'loss-mode-mean',
  'loss-log-toggle',
  'loss-restore-best',
];
const LABEL_ID_SET = new Set<string>(LABEL_IDS);

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

  let params = json['params'];
  if (params !== undefined && !isRecord(params)) {
    return fail('The saved weights in this file must be an object.');
  }
  if (params !== undefined) {
    const norm = normaliseParams(params as Record<string, unknown>, g.graph);
    if (typeof norm === 'string') return fail(norm);
    params = norm;
  }

  // A version 1 file is migrated by parseGraph; what we hold is always a version 2 file.
  const file: SaveFile = { kind: 'tinker', version: 2, graph: g.graph };
  if (params !== undefined) file.params = params as SaveFile['params'];

  if (
    'title' in json ||
    'prompt' in json ||
    'instructions' in json ||
    'checks' in json ||
    'locked' in json ||
    'show' in json ||
    'labels' in json ||
    'options' in json
  ) {
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
    const instr = json['instructions'];
    if (instr !== undefined) {
      if (!Array.isArray(instr)) {
        return fail('The preset instructions must be a list of steps, each one a piece of text.');
      }
      if (!instr.every((x) => typeof x === 'string')) {
        return fail('Each step of the preset instructions must be text.');
      }
      if (instr.some((x) => (x as string).trim() === '')) {
        return fail('A step of the preset instructions is empty.');
      }
    }
    let checks: Check[] | undefined;
    if (json['checks'] !== undefined) {
      const c = parseChecks(
        json['checks'],
        g.graph,
        Array.isArray(instr) ? instr.length : 0,
        g.idMap,
      );
      if (typeof c === 'string') return fail(c);
      checks = c;
    }
    const shown = parseShow(json['show']);
    if (typeof shown === 'string') return fail(shown);
    const labels = parseLabels(json['labels']);
    if (typeof labels === 'string') return fail(labels);
    const options = parseOptions(json['options']);
    if (typeof options === 'string') return fail(options);
    const preset: Preset = { ...file, title };
    if (shown) preset.show = shown;
    if (options) preset.options = options;
    if (labels) preset.labels = labels;
    if (prompt !== undefined) preset.prompt = prompt;
    if (instr !== undefined) preset.instructions = instr as string[];
    if (checks !== undefined) preset.checks = checks;
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

/**
 * Scalar shorthand (CONTRACTS.md section 12): a plain number for a Weight (`"w1": 0.5` is
 * `{ "W": [0.5] }`) or a Bias (`"b": 0` is `{ "b": [0] }`). Anything else is copied as it is (the
 * {W, b} arrays are checked against the graph later, by `paramsFromJson`, which drops what does not
 * fit). Returns one plain sentence for a shorthand that cannot mean anything.
 */
function normaliseParams(
  params: Record<string, unknown>,
  graph: Graph,
): Record<string, unknown> | string {
  const out: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const id of Object.keys(params)) {
    const v = params[id];
    if (typeof v !== 'number') {
      out[id] = v;
      continue;
    }
    const node = graph.nodes.find((n) => n.id === id);
    if (!node) return `The saved weights name "${id}", which is not a piece in this graph.`;
    if (node.type !== 'scalew' && node.type !== 'bias') {
      return `The saved weights give "${id}" a single number, but only a Weight or a Bias takes one; use lists such as {"W": [...], "b": [...]}.`;
    }
    if (!Number.isFinite(v)) return `The saved value for "${id}" must be a finite number.`;
    out[id] = node.type === 'scalew' ? { W: [v] } : { b: [v] };
  }
  return out;
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

/** The preset's `labels` field, or one plain sentence when it is malformed. */
function parseLabels(v: unknown): Record<string, string> | string | undefined {
  if (v === undefined) return undefined;
  if (!isRecord(v)) return 'The preset "labels" must be an object from control ids to text.';
  const out: Record<string, string> = Object.create(null) as Record<string, string>;
  for (const [id, text] of Object.entries(v)) {
    if (!UI_ID_SET.has(id)) {
      return `The preset "labels" names "${id}", which is not a control or panel.`;
    }
    if (!LABEL_ID_SET.has(id)) return `The preset "labels" cannot rename "${id}".`;
    if (typeof text !== 'string') return `The label for "${id}" must be text.`;
    if (text.trim() === '') return `The label for "${id}" is empty.`;
    if (text.length > LABEL_MAX_LENGTH) {
      return `The label for "${id}" is longer than ${LABEL_MAX_LENGTH} characters.`;
    }
    out[id] = text;
  }
  return { ...out };
}

/** The preset's `options` field, or one plain sentence when it is malformed. */
function parseOptions(v: unknown): Record<string, boolean | string> | string | undefined {
  if (v === undefined) return undefined;
  if (!isRecord(v)) return 'The preset "options" must be an object from option names to values.';
  const out: Record<string, boolean | string> = Object.create(null) as Record<
    string,
    boolean | string
  >;
  for (const [id, value] of Object.entries(v)) {
    if ((OPTION_BOOL_IDS as readonly string[]).includes(id)) {
      if (typeof value !== 'boolean') return `The option "${id}" must be true or false.`;
    } else if (id === OPTION_LOSS_MODE) {
      if (!(OPTION_LOSS_MODES as readonly string[]).includes(value as string)) {
        return `The option "loss-mode" must be one of ${OPTION_LOSS_MODES.map((m) => `"${m}"`).join(', ')}.`;
      }
    } else {
      return `The preset "options" names "${id}", which is not an option a preset can set.`;
    }
    out[id] = value as boolean | string;
  }
  return { ...out };
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
