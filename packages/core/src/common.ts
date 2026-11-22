/** Pieces of the graph format shared by every graph version: data and training settings, ids, ports. */
import { z } from 'zod';

export const DATASET_NAMES = [
  'line',
  'quadratic',
  'sine',
  'step',
  'wiggle',
  'plane',
  'bowl',
  'saddle',
  'blobs',
  'circle',
  'xor',
  'moons',
  'delivery_late',
  'delivery_time',
  'delivery_lateness',
  'blobs3',
  'spiral3',
  'iris',
  'penguins',
  'wine',
  'diabetes',
  'csv',
] as const;
export const ACTIVATION_FNS = ['sigmoid', 'tanh', 'relu', 'gelu'] as const;
export const OPTIMIZERS = ['sgd', 'momentum', 'adam'] as const;

export const oneOf = (values: readonly string[]) => `must be one of ${values.join(', ')}`;
export const NODE_NAME = 'must be a node name (text)';

export function num(message: string, ok: (n: number) => boolean) {
  return z
    .number({ error: message })
    .refine((n) => Number.isFinite(n) && ok(n), { error: message });
}
export const int = (lo: number, hi: number) =>
  num(
    `must be a whole number from ${lo} to ${hi}`,
    (n) => Number.isInteger(n) && n >= lo && n <= hi,
  );
export const bool = z.boolean({ error: 'must be true or false' });

export const nodeId = z.string({ error: 'must be text' }).regex(/^[a-z][a-z0-9_]*$/, {
  error: 'must start with a lowercase letter and use only a-z, 0-9 and _',
});

export const edgeSchema = z.tuple(
  [z.string({ error: NODE_NAME }), z.string({ error: NODE_NAME })],
  { error: 'must be a pair of node names' },
);

export const dataSchema = z.strictObject({
  set: z.enum(DATASET_NAMES, { error: oneOf(DATASET_NAMES) }),
  n: int(10, 500),
  noise: num('must be a number from 0 to 1', (n) => n >= 0 && n <= 1),
  val: num('must be a number from 0 to 0.5', (n) => n >= 0 && n <= 0.5),
  seed: int(-2147483648, 4294967295),
  standardize: bool.default(false),
});

export const trainSchema = z.strictObject({
  optimizer: z.enum(OPTIMIZERS, { error: oneOf(OPTIMIZERS) }),
  lr: num('must be a number above 0', (n) => n > 0),
  batch: z.union([z.literal('full'), int(1, 1_000_000)], {
    error: 'must be "full" or a whole number of at least 1',
  }),
  seed: int(-2147483648, 4294967295),
});

export type NodeId = string;
export type ActivationFn = (typeof ACTIVATION_FNS)[number];
export type DatasetName = (typeof DATASET_NAMES)[number];
export type PortRef = string;
export type Edge = [from: PortRef, to: PortRef];
export type DataConfig = z.infer<typeof dataSchema>;
export type TrainConfig = z.infer<typeof trainSchema>;
/** What the dataset says about itself (CONTRACTS.md section 4). */
export type DataInfo = {
  /** Column identifiers: `x1`, `x2`, ... always (what Input and Column pieces refer to). */
  columns: string[];
  /** Display names of the columns, same order and length as `columns`; absent: show the identifiers. */
  featureNames?: string[];
  kind: 'regression' | 'binary' | 'multiclass';
  /** Regression 1, binary 2, multiclass k >= 3. */
  numClasses: number;
  /** Name of the target; `sample` always sets it ('y' for synthetic sets). */
  targetName?: string;
  /** Classification only: one name per class, in label order (`numClasses` entries). */
  classNames?: string[];
  /**
   * Per column (same order as `columns`): 'numeric' or 'categorical'. Absent: every column is numeric.
   * A categorical column holds the category's index (0..K-1) as a float in `x`; Standardise skips it.
   */
  columnKinds?: ColumnKind[];
  /** Category names by column id, for each categorical column (K = the list's length). */
  categories?: Record<string, string[]>;
};
export type ColumnKind = 'numeric' | 'categorical';
/** The numeric column ids of a dataset (all columns when none is numeric): what a new Input starts with. */
export const numericColumns = (info: DataInfo): string[] => {
  const nums = info.columns.filter((_, i) => info.columnKinds?.[i] !== 'categorical');
  return nums.length > 0 ? nums : [...info.columns];
};
/** Number of categories of a column id, or 0 when the column is numeric (or unknown). */
export const categoryCount = (info: Pick<DataInfo, 'categories'>, col: string): number =>
  info.categories?.[col]?.length ?? 0;
export type Endpoint = { node: NodeId; port: string };

/** Splits "node" or "node.port". `port` is null for a bare node id. */
export function parsePort(ref: PortRef): { node: NodeId; port: string | null } {
  const dot = ref.indexOf('.');
  if (dot < 0) return { node: ref, port: null };
  return { node: ref.slice(0, dot), port: ref.slice(dot + 1) };
}

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}
