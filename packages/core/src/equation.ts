/**
 * The model as equations (CONTRACTS.md section 3, "Equations"). Pure: TeX strings, no DOM, no jax.
 *
 * `modelEquations` walks from each ŷ pin and from L through `analysis.order` / `analysis.inputs`
 * and writes the pieces as notation. A non-trivial piece with more than one consumer gets its own
 * definition line (`h_k = ...`, kind 'def') and is referenced by name; a ŷ node is written
 * `\hat{y}` wherever another line uses it. Under a Mean, per-example names carry the index `i`.
 * A line that does not resolve is omitted. `numeric` repeats the lines with live values.
 */
import type { Analysis } from './analyse.js';
import { parsePort, type DataInfo, type Graph, type Node, type NodeId } from './graph.js';
import type { HostParams } from './params.js';

export type EquationLine = { lhs: string; rhs: string; kind: 'pred' | 'loss' | 'def' };
export type ModelEquations = { lines: EquationLine[]; numeric?: EquationLine[] };

/** TeX (no dollars) of a Weight's name: w1 is `w_{1}`, any other id is set upright. */
export function weightTex(id: NodeId): string {
  const m = /^w(\d+)$/.exec(id);
  if (m) return `w_{${m[1]}}`;
  return id === 'w' ? 'w' : `\\mathrm{${id.replace(/_/g, '\\_')}}`;
}

/** A weight value as shown in its pill: two decimals, fewer as it grows (the limit is 100). */
export function fmtWeight(v: number): string {
  const a = Math.abs(v);
  return a < 10 ? v.toFixed(2) : a < 100 ? v.toFixed(1) : v.toFixed(0);
}

/** Numeric lines appear when every live parameter is a scalar piece or there are at most this many. */
export const NUMERIC_MAX_PARAMS = 8;

const ATOM = 3;
const PROD = 2;
const SUM = 1;
type Ex = { tex: string; prec: number };
class Unresolved extends Error {}

const ATOMIC: ReadonlySet<string> = new Set(['input', 'column', 'target', 'weights']);

const paren = (e: Ex, min: number): string => (e.prec >= min ? e.tex : `\\left(${e.tex}\\right)`);
const mul = (a: string, b: string): string =>
  /^[\d.-]/.test(b) ? `${a} \\cdot ${b}` : `${a}\\,${b}`;
/** Joins terms with their signs folded: `a`, `-0.30` give `a - 0.30`. */
const terms = (ts: string[]): string =>
  ts.reduce(
    (acc, t, i) => (i === 0 ? t : t.startsWith('-') ? `${acc} - ${t.slice(1)}` : `${acc} + ${t}`),
    '',
  );
const colNumber = (name: string, data: DataInfo): string | null => {
  const m = /^x(\d+)$/.exec(name);
  if (m) return m[1] as string;
  const i = data.columns.indexOf(name);
  return i >= 0 ? String(i + 1) : null;
};
/** `fmtWeight` for equations: a value that rounds to zero prints `0.00`, never `-0.00`. */
const fmtEq = (v: number): string => {
  const t = fmtWeight(v);
  return /^-0(\.0+)?$/.test(t) ? t.slice(1) : t;
};
const short = (c: number): string => String(Number(c.toPrecision(3)));

/** Whether numeric lines apply: every live parameter is a scalar piece, or there are at most `NUMERIC_MAX_PARAMS`. */
export function wantsNumeric(graph: Graph, analysis: Analysis): boolean {
  if (analysis.paramNodes.length === 0) return false;
  const type = new Map(graph.nodes.map((n) => [n.id, n.type]));
  const onlyScalars = analysis.paramNodes.every((id) => {
    const t = type.get(id);
    return t === 'scalew' || t === 'bias';
  });
  return onlyScalars || analysis.totalParams <= NUMERIC_MAX_PARAMS;
}

/** The numeric lines worth showing: those that differ from the symbolic line with the same lhs. */
export function numericExtras(eq: ModelEquations): EquationLine[] {
  return (eq.numeric ?? []).filter(
    (l) => l.rhs !== eq.lines.find((x) => x.lhs === l.lhs && x.kind === l.kind)?.rhs,
  );
}

/** Never throws: a graph that does not resolve gives fewer (or no) lines. */
export function modelEquations(
  graph: Graph,
  analysis: Analysis,
  data: DataInfo,
  host?: HostParams,
): ModelEquations {
  try {
    return build(graph, analysis, data, host);
  } catch {
    return { lines: [] };
  }
}

function build(
  graph: Graph,
  analysis: Analysis,
  data: DataInfo,
  host?: HostParams,
): ModelEquations {
  const byId = new Map<NodeId, Node>();
  for (const n of graph.nodes) if (!byId.has(n.id)) byId.set(n.id, n);
  const inOrder = new Set(analysis.order);
  const live = new Set<NodeId>([...analysis.order, ...analysis.paramNodes]);

  // Names: the k-th live Linear is W_k, b_k; scalar Bias pieces count separately.
  const linearK = new Map<NodeId, number>();
  const biasK = new Map<NodeId, number>();
  for (const n of byId.values()) {
    if (!live.has(n.id)) continue;
    if (n.type === 'linear') linearK.set(n.id, linearK.size + 1);
    else if (n.type === 'bias') biasK.set(n.id, biasK.size + 1);
  }

  const preds = graph.predictions.map((ref) => parsePort(ref).node);
  const predIndex = new Map<NodeId, number>();
  preds.forEach((id, i) => {
    if (!predIndex.has(id)) predIndex.set(id, i + 1);
  });
  const multi = graph.predictions.length > 1;

  const consumers = new Map<NodeId, number>();
  for (const id of analysis.order) {
    for (const list of Object.values(analysis.inputs[id] ?? {})) {
      for (const w of list) consumers.set(w.from, (consumers.get(w.from) ?? 0) + 1);
    }
  }
  const unwiredBias = (n: Node) => n.type === 'bias' && !analysis.inputs[n.id]?.['in']?.length;
  const defK = new Map<NodeId, number>();
  for (const id of analysis.order) {
    const n = byId.get(id);
    if (!n || ATOMIC.has(n.type) || unwiredBias(n) || predIndex.has(id)) continue;
    if ((consumers.get(id) ?? 0) > 1) defK.set(id, defK.size + 1);
  }

  const gen = (numeric: boolean): EquationLine[] => {
    const used = new Set<NodeId>();
    const visiting = new Set<NodeId>();
    const sub = (base: string, s: string | undefined, idx: boolean): string =>
      s === undefined ? (idx ? `${base}_{i}` : base) : idx ? `${base}_{${s},i}` : `${base}_{${s}}`;

    const num = (v: number | undefined): Ex => {
      if (v === undefined || !Number.isFinite(v)) throw new Unresolved();
      const t = fmtEq(v);
      return { tex: t, prec: t.startsWith('-') ? SUM : ATOM };
    };
    const matrix = (vals: number[], rows: number, cols: number): Ex => {
      if (vals.some((v) => !Number.isFinite(v))) throw new Unresolved();
      const rowsTex: string[] = [];
      for (let r = 0; r < rows; r++) {
        const cells: string[] = [];
        for (let c = 0; c < cols; c++) cells.push(fmtEq(vals[r * cols + c] as number));
        rowsTex.push(cells.join(' & '));
      }
      return { tex: `\\begin{bmatrix}${rowsTex.join(' \\\\ ')}\\end{bmatrix}`, prec: ATOM };
    };
    /** W of a Linear as out x in (stored `W[in][out]`). */
    const wValues = (id: NodeId): { vals: number[]; rows: number; cols: number } => {
      const shape = analysis.paramShapes[id]?.['W'];
      const W = host?.[id]?.['W'];
      if (!shape || !W || shape.length < 2) throw new Unresolved();
      const [nin, nout] = shape as [number, number];
      const vals: number[] = [];
      for (let o = 0; o < nout; o++)
        for (let i = 0; i < nin; i++) vals.push(W[i * nout + o] ?? NaN);
      return { vals, rows: nout, cols: nin };
    };
    const wMat = (id: NodeId): Ex => {
      const { vals, rows, cols } = wValues(id);
      return rows === 1 && cols === 1 ? num(vals[0]) : matrix(vals, rows, cols);
    };
    const bVec = (id: NodeId): Ex => {
      const b = host?.[id]?.['b'];
      if (!b) throw new Unresolved();
      return b.length === 1 ? num(b[0]) : matrix(Array.from(b), b.length, 1);
    };

    const inputOf = (id: NodeId, port: string): NodeId => {
      const w = analysis.inputs[id]?.[port]?.[0];
      if (!w) throw new Unresolved();
      return w.from;
    };

    const expr = (id: NodeId, idx: boolean, root: boolean): Ex => {
      const n = byId.get(id);
      if (!n || !inOrder.has(id)) throw new Unresolved();
      if (!root) {
        const p = predIndex.get(id);
        if (p !== undefined) {
          return { tex: sub('\\hat{y}', multi ? String(p) : undefined, idx), prec: ATOM };
        }
        const d = defK.get(id);
        if (d !== undefined) {
          used.add(id);
          return { tex: sub('h', String(d), idx), prec: ATOM };
        }
      }
      if (visiting.has(id)) throw new Unresolved();
      visiting.add(id);
      try {
        return expand(n, idx);
      } finally {
        visiting.delete(id);
      }
    };

    const expand = (n: Node, idx: boolean): Ex => {
      const one = (port = 'in', i = idx) => expr(inputOf(n.id, port), i, false);
      const fn = (name: string): Ex => ({
        tex: `${name}\\left(${one().tex}\\right)`,
        prec: ATOM,
      });
      switch (n.type) {
        case 'input': {
          const nums = n.columns.map((c) => colNumber(c, data));
          if (nums.some((x) => x === null)) throw new Unresolved();
          if (n.columns.length === data.columns.length) {
            return { tex: sub('x', undefined, idx), prec: ATOM };
          }
          const ks = nums.map(Number);
          const contiguous = ks.every((k, i) => i === 0 || k === (ks[i - 1] as number) + 1);
          const s =
            ks.length === 1
              ? String(ks[0])
              : contiguous
                ? `[${ks[0]},${ks[ks.length - 1]}]`
                : `\\{${ks.join(',')}\\}`;
          return { tex: sub('x', s, idx), prec: ATOM };
        }
        case 'column': {
          const k = colNumber(n.name, data);
          if (k === null) throw new Unresolved();
          return { tex: sub('x', k, idx), prec: ATOM };
        }
        case 'target':
          return { tex: sub('y', undefined, idx), prec: ATOM };
        case 'linear': {
          const k = linearK.get(n.id);
          if (k === undefined) throw new Unresolved();
          const h = one();
          const W = numeric ? wMat(n.id) : { tex: `\\mathbf{W}_{${k}}`, prec: ATOM };
          const wh = mul(W.tex, paren(h, ATOM));
          if (!n.bias) return { tex: wh, prec: PROD };
          const b = numeric ? bVec(n.id) : { tex: `\\mathbf{b}_{${k}}`, prec: ATOM };
          return { tex: terms([wh, b.tex]), prec: SUM };
        }
        case 'scalew': {
          const h = one();
          const w = numeric ? num(host?.[n.id]?.['W']?.[0]) : { tex: weightTex(n.id), prec: ATOM };
          return { tex: mul(w.tex, paren(h, ATOM)), prec: PROD };
        }
        case 'bias': {
          const k = biasK.get(n.id);
          if (k === undefined) throw new Unresolved();
          const b = numeric ? num(host?.[n.id]?.['b']?.[0]) : { tex: `b_{${k}}`, prec: ATOM };
          if (unwiredBias(n)) return b;
          return { tex: terms([one().tex, b.tex]), prec: SUM };
        }
        case 'sum': {
          const ws = analysis.inputs[n.id]?.['in'];
          if (!ws?.length) throw new Unresolved();
          return { tex: terms(ws.map((w) => expr(w.from, idx, false).tex)), prec: SUM };
        }
        case 'add':
          return { tex: terms([one('a').tex, one('b').tex]), prec: SUM };
        case 'subtract': {
          const a = one('a');
          const b = one('b');
          return { tex: `${a.tex} - ${paren(b, PROD)}`, prec: SUM };
        }
        case 'activation':
          return fn(
            {
              sigmoid: '\\sigma',
              tanh: '\\tanh',
              relu: '\\mathrm{ReLU}',
              gelu: '\\mathrm{GELU}',
            }[n.fn],
          );
        case 'link':
          return fn(n.fn === 'sigmoid' ? '\\sigma' : '\\mathrm{softmax}');
        case 'feature':
          return fn('\\phi');
        case 'onehot':
          return fn('\\mathrm{onehot}');
        case 'lookup':
          return fn('\\mathrm{lookup}');
        case 'dropout':
          return fn(`\\mathrm{drop}_{${short(n.p)}}`);
        case 'scale':
          return { tex: `${short(n.c)} \\cdot ${paren(one(), ATOM)}`, prec: PROD };
        case 'square':
          return { tex: `${paren(one(), ATOM)}^2`, prec: ATOM };
        case 'abs':
          return { tex: `\\left|${one().tex}\\right|`, prec: ATOM };
        case 'huber':
          return fn('H_{\\delta}');
        case 'log_loss':
          return {
            tex: `\\mathrm{BCE}\\left(${one('pred').tex}, ${one('target').tex}\\right)`,
            prec: ATOM,
          };
        case 'cross_entropy':
          return {
            tex: `\\mathrm{CE}\\left(${one('pred').tex}, ${one('target').tex}\\right)`,
            prec: ATOM,
          };
        case 'mean':
          return { tex: `\\frac{1}{n}\\sum_i ${paren(one('in', true), PROD)}`, prec: PROD };
        case 'weights': {
          const k = linearK.get(n.of);
          if (k === undefined) throw new Unresolved();
          return numeric ? wMat(n.of) : { tex: `\\mathbf{W}_{${k}}`, prec: ATOM };
        }
        case 'norm': {
          const src = byId.get(inputOf(n.id, 'in'));
          if (src?.type !== 'weights') throw new Unresolved();
          const k = linearK.get(src.of);
          if (k === undefined) throw new Unresolved();
          if (numeric) {
            const { vals } = wValues(src.of);
            return num(
              n.kind === 'l2'
                ? vals.reduce((s, x) => s + x * x, 0)
                : vals.reduce((s, x) => s + Math.abs(x), 0),
            );
          }
          return {
            tex: n.kind === 'l2' ? `\\|\\mathbf{W}_{${k}}\\|_2^2` : `\\|\\mathbf{W}_{${k}}\\|_1`,
            prec: ATOM,
          };
        }
      }
    };

    const line = (lhs: string, id: NodeId, kind: EquationLine['kind']): EquationLine | null => {
      try {
        return { lhs, rhs: expr(id, false, true).tex, kind };
      } catch (e) {
        if (e instanceof Unresolved) return null;
        throw e;
      }
    };

    const out: EquationLine[] = [];
    preds.forEach((id, i) => {
      const l = line(multi ? `\\hat{y}_{${i + 1}}` : '\\hat{y}', id, 'pred');
      if (l) out.push(l);
    });
    if (graph.loss !== null) {
      const l = line('L', parsePort(graph.loss).node, 'loss');
      if (l) out.push(l);
    }
    // Definitions, earliest first; a definition can pull in earlier ones.
    const defs: EquationLine[] = [];
    for (let i = analysis.order.length - 1; i >= 0; i--) {
      const id = analysis.order[i] as NodeId;
      if (!used.has(id)) continue;
      const l = line(sub('h', String(defK.get(id)), false), id, 'def');
      if (l) defs.unshift(l);
    }
    return [...defs, ...out];
  };

  const lines = gen(false);
  if (host !== undefined && wantsNumeric(graph, analysis)) {
    try {
      const numeric = gen(true);
      if (numeric.length === lines.length) return { lines, numeric };
    } catch {
      /* values missing: no numeric lines */
    }
  }
  return { lines };
}
