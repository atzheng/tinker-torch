// Predicted-versus-observed class bars (tinker-6t7.16): pure model. Nothing engine-side is kept.
import { probBarOf } from '../canvas/probBarPure.js';

export type ClassBar = {
  /** Class name (the data's `classNames`, else "class k"). */
  name: string;
  /** Mean predicted probability of the class over the examples. */
  predicted: number;
  /** Share of the examples whose label is the class. */
  observed: number;
};

/** Rows of the training set fed to `predict` (both columns use the same rows). */
export const CLASS_BARS_MAX_ROWS = 2000;

/**
 * The bars for one prediction readback `out` (`n` rows of `width` columns) and the labels `y`
 * (first `n` used). Width 1 is a sigmoid: P(class 1) = value, P(class 0) = 1 - value (two
 * classes). Width k is a softmax: one bar per column. Null when the shape does not fit the
 * kind, or a value is not a probability.
 */
export function classBars(
  out: Float32Array | undefined,
  n: number,
  width: number,
  y: ArrayLike<number>,
  numClasses: number,
  classNames?: readonly string[],
): ClassBar[] | null {
  if (n < 1 || y.length < n) return null;
  const bar = probBarOf(out, n, width);
  if (!bar) return null;
  let pred: number[];
  if (width === 1 && numClasses === 2) {
    const p1 = bar.values[0] as number;
    pred = [1 - p1, p1];
  } else if (width === numClasses && width >= 2) pred = bar.values;
  else return null;
  const counts = new Array<number>(numClasses).fill(0);
  for (let i = 0; i < n; i++) {
    const c = Math.round(y[i] as number);
    if (c >= 0 && c < numClasses) counts[c] = (counts[c] as number) + 1;
  }
  return pred.map((predicted, k) => ({
    name: classNames?.[k] ?? `class ${k}`,
    predicted,
    observed: (counts[k] as number) / n,
  }));
}
