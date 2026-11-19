/** Result of `biasVariance`. All values are averages over the grid points. */
export type BiasVariance = {
  /** Pointwise mean of the copies' predictions, length m. */
  mean: Float32Array;
  /** Mean over x of (mean prediction - truth)^2. */
  bias2: number;
  /** Mean over x of the population variance (divide by k) of the copies' predictions. */
  variance: number;
  /** bias2 + variance: the mean over x and copies of (prediction - truth)^2. */
  mse: number;
  /** Number of copies used. */
  k: number;
};

/**
 * Bias and variance of k fitted curves against the true function, on one shared grid of m points
 * (equally spaced over the plotted input range, so the plain mean over points is the average over
 * the range). `curves[i][j]` is copy i at grid point j; `truth[j]` is f*(x_j).
 *
 * With the population variance, bias2 + variance equals the mean squared error to the truth
 * exactly. Copies containing a non-finite value are skipped; with no usable copy every number is NaN.
 */
export function biasVariance(
  curves: readonly ArrayLike<number>[],
  truth: ArrayLike<number>,
): BiasVariance {
  const m = truth.length;
  const ok = curves.filter((c) => {
    if (c.length !== m) return false;
    for (let j = 0; j < m; j++) if (!Number.isFinite(c[j])) return false;
    return true;
  });
  const k = ok.length;
  const mean = new Float32Array(m);
  if (k === 0 || m === 0) {
    mean.fill(Number.NaN);
    return { mean, bias2: Number.NaN, variance: Number.NaN, mse: Number.NaN, k };
  }
  let bias2 = 0;
  let variance = 0;
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (const c of ok) s += c[j] ?? 0;
    const mu = s / k;
    let v = 0;
    for (const c of ok) v += ((c[j] ?? 0) - mu) ** 2;
    mean[j] = mu;
    bias2 += (mu - (truth[j] ?? 0)) ** 2;
    variance += v / k;
  }
  bias2 /= m;
  variance /= m;
  return { mean, bias2, variance, mse: bias2 + variance, k };
}
