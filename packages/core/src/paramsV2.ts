/** Host-side parameters for graph v2 (CONTRACTS.md section 5). Exported as `v2.initParams`. */
import type { Analysis } from './analyseV2.js';
import type { NodeId } from './common.js';
import type { HostParams } from './params.js';
import { initNode } from './params.js';

/**
 * nn.Linear-style init (W and b uniform on +/- 1/sqrt(in)) for every Linear in `analysis.paramNodes`
 * with a known shape, from `makeRng(seed).fork(nodeId)`: the same ids give the same values as
 * version 1, so migrated graphs start identically. A Linear read only through a Weights tap is live
 * and is initialised too (it is not in `analysis.order`).
 */
export function initParams(analysis: Analysis, seed: number): HostParams {
  const host = Object.create(null) as HostParams;
  for (const id of analysis.paramNodes) {
    const s = analysis.paramShapes[id as NodeId];
    if (!s?.W) continue;
    host[id] = initNode({ W: s.W, ...(s.b ? { b: s.b } : {}) }, seed, id);
  }
  return host;
}
