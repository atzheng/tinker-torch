import { parseFile, type Graph } from '@tinker/core';
import mlp from '../src/presets/mlp.json';

/** The small sine network that was the default start before tinker-9io.8 (preset `mlp`). */
export function mlpGraph(): Graph {
  const r = parseFile(mlp);
  if (!r.ok) throw new Error(r.error);
  return r.file.graph;
}
