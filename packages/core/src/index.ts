export const CORE_NAME = '@tinker/core';

// Graph v1 is what analyse, edit, layout, params, the engine and the web app still consume. It goes
// when those move to v2 (CONTRACTS.md section 2); v2 is available as the `v2` namespace meanwhile.
export * from './legacy/graph.js';
export * from './legacy/registry.js';
export * as v2 from './v2.js';
export * from './rng.js';
export * from './data/index.js';
export * from './analyse.js';
export * from './edit.js';
export * from './layout.js';
export * from './params.js';
export * from './legacy/file.js';
export * from './stats.js';
