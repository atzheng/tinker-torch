import { defineConfig } from 'vitest/config';

// Only used by `pnpm torch-smoke` (run from the repo root): the dump step runs as a vitest file so it
// can import the TypeScript sources of packages/core directly. Not part of any package's `test` script.
export default defineConfig({
  test: { include: ['tools/torch-smoke/dump.test.ts'] },
});
