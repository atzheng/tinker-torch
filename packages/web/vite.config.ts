import react from '@vitejs/plugin-react';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

/**
 * KaTeX's CSS lists woff2, woff and ttf for every font; browsers use woff2, so drop the other two
 * and keep ~800 kB of unused fonts out of dist.
 */
const katexWoff2Only: Plugin = {
  name: 'katex-woff2-only',
  enforce: 'pre',
  transform(code, id) {
    if (!/katex[\\/]dist[\\/]katex(\.min)?\.css/.test(id)) return null;
    return code.replace(
      /,url\([^)]*\.woff\) format\("woff"\),url\([^)]*\.ttf\) format\("truetype"\)/g,
      '',
    );
  },
};

export default defineConfig({
  plugins: [react(), katexWoff2Only],
  server: { port: 5173, strictPort: true },
  test: { include: ['test/**/*.test.{ts,tsx}', 'src/**/*.test.{ts,tsx}'] },
});
