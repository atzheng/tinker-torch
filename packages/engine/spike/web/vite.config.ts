// Vite config for the spike page. SPIKE_ISOLATED=1 serves it with COOP/COEP so that
// crossOriginIsolated and SharedArrayBuffer are available (multi-threaded Wasm).
// A plain object, no imports, so it needs no extra dependency in this package.
const isolated = process.env.SPIKE_ISOLATED === '1';

declare const process: { env: Record<string, string | undefined> };

export default {
  server: {
    port: 5199,
    strictPort: true,
    headers: isolated
      ? {
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        }
      : {},
  },
  optimizeDeps: { exclude: ['@jax-js/jax'] },
};
