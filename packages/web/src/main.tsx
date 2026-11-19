import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.js';
import { loadBundled } from './topbar/PresetMenu.js';
import './theme.css';
import 'katex/dist/katex.min.css';
import './tex/tex.css';

if (import.meta.env.DEV) {
  // Dev only: lets browser checks read the store (window.__tinker.store) next to liveArrays.
  void import('./state/store.js').then((m) => {
    const w = window as unknown as { __tinker?: Record<string, unknown> };
    w.__tinker = { ...w.__tinker, store: m.useStore };
  });
}

// `?preset=<name>` loads a bundled preset on start (an unknown name is ignored).
const presetName = new URLSearchParams(window.location.search).get('preset');
if (presetName) loadBundled(presetName);

const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
